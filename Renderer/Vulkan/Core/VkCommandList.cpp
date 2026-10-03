#include "Renderer/Vulkan/Core/VkCommandList.h"

#include "Renderer/Vulkan/Core/VkConversions.h"
#include "Renderer/Vulkan/Core/VkPipeline.h"
#include "Renderer/Vulkan/Resources/AccelerationStructure.h"
#include "Renderer/Vulkan/Resources/Buffer.h"
#include "Renderer/Vulkan/Resources/DescriptorSet.h"
#include "Renderer/Vulkan/Resources/Texture.h"
#include "Utilities/Assert.h"

#include <cstring>
#include <utility>
#include <vector>

namespace mukki::vulkan
{
namespace
{

vk::ImageMemoryBarrier2 MakeImageBarrier(
    const vk::Image& image,
    vk::ImageLayout oldLayout,
    vk::ImageLayout newLayout,
    vk::ImageAspectFlags aspect,
    vk::PipelineStageFlags2 srcStage,
    vk::AccessFlags2 srcAccess,
    vk::PipelineStageFlags2 dstStage,
    vk::AccessFlags2 dstAccess,
    std::uint32_t layerCount = 1)
{
    vk::ImageMemoryBarrier2 barrier{};
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.image = image;
    barrier.subresourceRange = vk::ImageSubresourceRange{}
                                   .setAspectMask(aspect)
                                   .setBaseMipLevel(0)
                                   .setLevelCount(1)
                                   .setBaseArrayLayer(0)
                                   .setLayerCount(layerCount);
    return barrier;
}

} // namespace

void VkCommandList::Init(const vk::raii::Device& device, std::uint32_t queueFamilyIndex, bool secondary)
{
    m_secondary = secondary;

    const vk::CommandPoolCreateInfo poolInfo = vk::CommandPoolCreateInfo{}
                                                   .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                                                   .setQueueFamilyIndex(queueFamilyIndex);
    m_commandPool = vk::raii::CommandPool{device, poolInfo};

    const vk::CommandBufferAllocateInfo allocateInfo = vk::CommandBufferAllocateInfo{}
                                                           .setCommandPool(*m_commandPool)
                                                           .setLevel(
                                                               secondary ? vk::CommandBufferLevel::eSecondary
                                                                         : vk::CommandBufferLevel::ePrimary)
                                                           .setCommandBufferCount(1);

    std::vector<vk::raii::CommandBuffer> commandBuffers = device.allocateCommandBuffers(allocateInfo);
    MRE_ASSERT(commandBuffers.size() == 1, "Expected exactly one command buffer");
    m_commandBuffer = std::move(commandBuffers.front());
}

void VkCommandList::Shutdown()
{
#ifdef MRE_ENABLE_TRACY
    m_gpuZoneScope.reset();
    m_tracyContext = nullptr;
#endif
    m_commandBuffer = vk::raii::CommandBuffer{nullptr};
    m_commandPool = vk::raii::CommandPool{nullptr};
    m_swapchainImage = nullptr;
    m_swapchainImageView = nullptr;
    m_swapchainExtent = vk::Extent2D{0, 0};
    m_swapchainImageLayout = vk::ImageLayout::eUndefined;
    m_boundPipelineLayout = vk::PipelineLayout{nullptr};
    m_boundPipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    m_renderingColorTextures.clear();
    m_renderingDepthTexture = nullptr;
    m_swapchainImageIndex = 0;
    m_hasBoundPipeline = false;
    m_renderedThisFrame = false;
    m_recording = false;
    m_rendering = false;
    m_renderingSwapchain = false;
}

void VkCommandList::SetSwapchainTarget(
    const vk::Image& image,
    const vk::ImageView& imageView,
    std::uint32_t imageIndex,
    bool imageInitialized,
    const vk::Extent2D& extent)
{
    m_swapchainImage = image;
    m_swapchainImageView = imageView;
    m_swapchainImageIndex = imageIndex;
    m_swapchainImageLayout = imageInitialized ? vk::ImageLayout::ePresentSrcKHR : vk::ImageLayout::eUndefined;
    m_swapchainExtent = extent;
    m_renderedThisFrame = false;
    m_hasBoundPipeline = false;
    m_boundPipelineLayout = vk::PipelineLayout{nullptr};
    m_boundPipelineBindPoint = vk::PipelineBindPoint::eGraphics;
}

void VkCommandList::Begin()
{
    MRE_ASSERT(!m_recording, "VkCommandList::Begin called while already recording");
    MRE_ASSERT(!m_rendering, "VkCommandList::Begin called while a rendering scope is open");

    m_commandBuffer.reset();
    m_commandBuffer.begin(vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    m_recording = true;
    m_renderedThisFrame = false;
    m_hasBoundPipeline = false;
    m_boundPipelineLayout = vk::PipelineLayout{nullptr};
    m_boundPipelineBindPoint = vk::PipelineBindPoint::eGraphics;
}

void VkCommandList::End()
{
    MRE_ASSERT(m_recording, "VkCommandList::End called without Begin");
    MRE_ASSERT(m_secondary || !m_rendering, "VkCommandList::End called while a rendering scope is open");
#ifdef MRE_ENABLE_TRACY
    MRE_ASSERT(!m_gpuZoneScope.has_value(), "VkCommandList::End called with an open GPU zone");
#endif

#ifdef MRE_ENABLE_TRACY
    if (!m_secondary && m_tracyContext != nullptr)
    {
        // Collect must run while the command buffer is still recording.
        m_tracyContext->Collect(static_cast<VkCommandBuffer>(*m_commandBuffer));
    }
#endif

    m_commandBuffer.end();
    m_recording = false;
    m_rendering = false;
}

void VkCommandList::BeginSwapchainRendering(const rhi::ClearColor& clearColor, bool clear)
{
    BeginSwapchainRendering(clearColor, rhi::DepthAttachment{}, clear);
}

void VkCommandList::BeginSwapchainRendering(
    const rhi::ClearColor& clearColor,
    const rhi::DepthAttachment& depthAttachment,
    bool clear)
{
    MRE_ASSERT(m_recording, "VkCommandList::BeginSwapchainRendering called outside of a recording scope");
    MRE_ASSERT(!m_rendering, "VkCommandList::BeginSwapchainRendering called while a rendering scope is open");
    MRE_ASSERT(m_swapchainImage != nullptr, "No swapchain target has been set for this command list");

    if (m_swapchainImageLayout != vk::ImageLayout::eColorAttachmentOptimal)
    {
        const bool wasInitialized = m_swapchainImageLayout != vk::ImageLayout::eUndefined;
        const vk::ImageMemoryBarrier2 barrier = MakeImageBarrier(
            m_swapchainImage,
            m_swapchainImageLayout,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageAspectFlagBits::eColor,
            wasInitialized ? vk::PipelineStageFlagBits2::eColorAttachmentOutput : vk::PipelineStageFlagBits2::eNone,
            vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentWrite);
        m_commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
        m_swapchainImageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    }

    const vk::RenderingAttachmentInfo colorAttachment = vk::RenderingAttachmentInfo{}
                                                            .setImageView(m_swapchainImageView)
                                                            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                                                            .setLoadOp(clear ? vk::AttachmentLoadOp::eClear
                                                                             : vk::AttachmentLoadOp::eLoad)
                                                            .setStoreOp(vk::AttachmentStoreOp::eStore)
                                                            .setClearValue(vk::ClearValue{vk::ClearColorValue{
                                                                clearColor.r,
                                                                clearColor.g,
                                                                clearColor.b,
                                                                clearColor.a}});

    vk::RenderingAttachmentInfo depthAttachmentInfo{};
    const bool hasDepth = depthAttachment.texture != nullptr;
    if (hasDepth)
    {
        const auto* depthTexture = static_cast<const VkTexture*>(depthAttachment.texture);
        MRE_ASSERT(depthTexture->IsDepth(), "The depth attachment texture must be a depth texture");

        if (depthTexture->GetLayout() != vk::ImageLayout::eDepthAttachmentOptimal)
        {
            const bool wasInitialized = depthTexture->GetLayout() != vk::ImageLayout::eUndefined;
            const vk::ImageMemoryBarrier2 barrier = MakeImageBarrier(
                depthTexture->GetImage(),
                depthTexture->GetLayout(),
                vk::ImageLayout::eDepthAttachmentOptimal,
                vk::ImageAspectFlagBits::eDepth,
                wasInitialized ? vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                     vk::PipelineStageFlagBits2::eLateFragmentTests
                               : vk::PipelineStageFlagBits2::eNone,
                wasInitialized ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                               : vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                depthTexture->GetArrayLayers());
            m_commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
            depthTexture->SetLayout(vk::ImageLayout::eDepthAttachmentOptimal);
        }

        depthAttachmentInfo = vk::RenderingAttachmentInfo{}
                                  .setImageView(depthTexture->GetLayerView(depthAttachment.layer))
                                  .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
                                  .setLoadOp(depthAttachment.clear ? vk::AttachmentLoadOp::eClear
                                                                   : vk::AttachmentLoadOp::eLoad)
                                  .setStoreOp(vk::AttachmentStoreOp::eStore)
                                  .setClearValue(vk::ClearValue{vk::ClearDepthStencilValue{
                                      depthAttachment.clearDepth,
                                      0}});
        m_renderingDepthTexture = depthTexture;
    }

    vk::RenderingInfo renderingInfo = vk::RenderingInfo{}
                                          .setRenderArea(vk::Rect2D{vk::Offset2D{0, 0}, m_swapchainExtent})
                                          .setLayerCount(1)
                                          .setColorAttachments(colorAttachment);
    if (hasDepth)
    {
        renderingInfo.pDepthAttachment = &depthAttachmentInfo;
    }

    m_commandBuffer.beginRendering(renderingInfo);
    SetViewportAndScissor(0, 0, m_swapchainExtent.width, m_swapchainExtent.height);

    m_rendering = true;
    m_renderingSwapchain = true;
}

void VkCommandList::EndSwapchainRendering()
{
    MRE_ASSERT(m_renderingSwapchain, "VkCommandList::EndSwapchainRendering called without an open rendering scope");

    m_commandBuffer.endRendering();

    const vk::ImageMemoryBarrier2 barrier = MakeImageBarrier(
        m_swapchainImage,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::ImageAspectFlagBits::eColor,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eNone,
        vk::AccessFlagBits2::eNone);
    m_commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
    m_swapchainImageLayout = vk::ImageLayout::ePresentSrcKHR;

    if (m_renderingDepthTexture != nullptr)
    {
        const vk::ImageMemoryBarrier2 depthBarrier = MakeImageBarrier(
            m_renderingDepthTexture->GetImage(),
            vk::ImageLayout::eDepthAttachmentOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::ImageAspectFlagBits::eDepth,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::AccessFlagBits2::eShaderSampledRead,
            m_renderingDepthTexture->GetArrayLayers());
        m_commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(depthBarrier));
        m_renderingDepthTexture->SetLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
        m_renderingDepthTexture = nullptr;
    }

    m_rendering = false;
    m_renderingSwapchain = false;
    m_renderedThisFrame = true;
}

void VkCommandList::BeginRendering(const rhi::RenderingDesc& desc)
{
    MRE_ASSERT(m_recording, "VkCommandList::BeginRendering called outside of a recording scope");
    MRE_ASSERT(!m_rendering, "VkCommandList::BeginRendering called while a rendering scope is open");
    MRE_ASSERT(
        !desc.colorAttachments.empty() || desc.depthAttachment.texture != nullptr,
        "BeginRendering requires at least one color or depth attachment");

    std::vector<vk::RenderingAttachmentInfo> colorAttachments{};
    colorAttachments.reserve(desc.colorAttachments.size());
    m_renderingColorTextures.clear();
    m_renderingColorTextures.reserve(desc.colorAttachments.size());

    vk::Extent2D extent{0, 0};
    for (const rhi::ColorAttachment& attachment : desc.colorAttachments)
    {
        MRE_ASSERT(attachment.texture != nullptr, "Color attachments require a texture");
        const auto* texture = static_cast<const VkTexture*>(attachment.texture);

        if (texture->GetLayout() != vk::ImageLayout::eColorAttachmentOptimal)
        {
            const bool wasInitialized = texture->GetLayout() != vk::ImageLayout::eUndefined;
            const vk::ImageMemoryBarrier2 barrier = MakeImageBarrier(
                texture->GetImage(),
                texture->GetLayout(),
                vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageAspectFlagBits::eColor,
                wasInitialized ? vk::PipelineStageFlagBits2::eColorAttachmentOutput
                               : vk::PipelineStageFlagBits2::eNone,
                wasInitialized ? vk::AccessFlagBits2::eColorAttachmentWrite : vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::AccessFlagBits2::eColorAttachmentWrite,
                texture->GetArrayLayers());
            m_commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
            texture->SetLayout(vk::ImageLayout::eColorAttachmentOptimal);
        }

        colorAttachments.push_back(vk::RenderingAttachmentInfo{}
                                       .setImageView(texture->GetLayerView(0))
                                       .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                                       .setLoadOp(attachment.clear ? vk::AttachmentLoadOp::eClear
                                                                   : vk::AttachmentLoadOp::eLoad)
                                       .setStoreOp(vk::AttachmentStoreOp::eStore)
                                       .setClearValue(vk::ClearValue{vk::ClearColorValue{
                                           attachment.clearColor.r,
                                           attachment.clearColor.g,
                                           attachment.clearColor.b,
                                           attachment.clearColor.a}}));

        m_renderingColorTextures.push_back(texture);
        extent = vk::Extent2D{texture->GetWidth(), texture->GetHeight()};
    }

    vk::RenderingAttachmentInfo depthAttachmentInfo{};
    const bool hasDepth = desc.depthAttachment.texture != nullptr;
    if (hasDepth)
    {
        const auto* depthTexture = static_cast<const VkTexture*>(desc.depthAttachment.texture);
        MRE_ASSERT(depthTexture->IsDepth(), "The depth attachment texture must be a depth texture");

        if (depthTexture->GetLayout() != vk::ImageLayout::eDepthAttachmentOptimal)
        {
            const bool wasInitialized = depthTexture->GetLayout() != vk::ImageLayout::eUndefined;
            const vk::ImageMemoryBarrier2 barrier = MakeImageBarrier(
                depthTexture->GetImage(),
                depthTexture->GetLayout(),
                vk::ImageLayout::eDepthAttachmentOptimal,
                vk::ImageAspectFlagBits::eDepth,
                wasInitialized ? vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                     vk::PipelineStageFlagBits2::eLateFragmentTests
                               : vk::PipelineStageFlagBits2::eNone,
                wasInitialized ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                               : vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                depthTexture->GetArrayLayers());
            m_commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
            depthTexture->SetLayout(vk::ImageLayout::eDepthAttachmentOptimal);
        }

        depthAttachmentInfo = vk::RenderingAttachmentInfo{}
                                  .setImageView(depthTexture->GetLayerView(desc.depthAttachment.layer))
                                  .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
                                  .setLoadOp(desc.depthAttachment.clear ? vk::AttachmentLoadOp::eClear
                                                                        : vk::AttachmentLoadOp::eLoad)
                                  .setStoreOp(vk::AttachmentStoreOp::eStore)
                                  .setClearValue(vk::ClearValue{vk::ClearDepthStencilValue{
                                      desc.depthAttachment.clearDepth,
                                      0}});

        m_renderingDepthTexture = depthTexture;
        if (extent.width == 0)
        {
            extent = vk::Extent2D{depthTexture->GetWidth(), depthTexture->GetHeight()};
        }
    }

    const bool hasArea = desc.area.width > 0 && desc.area.height > 0;
    const vk::Rect2D renderArea = hasArea
        ? vk::Rect2D{
              vk::Offset2D{
                  static_cast<std::int32_t>(desc.area.x),
                  static_cast<std::int32_t>(desc.area.y)},
              vk::Extent2D{desc.area.width, desc.area.height}}
        : vk::Rect2D{vk::Offset2D{0, 0}, extent};

    vk::RenderingInfo renderingInfo = vk::RenderingInfo{}
                                          .setRenderArea(renderArea)
                                          .setLayerCount(1)
                                          .setColorAttachments(colorAttachments);
    if (desc.secondaryCommandBuffers)
    {
        renderingInfo.flags |= vk::RenderingFlagBits::eContentsSecondaryCommandBuffers;
    }
    if (hasDepth)
    {
        renderingInfo.pDepthAttachment = &depthAttachmentInfo;
    }

    m_commandBuffer.beginRendering(renderingInfo);
    SetViewportAndScissor(
        static_cast<std::uint32_t>(renderArea.offset.x),
        static_cast<std::uint32_t>(renderArea.offset.y),
        renderArea.extent.width,
        renderArea.extent.height);

    m_rendering = true;
    m_renderingSwapchain = false;
}

void VkCommandList::EndRendering()
{
    MRE_ASSERT(m_rendering && !m_renderingSwapchain, "VkCommandList::EndRendering called without an open scope");

    m_commandBuffer.endRendering();

    for (const VkTexture* texture : m_renderingColorTextures)
    {
        const vk::ImageMemoryBarrier2 barrier = MakeImageBarrier(
            texture->GetImage(),
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::ImageAspectFlagBits::eColor,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::AccessFlagBits2::eShaderSampledRead,
            texture->GetArrayLayers());
        m_commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
        texture->SetLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    if (m_renderingDepthTexture != nullptr)
    {
        const vk::ImageMemoryBarrier2 barrier = MakeImageBarrier(
            m_renderingDepthTexture->GetImage(),
            vk::ImageLayout::eDepthAttachmentOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::ImageAspectFlagBits::eDepth,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::AccessFlagBits2::eShaderSampledRead,
            m_renderingDepthTexture->GetArrayLayers());
        m_commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
        m_renderingDepthTexture->SetLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    m_renderingColorTextures.clear();
    m_renderingDepthTexture = nullptr;
    m_rendering = false;
}

void VkCommandList::BeginSecondary(const rhi::RenderingDesc& desc)
{
    MRE_ASSERT(m_secondary, "VkCommandList::BeginSecondary requires a secondary command list");
    MRE_ASSERT(!m_recording, "VkCommandList::BeginSecondary called while already recording");
    MRE_ASSERT(
        !desc.colorAttachments.empty() || desc.depthAttachment.texture != nullptr,
        "BeginSecondary requires at least one color or depth attachment");

    std::vector<vk::Format> colorFormats{};
    colorFormats.reserve(desc.colorAttachments.size());

    vk::Extent2D extent{0, 0};
    for (const rhi::ColorAttachment& attachment : desc.colorAttachments)
    {
        MRE_ASSERT(attachment.texture != nullptr, "Color attachments require a texture");
        const auto* texture = static_cast<const VkTexture*>(attachment.texture);
        colorFormats.push_back(ToVkFormat(texture->GetFormat()));
        extent = vk::Extent2D{texture->GetWidth(), texture->GetHeight()};
    }

    vk::Format depthFormat = vk::Format::eUndefined;
    if (desc.depthAttachment.texture != nullptr)
    {
        const auto* depthTexture = static_cast<const VkTexture*>(desc.depthAttachment.texture);
        MRE_ASSERT(depthTexture->IsDepth(), "The depth attachment texture must be a depth texture");
        depthFormat = ToVkFormat(depthTexture->GetFormat());
        if (extent.width == 0)
        {
            extent = vk::Extent2D{depthTexture->GetWidth(), depthTexture->GetHeight()};
        }
    }

    vk::CommandBufferInheritanceRenderingInfo inheritanceRenderingInfo{};
    inheritanceRenderingInfo.colorAttachmentCount = static_cast<std::uint32_t>(colorFormats.size());
    inheritanceRenderingInfo.pColorAttachmentFormats = colorFormats.empty() ? nullptr : colorFormats.data();
    inheritanceRenderingInfo.depthAttachmentFormat = depthFormat;
    inheritanceRenderingInfo.stencilAttachmentFormat = vk::Format::eUndefined;
    inheritanceRenderingInfo.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::CommandBufferInheritanceInfo inheritanceInfo{};
    inheritanceInfo.pNext = &inheritanceRenderingInfo;

    m_commandBuffer.reset();
    m_commandBuffer.begin(vk::CommandBufferBeginInfo{}
                              .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit |
                                        vk::CommandBufferUsageFlagBits::eRenderPassContinue)
                              .setPInheritanceInfo(&inheritanceInfo));

    m_recording = true;
    m_rendering = true;
    m_renderingSwapchain = false;
    m_hasBoundPipeline = false;
    m_boundPipelineLayout = vk::PipelineLayout{nullptr};
    m_boundPipelineBindPoint = vk::PipelineBindPoint::eGraphics;

    const bool hasArea = desc.area.width > 0 && desc.area.height > 0;
    SetViewportAndScissor(
        hasArea ? desc.area.x : 0,
        hasArea ? desc.area.y : 0,
        hasArea ? desc.area.width : extent.width,
        hasArea ? desc.area.height : extent.height);
}

void VkCommandList::ExecuteCommands(const std::vector<const rhi::CommandList*>& secondaryCommandLists)
{
    MRE_ASSERT(m_recording, "VkCommandList::ExecuteCommands called outside of a recording scope");
    MRE_ASSERT(m_rendering, "VkCommandList::ExecuteCommands called outside of a rendering scope");
    MRE_ASSERT(!m_secondary, "VkCommandList::ExecuteCommands can only be called on a primary command list");

    if (secondaryCommandLists.empty())
    {
        return;
    }

    std::vector<vk::CommandBuffer> commandBuffers{};
    commandBuffers.reserve(secondaryCommandLists.size());
    for (const rhi::CommandList* secondary : secondaryCommandLists)
    {
        MRE_ASSERT(secondary != nullptr, "ExecuteCommands requires non-null secondary command lists");
        MRE_ASSERT(secondary->IsSecondary(), "ExecuteCommands requires secondary command lists");
        const auto& vkSecondary = static_cast<const VkCommandList&>(*secondary);
        commandBuffers.push_back(*vkSecondary.GetHandle());
    }

    m_commandBuffer.executeCommands(commandBuffers);
}

void VkCommandList::BeginGpuZone(const char* name)
{
    MRE_ASSERT(m_recording, "VkCommandList::BeginGpuZone called outside of a recording scope");
    MRE_ASSERT(name != nullptr, "VkCommandList::BeginGpuZone requires a zone name");

#ifdef MRE_ENABLE_TRACY
    MRE_ASSERT(!m_gpuZoneScope.has_value(), "VkCommandList::BeginGpuZone called with an open GPU zone");
    MRE_ASSERT(m_tracyContext != nullptr, "VkCommandList::BeginGpuZone requires a Tracy context");

    m_gpuZoneScope.emplace(
        m_tracyContext,
        __LINE__,
        __FILE__,
        std::strlen(__FILE__),
        __FUNCTION__,
        std::strlen(__FUNCTION__),
        name,
        std::strlen(name),
        static_cast<VkCommandBuffer>(*m_commandBuffer),
        true);
#else
    (void)name;
#endif
}

void VkCommandList::EndGpuZone()
{
    MRE_ASSERT(m_recording, "VkCommandList::EndGpuZone called outside of a recording scope");

#ifdef MRE_ENABLE_TRACY
    MRE_ASSERT(m_gpuZoneScope.has_value(), "VkCommandList::EndGpuZone called without an open GPU zone");
    m_gpuZoneScope.reset();
#endif
}

void VkCommandList::BindPipeline(const rhi::Pipeline& pipeline)
{
    MRE_ASSERT(m_recording, "VkCommandList::BindPipeline called outside of a recording scope");

    const auto& vkPipeline = static_cast<const VkPipeline&>(pipeline);
    m_boundPipelineBindPoint = vkPipeline.IsCompute() ? vk::PipelineBindPoint::eCompute
                                                      : vk::PipelineBindPoint::eGraphics;
    m_commandBuffer.bindPipeline(m_boundPipelineBindPoint, vkPipeline.GetHandle());
    m_boundPipelineLayout = vkPipeline.GetLayout();
    m_hasBoundPipeline = true;
}

void VkCommandList::BindDescriptorSet(std::uint32_t setIndex, const rhi::DescriptorSet& descriptorSet)
{
    MRE_ASSERT(m_recording, "VkCommandList::BindDescriptorSet called outside of a recording scope");
    MRE_ASSERT(m_hasBoundPipeline, "VkCommandList::BindDescriptorSet requires a bound pipeline");

    const auto& vkDescriptorSet = static_cast<const VkDescriptorSet&>(descriptorSet);
    const vk::DescriptorSet handle = vkDescriptorSet.GetHandle();
    m_commandBuffer.bindDescriptorSets(m_boundPipelineBindPoint, m_boundPipelineLayout, setIndex, handle, {});
}

void VkCommandList::BindVertexBuffer(const rhi::Buffer& buffer)
{
    MRE_ASSERT(m_recording, "VkCommandList::BindVertexBuffer called outside of a recording scope");

    const auto& vkBuffer = static_cast<const VkBuffer&>(buffer);
    m_commandBuffer.bindVertexBuffers(0, vkBuffer.GetHandle(), vk::DeviceSize{0});
}

void VkCommandList::BindIndexBuffer(const rhi::Buffer& buffer, bool use32BitIndices)
{
    MRE_ASSERT(m_recording, "VkCommandList::BindIndexBuffer called outside of a recording scope");

    const auto& vkBuffer = static_cast<const VkBuffer&>(buffer);
    m_commandBuffer.bindIndexBuffer(
        vkBuffer.GetHandle(),
        0,
        use32BitIndices ? vk::IndexType::eUint32 : vk::IndexType::eUint16);
}

void VkCommandList::SetPushConstants(const void* data, std::uint32_t size, rhi::ShaderStageMask stages)
{
    MRE_ASSERT(m_recording, "VkCommandList::SetPushConstants called outside of a recording scope");
    MRE_ASSERT(m_hasBoundPipeline, "VkCommandList::SetPushConstants requires a bound pipeline");
    MRE_ASSERT(data != nullptr && size > 0, "VkCommandList::SetPushConstants requires data and a size");

    m_commandBuffer.pushConstants(m_boundPipelineLayout, ToVkShaderStageMask(stages), 0, size, data);
}

void VkCommandList::Draw(std::uint32_t vertexCount, std::uint32_t firstVertex)
{
    MRE_ASSERT(m_recording, "VkCommandList::Draw called outside of a recording scope");
    MRE_ASSERT(m_rendering, "VkCommandList::Draw called outside of a rendering scope");

    m_commandBuffer.draw(vertexCount, 1, firstVertex, 0);
}

void VkCommandList::DrawIndexed(std::uint32_t indexCount, std::uint32_t firstIndex)
{
    MRE_ASSERT(m_recording, "VkCommandList::DrawIndexed called outside of a recording scope");
    MRE_ASSERT(m_rendering, "VkCommandList::DrawIndexed called outside of a rendering scope");

    m_commandBuffer.drawIndexed(indexCount, 1, firstIndex, 0, 0);
}

void VkCommandList::Dispatch(std::uint32_t groupCountX, std::uint32_t groupCountY, std::uint32_t groupCountZ)
{
    MRE_ASSERT(m_recording, "VkCommandList::Dispatch called outside of a recording scope");
    MRE_ASSERT(!m_rendering, "VkCommandList::Dispatch cannot be called inside a rendering scope");
    MRE_ASSERT(m_hasBoundPipeline, "VkCommandList::Dispatch requires a bound pipeline");
    MRE_ASSERT(
        m_boundPipelineBindPoint == vk::PipelineBindPoint::eCompute,
        "VkCommandList::Dispatch requires a bound compute pipeline");

    m_commandBuffer.dispatch(groupCountX, groupCountY, groupCountZ);
}

void VkCommandList::ShaderMemoryBarrier(rhi::ShaderStageMask before, rhi::ShaderStageMask after)
{
    MRE_ASSERT(m_recording, "VkCommandList::ShaderMemoryBarrier called outside of a recording scope");

    const vk::MemoryBarrier2 barrier = vk::MemoryBarrier2{}
                                           .setSrcStageMask(ToVkPipelineStageFlags2(before))
                                           .setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
                                           .setDstStageMask(ToVkPipelineStageFlags2(after))
                                           .setDstAccessMask(vk::AccessFlagBits2::eShaderRead);
    m_commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setMemoryBarriers(barrier));
}

void VkCommandList::SetDepthBias(float constantFactor, float slopeFactor, float clamp)
{
    MRE_ASSERT(m_recording, "VkCommandList::SetDepthBias called outside of a recording scope");
    MRE_ASSERT(m_hasBoundPipeline, "VkCommandList::SetDepthBias requires a bound pipeline");

    m_commandBuffer.setDepthBias(constantFactor, clamp, slopeFactor);
}

void VkCommandList::BuildAccelerationStructure(const rhi::AccelerationStructure& accelerationStructure)
{
    MRE_ASSERT(m_recording, "VkCommandList::BuildAccelerationStructure called outside of a recording scope");
    MRE_ASSERT(
        !m_rendering,
        "VkCommandList::BuildAccelerationStructure cannot be called inside a rendering scope");

    const auto& vkAccelerationStructure = static_cast<const VkAccelerationStructure&>(accelerationStructure);
    vkAccelerationStructure.RecordBuild(*m_commandBuffer);
}

void VkCommandList::AccelerationStructureBarrier()
{
    MRE_ASSERT(m_recording, "VkCommandList::AccelerationStructureBarrier called outside of a recording scope");

    const vk::MemoryBarrier2 barrier = vk::MemoryBarrier2{}
                                           .setSrcStageMask(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR)
                                           .setSrcAccessMask(vk::AccessFlagBits2::eAccelerationStructureWriteKHR)
                                           .setDstStageMask(
                                               vk::PipelineStageFlagBits2::eComputeShader |
                                               vk::PipelineStageFlagBits2::eFragmentShader)
                                           .setDstAccessMask(vk::AccessFlagBits2::eAccelerationStructureReadKHR);
    m_commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setMemoryBarriers(barrier));
}

void VkCommandList::SetViewportAndScissor(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t width,
    std::uint32_t height)
{
    const vk::Viewport viewport = vk::Viewport{}
                                      .setX(static_cast<float>(x))
                                      .setY(static_cast<float>(y))
                                      .setWidth(static_cast<float>(width))
                                      .setHeight(static_cast<float>(height))
                                      .setMinDepth(0.0f)
                                      .setMaxDepth(1.0f);
    m_commandBuffer.setViewport(0, viewport);

    const vk::Rect2D scissor = vk::Rect2D{}
                                   .setOffset(vk::Offset2D{
                                       static_cast<std::int32_t>(x),
                                       static_cast<std::int32_t>(y)})
                                   .setExtent(vk::Extent2D{width, height});
    m_commandBuffer.setScissor(0, scissor);
}

} // namespace mukki::vulkan
