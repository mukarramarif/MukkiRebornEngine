#include "Renderer/Vulkan/Core/VkCommandList.h"

#include "Renderer/Vulkan/Core/VkConversions.h"
#include "Renderer/Vulkan/Core/VkPipeline.h"
#include "Renderer/Vulkan/Resources/Buffer.h"
#include "Utilities/Assert.h"

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
    vk::PipelineStageFlags2 srcStage,
    vk::AccessFlags2 srcAccess,
    vk::PipelineStageFlags2 dstStage,
    vk::AccessFlags2 dstAccess)
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
                                   .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                   .setBaseMipLevel(0)
                                   .setLevelCount(1)
                                   .setBaseArrayLayer(0)
                                   .setLayerCount(1);
    return barrier;
}

} // namespace

void VkCommandList::Init(const vk::raii::Device& device, std::uint32_t queueFamilyIndex)
{
    const vk::CommandPoolCreateInfo poolInfo = vk::CommandPoolCreateInfo{}
                                                   .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                                                   .setQueueFamilyIndex(queueFamilyIndex);
    m_commandPool = vk::raii::CommandPool{device, poolInfo};

    const vk::CommandBufferAllocateInfo allocateInfo = vk::CommandBufferAllocateInfo{}
                                                           .setCommandPool(*m_commandPool)
                                                           .setLevel(vk::CommandBufferLevel::ePrimary)
                                                           .setCommandBufferCount(1);

    std::vector<vk::raii::CommandBuffer> commandBuffers = device.allocateCommandBuffers(allocateInfo);
    MRE_ASSERT(commandBuffers.size() == 1, "Expected exactly one command buffer");
    m_commandBuffer = std::move(commandBuffers.front());
}

void VkCommandList::Shutdown()
{
    m_commandBuffer = vk::raii::CommandBuffer{nullptr};
    m_commandPool = vk::raii::CommandPool{nullptr};
    m_swapchainImage = nullptr;
    m_swapchainImageView = nullptr;
    m_swapchainExtent = vk::Extent2D{0, 0};
    m_swapchainImageLayout = vk::ImageLayout::eUndefined;
    m_boundPipelineLayout = vk::PipelineLayout{nullptr};
    m_swapchainImageIndex = 0;
    m_hasBoundPipeline = false;
    m_renderedThisFrame = false;
    m_recording = false;
    m_rendering = false;
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
}

void VkCommandList::End()
{
    MRE_ASSERT(m_recording, "VkCommandList::End called without Begin");
    MRE_ASSERT(!m_rendering, "VkCommandList::End called while a rendering scope is open");

    m_commandBuffer.end();
    m_recording = false;
}

void VkCommandList::BeginSwapchainRendering(const rhi::ClearColor& clearColor, bool clear)
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

    const vk::RenderingInfo renderingInfo = vk::RenderingInfo{}
                                                .setRenderArea(vk::Rect2D{vk::Offset2D{0, 0}, m_swapchainExtent})
                                                .setLayerCount(1)
                                                .setColorAttachments(colorAttachment);

    m_commandBuffer.beginRendering(renderingInfo);

    const vk::Viewport viewport = vk::Viewport{}
                                      .setX(0.0f)
                                      .setY(0.0f)
                                      .setWidth(static_cast<float>(m_swapchainExtent.width))
                                      .setHeight(static_cast<float>(m_swapchainExtent.height))
                                      .setMinDepth(0.0f)
                                      .setMaxDepth(1.0f);
    m_commandBuffer.setViewport(0, viewport);

    const vk::Rect2D scissor = vk::Rect2D{}.setOffset(vk::Offset2D{0, 0}).setExtent(m_swapchainExtent);
    m_commandBuffer.setScissor(0, scissor);

    m_rendering = true;
}

void VkCommandList::EndSwapchainRendering()
{
    MRE_ASSERT(m_rendering, "VkCommandList::EndSwapchainRendering called without an open rendering scope");

    m_commandBuffer.endRendering();

    const vk::ImageMemoryBarrier2 barrier = MakeImageBarrier(
        m_swapchainImage,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eNone,
        vk::AccessFlagBits2::eNone);
    m_commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(barrier));
    m_swapchainImageLayout = vk::ImageLayout::ePresentSrcKHR;

    m_rendering = false;
    m_renderedThisFrame = true;
}

void VkCommandList::BindPipeline(const rhi::Pipeline& pipeline)
{
    MRE_ASSERT(m_recording, "VkCommandList::BindPipeline called outside of a recording scope");

    const auto& vkPipeline = static_cast<const VkPipeline&>(pipeline);
    m_commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, vkPipeline.GetHandle());
    m_boundPipelineLayout = vkPipeline.GetLayout();
    m_hasBoundPipeline = true;
}

void VkCommandList::BindVertexBuffer(const rhi::Buffer& buffer)
{
    MRE_ASSERT(m_recording, "VkCommandList::BindVertexBuffer called outside of a recording scope");

    const auto& vkBuffer = static_cast<const VkBuffer&>(buffer);
    m_commandBuffer.bindVertexBuffers(0, vkBuffer.GetHandle(), vk::DeviceSize{0});
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

} // namespace mukki::vulkan
