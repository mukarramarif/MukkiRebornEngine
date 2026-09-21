#include "Renderer/Vulkan/Core/VkCommandList.h"

#include "Utilities/Assert.h"

#include <utility>
#include <vector>

namespace mukki::vulkan
{

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
    m_swapchainImageIndex = 0;
    m_swapchainImageInitialized = false;
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
    m_swapchainImageInitialized = imageInitialized;
    m_swapchainExtent = extent;
    m_renderedThisFrame = false;
}

void VkCommandList::Begin()
{
    MRE_ASSERT(!m_recording, "VkCommandList::Begin called while already recording");
    MRE_ASSERT(!m_rendering, "VkCommandList::Begin called while a rendering scope is open");

    m_commandBuffer.reset();
    m_commandBuffer.begin(vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    m_recording = true;
    m_renderedThisFrame = false;
}

void VkCommandList::End()
{
    MRE_ASSERT(m_recording, "VkCommandList::End called without Begin");
    MRE_ASSERT(!m_rendering, "VkCommandList::End called while a rendering scope is open");

    m_commandBuffer.end();
    m_recording = false;
}

void VkCommandList::BeginSwapchainRendering(const rhi::ClearColor& clearColor)
{
    MRE_ASSERT(m_recording, "VkCommandList::BeginSwapchainRendering called outside of a recording scope");
    MRE_ASSERT(!m_rendering, "VkCommandList::BeginSwapchainRendering called while a rendering scope is open");
    MRE_ASSERT(m_swapchainImage != nullptr, "No swapchain target has been set for this command list");

    vk::ImageMemoryBarrier2 barrier{};
    barrier.srcStageMask = m_swapchainImageInitialized ? vk::PipelineStageFlagBits2::eColorAttachmentOutput
                                                       : vk::PipelineStageFlagBits2::eNone;
    barrier.srcAccessMask = vk::AccessFlagBits2::eNone;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    barrier.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
    barrier.oldLayout = m_swapchainImageInitialized ? vk::ImageLayout::ePresentSrcKHR : vk::ImageLayout::eUndefined;
    barrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
    barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.image = m_swapchainImage;
    barrier.subresourceRange = vk::ImageSubresourceRange{}
                                   .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                   .setBaseMipLevel(0)
                                   .setLevelCount(1)
                                   .setBaseArrayLayer(0)
                                   .setLayerCount(1);

    const vk::DependencyInfo dependencyInfo = vk::DependencyInfo{}.setImageMemoryBarriers(barrier);
    m_commandBuffer.pipelineBarrier2(dependencyInfo);

    const vk::RenderingAttachmentInfo colorAttachment = vk::RenderingAttachmentInfo{}
                                                            .setImageView(m_swapchainImageView)
                                                            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                                                            .setLoadOp(vk::AttachmentLoadOp::eClear)
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
    m_rendering = true;
}

void VkCommandList::EndSwapchainRendering()
{
    MRE_ASSERT(m_rendering, "VkCommandList::EndSwapchainRendering called without an open rendering scope");

    m_commandBuffer.endRendering();

    vk::ImageMemoryBarrier2 barrier{};
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    barrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eNone;
    barrier.dstAccessMask = vk::AccessFlagBits2::eNone;
    barrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
    barrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
    barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.image = m_swapchainImage;
    barrier.subresourceRange = vk::ImageSubresourceRange{}
                                   .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                   .setBaseMipLevel(0)
                                   .setLevelCount(1)
                                   .setBaseArrayLayer(0)
                                   .setLayerCount(1);

    const vk::DependencyInfo dependencyInfo = vk::DependencyInfo{}.setImageMemoryBarriers(barrier);
    m_commandBuffer.pipelineBarrier2(dependencyInfo);

    m_rendering = false;
    m_swapchainImageInitialized = true;
    m_renderedThisFrame = true;
}

} // namespace mukki::vulkan
