#include "Renderer/Vulkan/Resources/ResourceAllocator.h"

#include "Renderer/Vulkan/Core/VkConversions.h"
#include "Renderer/Vulkan/Resources/Buffer.h"
#include "Renderer/Vulkan/Resources/Texture.h"
#include "Utilities/Assert.h"
#include "Utilities/Log.h"

#include <cstring>
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

vk::ImageUsageFlags ToVkTextureUsage(rhi::TextureUsage usage)
{
    switch (usage)
    {
    case rhi::TextureUsage::Sampled:
        return vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
    case rhi::TextureUsage::ColorAttachment:
        return vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
    case rhi::TextureUsage::DepthAttachment:
        return vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled;
    }

    MRE_UNREACHABLE();
}

} // namespace

void VkResourceAllocator::Init(
    const vk::raii::Device& device,
    VmaAllocator allocator,
    vk::Queue queue,
    std::uint32_t queueFamily)
{
    MRE_ASSERT(allocator != VK_NULL_HANDLE, "VkResourceAllocator::Init requires a valid VMA allocator");
    MRE_ASSERT(queue != nullptr, "VkResourceAllocator::Init requires a valid queue");

    m_device = &device;
    m_allocator = allocator;
    m_queue = queue;

    const vk::CommandPoolCreateInfo poolInfo = vk::CommandPoolCreateInfo{}
                                                   .setFlags(vk::CommandPoolCreateFlagBits::eTransient)
                                                   .setQueueFamilyIndex(queueFamily);
    m_commandPool = vk::raii::CommandPool{device, poolInfo};
}

void VkResourceAllocator::Shutdown()
{
    m_commandPool = vk::raii::CommandPool{nullptr};
    m_device = nullptr;
    m_allocator = VK_NULL_HANDLE;
    m_queue = nullptr;
}

std::unique_ptr<rhi::Buffer> VkResourceAllocator::CreateBuffer(
    const rhi::BufferDesc& desc,
    const void* initialData,
    std::uint64_t initialDataSize)
{
    MRE_ASSERT(m_allocator != VK_NULL_HANDLE, "VkResourceAllocator::CreateBuffer called before Init");
    MRE_ASSERT(desc.size > 0, "Buffer size must be greater than zero");
    MRE_ASSERT(initialDataSize <= desc.size, "Initial data exceeds the buffer size");
    MRE_ASSERT(
        initialData != nullptr || initialDataSize == 0,
        "Initial data size requires a valid initial data pointer");

    const bool useStaging = initialData != nullptr && initialDataSize > 0 && !desc.hostVisible;

    vk::BufferUsageFlags usage = ToVkBufferUsage(desc.usage);
    if (useStaging)
    {
        usage |= vk::BufferUsageFlagBits::eTransferDst;
    }

    const vk::BufferCreateInfo bufferInfo = vk::BufferCreateInfo{}
                                                .setSize(desc.size)
                                                .setUsage(usage)
                                                .setSharingMode(vk::SharingMode::eExclusive);

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = desc.hostVisible ? VMA_MEMORY_USAGE_AUTO : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (desc.hostVisible)
    {
        allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                               VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    ::VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    const VkBufferCreateInfo nativeInfo = static_cast<VkBufferCreateInfo>(bufferInfo);
    const VkResult result = vmaCreateBuffer(m_allocator, &nativeInfo, &allocationInfo, &buffer, &allocation, nullptr);
    if (result != VK_SUCCESS)
    {
        MRE_ERROR("Failed to create a {} byte buffer (VkResult {})", desc.size, static_cast<int>(result));
        return nullptr;
    }

    auto vkBuffer = std::make_unique<VkBuffer>(m_allocator, buffer, allocation, desc.size, desc.hostVisible);

    if (initialData != nullptr && initialDataSize > 0)
    {
        if (desc.hostVisible)
        {
            vkBuffer->Write(initialData, initialDataSize, 0);
        }
        else
        {
            const vk::BufferCreateInfo stagingInfo = vk::BufferCreateInfo{}
                                                         .setSize(initialDataSize)
                                                         .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
                                                         .setSharingMode(vk::SharingMode::eExclusive);

            VmaAllocationCreateInfo stagingAllocationInfo{};
            stagingAllocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
            stagingAllocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                          VMA_ALLOCATION_CREATE_MAPPED_BIT;

            ::VkBuffer stagingBuffer = VK_NULL_HANDLE;
            VmaAllocation stagingAllocation = VK_NULL_HANDLE;
            VmaAllocationInfo stagingAllocationResult{};
            const VkBufferCreateInfo nativeStagingInfo = static_cast<VkBufferCreateInfo>(stagingInfo);
            const VkResult stagingResult = vmaCreateBuffer(
                m_allocator,
                &nativeStagingInfo,
                &stagingAllocationInfo,
                &stagingBuffer,
                &stagingAllocation,
                &stagingAllocationResult);
            if (stagingResult != VK_SUCCESS)
            {
                MRE_ERROR("Failed to create a staging buffer (VkResult {})", static_cast<int>(stagingResult));
                return vkBuffer;
            }

            std::memcpy(stagingAllocationResult.pMappedData, initialData, initialDataSize);

            ImmediateSubmit([&](vk::CommandBuffer commandBuffer) {
                const vk::BufferCopy region{0, 0, initialDataSize};
                commandBuffer.copyBuffer(stagingBuffer, buffer, region);
            });

            vmaDestroyBuffer(m_allocator, stagingBuffer, stagingAllocation);
        }
    }

    return vkBuffer;
}

std::unique_ptr<rhi::Texture> VkResourceAllocator::CreateTexture(const rhi::TextureDesc& desc)
{
    MRE_ASSERT(m_allocator != VK_NULL_HANDLE, "VkResourceAllocator::CreateTexture called before Init");
    MRE_ASSERT(desc.width > 0 && desc.height > 0, "Texture dimensions must be greater than zero");
    MRE_ASSERT(desc.format != rhi::Format::Undefined, "Texture format must be defined");
    MRE_ASSERT(desc.arrayLayers > 0, "Texture layer count must be greater than zero");

    const bool hasData = desc.initialData != nullptr && desc.initialDataSize > 0;
    MRE_ASSERT(!hasData || desc.arrayLayers == 1, "Initial data is only supported for single-layer textures");
    const bool isDepth = desc.usage == rhi::TextureUsage::DepthAttachment;
    const vk::ImageAspectFlags aspect = isDepth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;

    if (hasData)
    {
        MRE_ASSERT(desc.usage == rhi::TextureUsage::Sampled, "Initial data is only supported for sampled textures");
        MRE_ASSERT(
            desc.format == rhi::Format::R8G8B8A8Unorm || desc.format == rhi::Format::R8G8B8A8Srgb,
            "Initial data requires an R8G8B8A8 format");
        MRE_ASSERT(
            desc.initialDataSize == static_cast<std::uint64_t>(desc.width) * desc.height * 4,
            "Initial data size must match width * height * 4");
    }

    vk::ImageUsageFlags usage = ToVkTextureUsage(desc.usage);
    if (hasData)
    {
        usage |= vk::ImageUsageFlagBits::eTransferDst;
    }

    vk::ImageCreateInfo imageInfo{};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = ToVkFormat(desc.format);
    imageInfo.extent = vk::Extent3D{desc.width, desc.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = desc.arrayLayers;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.usage = usage;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ::VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    const VkImageCreateInfo nativeInfo = static_cast<VkImageCreateInfo>(imageInfo);
    const VkResult result = vmaCreateImage(m_allocator, &nativeInfo, &allocationInfo, &image, &allocation, nullptr);
    if (result != VK_SUCCESS)
    {
        MRE_ERROR(
            "Failed to create a {}x{} texture (VkResult {})",
            desc.width,
            desc.height,
            static_cast<int>(result));
        return nullptr;
    }

    vk::ImageLayout layout = desc.usage == rhi::TextureUsage::Sampled ? vk::ImageLayout::eShaderReadOnlyOptimal
                                                                     : vk::ImageLayout::eUndefined;

    if (hasData)
    {
        const vk::BufferCreateInfo stagingInfo = vk::BufferCreateInfo{}
                                                     .setSize(desc.initialDataSize)
                                                     .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
                                                     .setSharingMode(vk::SharingMode::eExclusive);

        VmaAllocationCreateInfo stagingAllocationInfo{};
        stagingAllocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                      VMA_ALLOCATION_CREATE_MAPPED_BIT;

        ::VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VmaAllocation stagingAllocation = VK_NULL_HANDLE;
        VmaAllocationInfo stagingAllocationResult{};
        const VkBufferCreateInfo nativeStagingInfo = static_cast<VkBufferCreateInfo>(stagingInfo);
        const VkResult stagingResult = vmaCreateBuffer(
            m_allocator,
            &nativeStagingInfo,
            &stagingAllocationInfo,
            &stagingBuffer,
            &stagingAllocation,
            &stagingAllocationResult);
        if (stagingResult != VK_SUCCESS)
        {
            MRE_ERROR("Failed to create a texture staging buffer (VkResult {})", static_cast<int>(stagingResult));
            vmaDestroyImage(m_allocator, image, allocation);
            return nullptr;
        }

        std::memcpy(stagingAllocationResult.pMappedData, desc.initialData, desc.initialDataSize);

        ImmediateSubmit([&](vk::CommandBuffer commandBuffer) {
            const vk::ImageMemoryBarrier2 toTransfer = MakeImageBarrier(
                image,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal,
                aspect,
                vk::PipelineStageFlagBits2::eNone,
                vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eTransfer,
                vk::AccessFlagBits2::eTransferWrite);
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(toTransfer));

            const vk::BufferImageCopy copyRegion = vk::BufferImageCopy{}
                                                       .setBufferOffset(0)
                                                       .setBufferRowLength(0)
                                                       .setBufferImageHeight(0)
                                                       .setImageSubresource(vk::ImageSubresourceLayers{}
                                                                                .setAspectMask(aspect)
                                                                                .setMipLevel(0)
                                                                                .setBaseArrayLayer(0)
                                                                                .setLayerCount(1))
                                                       .setImageOffset(vk::Offset3D{0, 0, 0})
                                                       .setImageExtent(vk::Extent3D{desc.width, desc.height, 1});
            commandBuffer.copyBufferToImage(
                stagingBuffer,
                image,
                vk::ImageLayout::eTransferDstOptimal,
                copyRegion);

            const vk::ImageMemoryBarrier2 toShaderRead = MakeImageBarrier(
                image,
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal,
                aspect,
                vk::PipelineStageFlagBits2::eTransfer,
                vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eFragmentShader,
                vk::AccessFlagBits2::eShaderSampledRead);
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(toShaderRead));
        });

        vmaDestroyBuffer(m_allocator, stagingBuffer, stagingAllocation);
        layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    else if (isDepth)
    {
        // Depth targets are sampled by later passes (shadow maps, depth debug views); start them in
        // a shader readable layout so they can be bound before their first render pass.
        ImmediateSubmit([&](vk::CommandBuffer commandBuffer) {
            const vk::ImageMemoryBarrier2 toShaderRead = MakeImageBarrier(
                image,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eShaderReadOnlyOptimal,
                aspect,
                vk::PipelineStageFlagBits2::eNone,
                vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eFragmentShader,
                vk::AccessFlagBits2::eShaderSampledRead,
                desc.arrayLayers);
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(toShaderRead));
        });
        layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }

    const vk::ImageViewCreateInfo viewInfo = vk::ImageViewCreateInfo{}
                                                 .setImage(image)
                                                 .setViewType(
                                                     desc.arrayLayers > 1 ? vk::ImageViewType::e2DArray
                                                                          : vk::ImageViewType::e2D)
                                                 .setFormat(ToVkFormat(desc.format))
                                                 .setSubresourceRange(vk::ImageSubresourceRange{}
                                                                          .setAspectMask(aspect)
                                                                          .setBaseMipLevel(0)
                                                                          .setLevelCount(1)
                                                                          .setBaseArrayLayer(0)
                                                                          .setLayerCount(desc.arrayLayers));
    vk::raii::ImageView view{*m_device, viewInfo};

    // Depth arrays are rendered one layer at a time, which needs per-layer views.
    std::vector<vk::raii::ImageView> layerViews{};
    if (isDepth && desc.arrayLayers > 1)
    {
        layerViews.reserve(desc.arrayLayers);
        for (std::uint32_t layer = 0; layer < desc.arrayLayers; ++layer)
        {
            const vk::ImageViewCreateInfo layerViewInfo = vk::ImageViewCreateInfo{}
                                                              .setImage(image)
                                                              .setViewType(vk::ImageViewType::e2D)
                                                              .setFormat(ToVkFormat(desc.format))
                                                              .setSubresourceRange(vk::ImageSubresourceRange{}
                                                                                       .setAspectMask(aspect)
                                                                                       .setBaseMipLevel(0)
                                                                                       .setLevelCount(1)
                                                                                       .setBaseArrayLayer(layer)
                                                                                       .setLayerCount(1));
            layerViews.emplace_back(*m_device, layerViewInfo);
        }
    }

    return std::make_unique<VkTexture>(
        m_allocator,
        image,
        allocation,
        std::move(view),
        std::move(layerViews),
        desc,
        layout);
}

void VkResourceAllocator::ImmediateSubmit(const std::function<void(vk::CommandBuffer)>& record)
{
    MRE_ASSERT(m_device != nullptr, "VkResourceAllocator::ImmediateSubmit requires an initialized allocator");

    std::vector<vk::raii::CommandBuffer> commandBuffers = m_device->allocateCommandBuffers(
        vk::CommandBufferAllocateInfo{}
            .setCommandPool(*m_commandPool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(1));
    MRE_ASSERT(commandBuffers.size() == 1, "Expected exactly one command buffer");

    vk::raii::CommandBuffer& commandBuffer = commandBuffers.front();
    commandBuffer.begin(vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    record(*commandBuffer);
    commandBuffer.end();

    const vk::CommandBufferSubmitInfo commandBufferInfo = vk::CommandBufferSubmitInfo{}.setCommandBuffer(*commandBuffer);
    const vk::SubmitInfo2 submitInfo = vk::SubmitInfo2{}.setCommandBufferInfos(commandBufferInfo);
    m_queue.submit2(submitInfo);
    m_queue.waitIdle();
}

} // namespace mukki::vulkan
