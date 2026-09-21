#include "Renderer/Vulkan/Resources/BufferAllocator.h"

#include "Renderer/Vulkan/Core/VkConversions.h"
#include "Renderer/Vulkan/Resources/Buffer.h"
#include "Utilities/Assert.h"
#include "Utilities/Log.h"

#include <cstring>
#include <vector>

namespace mukki::vulkan
{
namespace
{

VkBufferCreateInfo ToNativeBufferCreateInfo(const vk::BufferCreateInfo& createInfo)
{
    return static_cast<VkBufferCreateInfo>(createInfo);
}

} // namespace

void VkBufferAllocator::Init(
    const vk::raii::Device& device,
    VmaAllocator allocator,
    vk::Queue queue,
    std::uint32_t queueFamily)
{
    MRE_ASSERT(allocator != VK_NULL_HANDLE, "VkBufferAllocator::Init requires a valid VMA allocator");
    MRE_ASSERT(queue != nullptr, "VkBufferAllocator::Init requires a valid queue");

    m_device = &device;
    m_allocator = allocator;
    m_queue = queue;

    const vk::CommandPoolCreateInfo poolInfo = vk::CommandPoolCreateInfo{}
                                                   .setFlags(vk::CommandPoolCreateFlagBits::eTransient)
                                                   .setQueueFamilyIndex(queueFamily);
    m_commandPool = vk::raii::CommandPool{device, poolInfo};
}

void VkBufferAllocator::Shutdown()
{
    m_commandPool = vk::raii::CommandPool{nullptr};
    m_device = nullptr;
    m_allocator = VK_NULL_HANDLE;
    m_queue = nullptr;
}

std::unique_ptr<rhi::Buffer> VkBufferAllocator::CreateBuffer(
    const rhi::BufferDesc& desc,
    const void* initialData,
    std::uint64_t initialDataSize)
{
    MRE_ASSERT(m_allocator != VK_NULL_HANDLE, "VkBufferAllocator::CreateBuffer called before Init");
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
    const VkBufferCreateInfo nativeInfo = ToNativeBufferCreateInfo(bufferInfo);
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
            const VkBufferCreateInfo nativeStagingInfo = ToNativeBufferCreateInfo(stagingInfo);
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

void VkBufferAllocator::ImmediateSubmit(const std::function<void(vk::CommandBuffer)>& record)
{
    MRE_ASSERT(m_device != nullptr, "VkBufferAllocator::ImmediateSubmit requires an initialized allocator");

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
