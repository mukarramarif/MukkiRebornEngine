#include "Renderer/Vulkan/Resources/Buffer.h"

#include "Utilities/Assert.h"

#include <cstring>

namespace mukki::vulkan
{

VkBuffer::VkBuffer(
    VmaAllocator allocator,
    vk::Buffer buffer,
    VmaAllocation allocation,
    std::uint64_t size,
    bool hostVisible,
    std::uint64_t deviceAddress)
    : m_allocator(allocator)
    , m_buffer(buffer)
    , m_allocation(allocation)
    , m_size(size)
    , m_deviceAddress(deviceAddress)
    , m_hostVisible(hostVisible)
{
    MRE_ASSERT(m_buffer != nullptr, "VkBuffer requires a valid buffer handle");
    MRE_ASSERT(m_allocation != VK_NULL_HANDLE, "VkBuffer requires a valid VMA allocation");
}

VkBuffer::~VkBuffer()
{
    if (m_buffer != nullptr && m_allocator != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
    }

    m_buffer = nullptr;
    m_allocation = VK_NULL_HANDLE;
}

void VkBuffer::Write(const void* data, std::uint64_t size, std::uint64_t offset)
{
    MRE_ASSERT(m_hostVisible, "VkBuffer::Write requires a host-visible buffer");
    MRE_ASSERT(data != nullptr, "VkBuffer::Write requires a data pointer");
    if (offset + size > m_size)
    {
        // Never corrupt neighbouring allocations, even when assertions are compiled out.
        MRE_CRITICAL(
            "VkBuffer::Write exceeds the buffer size (offset {}, size {}, capacity {})",
            offset,
            size,
            m_size);
        return;
    }

    void* mapped = nullptr;
    const VkResult result = vmaMapMemory(m_allocator, m_allocation, &mapped);
    if (result != VK_SUCCESS || mapped == nullptr)
    {
        MRE_ERROR("Failed to map a buffer (VkResult {})", static_cast<int>(result));
        return;
    }

    std::memcpy(static_cast<std::byte*>(mapped) + offset, data, size);
    vmaFlushAllocation(m_allocator, m_allocation, offset, size);
    vmaUnmapMemory(m_allocator, m_allocation);
}

} // namespace mukki::vulkan
