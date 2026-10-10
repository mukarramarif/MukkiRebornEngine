#pragma once

#include <cstdint>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Renderer/RHI/Resources.h"

namespace mukki::vulkan
{

class VkBuffer final : public rhi::Buffer
{
public:
    VkBuffer(
        VmaAllocator allocator,
        vk::Buffer buffer,
        VmaAllocation allocation,
        std::uint64_t size,
        bool hostVisible,
        std::uint64_t deviceAddress);
    ~VkBuffer() override;

    [[nodiscard]] std::uint64_t GetSize() const override { return m_size; }
    [[nodiscard]] bool IsHostVisible() const override { return m_hostVisible; }
    [[nodiscard]] std::uint64_t GetDeviceAddress() const override { return m_deviceAddress; }
    void Write(const void* data, std::uint64_t size, std::uint64_t offset) override;
    [[nodiscard]] bool Read(void* data, std::uint64_t size, std::uint64_t offset) override;

    [[nodiscard]] vk::Buffer GetHandle() const { return m_buffer; }

private:
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    vk::Buffer m_buffer{nullptr};
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    std::uint64_t m_size = 0;
    std::uint64_t m_deviceAddress = 0;
    bool m_hostVisible = false;
};

} // namespace mukki::vulkan
