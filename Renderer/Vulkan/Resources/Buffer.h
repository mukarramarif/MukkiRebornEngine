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
        bool hostVisible);
    ~VkBuffer() override;

    [[nodiscard]] std::uint64_t GetSize() const override { return m_size; }
    [[nodiscard]] bool IsHostVisible() const override { return m_hostVisible; }
    void Write(const void* data, std::uint64_t size, std::uint64_t offset) override;

    [[nodiscard]] vk::Buffer GetHandle() const { return m_buffer; }

private:
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    vk::Buffer m_buffer{nullptr};
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    std::uint64_t m_size = 0;
    bool m_hostVisible = false;
};

} // namespace mukki::vulkan
