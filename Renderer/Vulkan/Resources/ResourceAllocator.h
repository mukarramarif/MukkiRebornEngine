#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Renderer/RHI/Resources.h"

namespace mukki::vulkan
{

class VkResourceAllocator
{
public:
    VkResourceAllocator() = default;
    ~VkResourceAllocator() = default;

    VkResourceAllocator(const VkResourceAllocator&) = delete;
    VkResourceAllocator& operator=(const VkResourceAllocator&) = delete;

    void Init(const vk::raii::Device& device, VmaAllocator allocator, vk::Queue queue, std::uint32_t queueFamily);
    void Shutdown();

    [[nodiscard]] std::unique_ptr<rhi::Buffer> CreateBuffer(
        const rhi::BufferDesc& desc,
        const void* initialData,
        std::uint64_t initialDataSize);
    [[nodiscard]] std::unique_ptr<rhi::Texture> CreateTexture(const rhi::TextureDesc& desc);

private:
    void ImmediateSubmit(const std::function<void(vk::CommandBuffer)>& record);

    const vk::raii::Device* m_device = nullptr;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    vk::Queue m_queue{nullptr};
    vk::raii::CommandPool m_commandPool{nullptr};
};

} // namespace mukki::vulkan
