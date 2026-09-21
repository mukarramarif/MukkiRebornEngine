#pragma once

#include <cstdint>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Renderer/RHI/Resources.h"

namespace mukki::vulkan
{

class VkTexture final : public rhi::Texture
{
public:
    VkTexture(VmaAllocator allocator, const vk::raii::Device& device, const rhi::TextureDesc& desc);
    ~VkTexture() override;

    [[nodiscard]] std::uint32_t GetWidth() const override { return m_desc.width; }
    [[nodiscard]] std::uint32_t GetHeight() const override { return m_desc.height; }
    [[nodiscard]] rhi::Format GetFormat() const override { return m_desc.format; }
    [[nodiscard]] rhi::TextureUsage GetUsage() const override { return m_desc.usage; }

    [[nodiscard]] vk::Image GetImage() const { return m_image; }
    [[nodiscard]] vk::ImageView GetImageView() const { return *m_view; }
    [[nodiscard]] vk::ImageLayout GetLayout() const { return m_layout; }
    void SetLayout(vk::ImageLayout layout) const { m_layout = layout; }
    [[nodiscard]] bool IsDepth() const { return m_desc.usage == rhi::TextureUsage::DepthAttachment; }

private:
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    vk::Image m_image{nullptr};
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    vk::raii::ImageView m_view{nullptr};
    mutable vk::ImageLayout m_layout = vk::ImageLayout::eUndefined;
    rhi::TextureDesc m_desc{};
};

} // namespace mukki::vulkan
