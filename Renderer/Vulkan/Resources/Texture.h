#pragma once

#include <cstdint>
#include <vector>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Renderer/RHI/Resources.h"
#include "Utilities/Assert.h"

namespace mukki::vulkan
{

class VkTexture final : public rhi::Texture
{
public:
    VkTexture(
        VmaAllocator allocator,
        vk::Image image,
        VmaAllocation allocation,
        vk::raii::ImageView view,
        std::vector<vk::raii::ImageView> layerViews,
        const rhi::TextureDesc& desc,
        vk::ImageLayout layout);
    ~VkTexture() override;

    [[nodiscard]] std::uint32_t GetWidth() const override { return m_desc.width; }
    [[nodiscard]] std::uint32_t GetHeight() const override { return m_desc.height; }
    [[nodiscard]] rhi::Format GetFormat() const override { return m_desc.format; }
    [[nodiscard]] rhi::TextureUsage GetUsage() const override { return m_desc.usage; }
    [[nodiscard]] std::uint32_t GetArrayLayers() const override { return m_desc.arrayLayers; }
    [[nodiscard]] std::uint32_t GetDepth() const override { return m_desc.depth; }
    [[nodiscard]] std::uint32_t GetBindlessIndex() const override { return m_bindlessIndex; }

    void SetBindlessIndex(std::uint32_t index) { m_bindlessIndex = index; }

    [[nodiscard]] vk::Image GetImage() const { return m_image; }
    [[nodiscard]] vk::ImageView GetImageView() const { return *m_view; }

    // Single-layer view used as a rendering attachment. Falls back to the full view for textures
    // without per-layer views.
    [[nodiscard]] vk::ImageView GetLayerView(std::uint32_t layer) const
    {
        if (m_layerViews.empty())
        {
            return *m_view;
        }
        MRE_ASSERT(layer < m_layerViews.size(), "Texture layer index out of range");
        return *m_layerViews[layer];
    }

    [[nodiscard]] vk::ImageLayout GetLayout() const { return m_layout; }
    void SetLayout(vk::ImageLayout layout) const { m_layout = layout; }
    [[nodiscard]] bool IsDepth() const { return m_desc.usage == rhi::TextureUsage::DepthAttachment; }

private:
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    vk::Image m_image{nullptr};
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    vk::raii::ImageView m_view{nullptr};
    std::vector<vk::raii::ImageView> m_layerViews{};
    mutable vk::ImageLayout m_layout = vk::ImageLayout::eUndefined;
    rhi::TextureDesc m_desc{};
    std::uint32_t m_bindlessIndex = 0;
};

} // namespace mukki::vulkan
