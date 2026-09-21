#include "Renderer/Vulkan/Resources/Texture.h"

#include "Renderer/Vulkan/Core/VkConversions.h"
#include "Utilities/Assert.h"
#include "Utilities/Log.h"

namespace mukki::vulkan
{
namespace
{

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

VkTexture::VkTexture(VmaAllocator allocator, const vk::raii::Device& device, const rhi::TextureDesc& desc)
    : m_allocator(allocator)
    , m_desc(desc)
{
    MRE_ASSERT(m_allocator != VK_NULL_HANDLE, "VkTexture requires a valid VMA allocator");
    MRE_ASSERT(desc.width > 0 && desc.height > 0, "Texture dimensions must be greater than zero");
    MRE_ASSERT(desc.format != rhi::Format::Undefined, "Texture format must be defined");

    const bool isDepth = desc.usage == rhi::TextureUsage::DepthAttachment;

    vk::ImageCreateInfo imageInfo{};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = ToVkFormat(desc.format);
    imageInfo.extent = vk::Extent3D{desc.width, desc.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.usage = ToVkTextureUsage(desc.usage);
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    const VkImageCreateInfo nativeInfo = static_cast<VkImageCreateInfo>(imageInfo);
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    const VkResult result = vmaCreateImage(
        m_allocator,
        &nativeInfo,
        &allocationInfo,
        &image,
        &allocation,
        nullptr);
    if (result != VK_SUCCESS)
    {
        MRE_CRITICAL(
            "Failed to create a {}x{} texture (VkResult {})",
            desc.width,
            desc.height,
            static_cast<int>(result));
        return;
    }

    m_image = image;
    m_allocation = allocation;

    const vk::ImageViewCreateInfo viewInfo = vk::ImageViewCreateInfo{}
                                                 .setImage(m_image)
                                                 .setViewType(vk::ImageViewType::e2D)
                                                 .setFormat(ToVkFormat(desc.format))
                                                 .setSubresourceRange(vk::ImageSubresourceRange{}
                                                                          .setAspectMask(isDepth
                                                                                             ? vk::ImageAspectFlagBits::eDepth
                                                                                             : vk::ImageAspectFlagBits::eColor)
                                                                          .setBaseMipLevel(0)
                                                                          .setLevelCount(1)
                                                                          .setBaseArrayLayer(0)
                                                                          .setLayerCount(1));
    m_view = vk::raii::ImageView{device, viewInfo};

    m_layout = desc.usage == rhi::TextureUsage::Sampled ? vk::ImageLayout::eShaderReadOnlyOptimal
                                                        : vk::ImageLayout::eUndefined;
}

VkTexture::~VkTexture()
{
    m_view = vk::raii::ImageView{nullptr};

    if (m_image != nullptr && m_allocator != VK_NULL_HANDLE)
    {
        vmaDestroyImage(m_allocator, m_image, m_allocation);
    }

    m_image = nullptr;
    m_allocation = VK_NULL_HANDLE;
}

} // namespace mukki::vulkan
