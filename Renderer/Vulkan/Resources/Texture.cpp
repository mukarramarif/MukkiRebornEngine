#include "Renderer/Vulkan/Resources/Texture.h"

#include "Utilities/Assert.h"

namespace mukki::vulkan
{

VkTexture::VkTexture(
    VmaAllocator allocator,
    vk::Image image,
    VmaAllocation allocation,
    vk::raii::ImageView view,
    const rhi::TextureDesc& desc,
    vk::ImageLayout layout)
    : m_allocator(allocator)
    , m_image(image)
    , m_allocation(allocation)
    , m_view(std::move(view))
    , m_layout(layout)
    , m_desc(desc)
{
    MRE_ASSERT(m_allocator != VK_NULL_HANDLE, "VkTexture requires a valid VMA allocator");
    MRE_ASSERT(m_image != nullptr, "VkTexture requires a valid image handle");
    MRE_ASSERT(m_allocation != VK_NULL_HANDLE, "VkTexture requires a valid VMA allocation");
    MRE_ASSERT(m_view != nullptr, "VkTexture requires a valid image view");
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
