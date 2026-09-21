#include "Renderer/Vulkan/Core/VkSwapchain.h"

#include "Utilities/Assert.h"
#include "Utilities/Log.h"

#include <algorithm>
#include <limits>

namespace mukki::vulkan
{

bool VkSwapchain::Init(
    const vk::raii::PhysicalDevice& physicalDevice,
    const vk::raii::Device& device,
    const vk::raii::SurfaceKHR& surface,
    std::uint32_t queueFamilyIndex,
    const rhi::Extent2D& requestedExtent,
    vk::SwapchainKHR oldSwapchain)
{
    MRE_ASSERT(!m_initialized, "VkSwapchain::Init called twice");

    const vk::SurfaceCapabilitiesKHR capabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    const std::vector<vk::SurfaceFormatKHR> formats = physicalDevice.getSurfaceFormatsKHR(*surface);
    const std::vector<vk::PresentModeKHR> presentModes = physicalDevice.getSurfacePresentModesKHR(*surface);

    if (formats.empty() || presentModes.empty())
    {
        MRE_CRITICAL("The surface does not support any swapchain format or present mode");
        return false;
    }

    const vk::SurfaceFormatKHR surfaceFormat = ChooseSurfaceFormat(formats);
    const vk::PresentModeKHR presentMode = ChoosePresentMode(presentModes);
    const vk::Extent2D extent = ChooseExtent(capabilities, requestedExtent);

    std::uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
    {
        imageCount = capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo{};
    createInfo.surface = *surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    createInfo.imageSharingMode = vk::SharingMode::eExclusive;
    createInfo.queueFamilyIndexCount = 1;
    createInfo.pQueueFamilyIndices = &queueFamilyIndex;
    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    createInfo.presentMode = presentMode;
    createInfo.clipped = true;
    createInfo.oldSwapchain = oldSwapchain;

    try
    {
        m_swapchain = vk::raii::SwapchainKHR{device, createInfo};
    }
    catch (const vk::SystemError& error)
    {
        MRE_CRITICAL("Failed to create the swapchain: {}", error.what());
        return false;
    }

    m_images = m_swapchain.getImages();
    m_format = surfaceFormat.format;
    m_extent = extent;
    m_imageInitialized.assign(m_images.size(), false);

    m_imageViews.clear();
    m_imageViews.reserve(m_images.size());
    m_renderFinished.clear();
    m_renderFinished.reserve(m_images.size());
    for (const vk::Image image : m_images)
    {
        const vk::ImageViewCreateInfo viewInfo = vk::ImageViewCreateInfo{}
                                                     .setImage(image)
                                                     .setViewType(vk::ImageViewType::e2D)
                                                     .setFormat(m_format)
                                                     .setSubresourceRange(vk::ImageSubresourceRange{}
                                                                              .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                                                              .setBaseMipLevel(0)
                                                                              .setLevelCount(1)
                                                                              .setBaseArrayLayer(0)
                                                                              .setLayerCount(1));
        m_imageViews.emplace_back(device, viewInfo);
        m_renderFinished.emplace_back(device, vk::SemaphoreCreateInfo{});
    }

    MRE_INFO(
        "Swapchain created ({}x{}, {} images, format {})",
        extent.width,
        extent.height,
        m_images.size(),
        vk::to_string(m_format));

    m_initialized = true;
    return true;
}

void VkSwapchain::Shutdown()
{
    m_renderFinished.clear();
    m_imageViews.clear();
    m_images.clear();
    m_imageInitialized.clear();
    m_swapchain = vk::raii::SwapchainKHR{nullptr};
    m_format = vk::Format::eUndefined;
    m_extent = vk::Extent2D{0, 0};
    m_initialized = false;
}

vk::SurfaceFormatKHR VkSwapchain::ChooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats)
{
    if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined)
    {
        return vk::SurfaceFormatKHR{vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear};
    }

    const auto preferred = std::find_if(formats.begin(), formats.end(), [](const vk::SurfaceFormatKHR& candidate) {
        return candidate.format == vk::Format::eB8G8R8A8Srgb && candidate.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });
    if (preferred != formats.end())
    {
        return *preferred;
    }

    return formats[0];
}

vk::PresentModeKHR VkSwapchain::ChoosePresentMode(const std::vector<vk::PresentModeKHR>& presentModes)
{
    const auto mailbox = std::find(presentModes.begin(), presentModes.end(), vk::PresentModeKHR::eMailbox);
    if (mailbox != presentModes.end())
    {
        return *mailbox;
    }

    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D VkSwapchain::ChooseExtent(
    const vk::SurfaceCapabilitiesKHR& capabilities,
    const rhi::Extent2D& requested)
{
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
    {
        return capabilities.currentExtent;
    }

    const std::uint32_t width = std::clamp(
        requested.width,
        capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width);
    const std::uint32_t height = std::clamp(
        requested.height,
        capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height);
    return vk::Extent2D{width, height};
}

} // namespace mukki::vulkan
