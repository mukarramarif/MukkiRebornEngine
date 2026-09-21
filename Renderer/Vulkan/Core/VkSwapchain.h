#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Renderer/RHI/Resources.h"

namespace mukki::vulkan
{

class VkSwapchain
{
public:
    VkSwapchain() = default;
    ~VkSwapchain() = default;

    VkSwapchain(const VkSwapchain&) = delete;
    VkSwapchain& operator=(const VkSwapchain&) = delete;
    VkSwapchain(VkSwapchain&&) = delete;
    VkSwapchain& operator=(VkSwapchain&&) = delete;

    bool Init(
        const vk::raii::PhysicalDevice& physicalDevice,
        const vk::raii::Device& device,
        const vk::raii::SurfaceKHR& surface,
        std::uint32_t queueFamilyIndex,
        const rhi::Extent2D& requestedExtent,
        vk::SwapchainKHR oldSwapchain = nullptr);
    void Shutdown();

    [[nodiscard]] bool IsInitialized() const { return m_initialized; }
    [[nodiscard]] vk::SwapchainKHR GetHandle() const { return *m_swapchain; }
    [[nodiscard]] vk::Format GetFormat() const { return m_format; }
    [[nodiscard]] const vk::Extent2D& GetExtent() const { return m_extent; }
    [[nodiscard]] std::uint32_t GetImageCount() const { return static_cast<std::uint32_t>(m_images.size()); }
    [[nodiscard]] vk::Image GetImage(std::uint32_t index) const { return m_images[index]; }
    [[nodiscard]] vk::ImageView GetImageView(std::uint32_t index) const { return *m_imageViews[index]; }
    [[nodiscard]] vk::Semaphore GetRenderFinished(std::uint32_t index) const { return *m_renderFinished[index]; }
    [[nodiscard]] bool IsImageInitialized(std::uint32_t index) const { return m_imageInitialized[index]; }
    void MarkImageInitialized(std::uint32_t index) { m_imageInitialized[index] = true; }

private:
    static vk::SurfaceFormatKHR ChooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats);
    static vk::PresentModeKHR ChoosePresentMode(const std::vector<vk::PresentModeKHR>& presentModes);
    static vk::Extent2D ChooseExtent(const vk::SurfaceCapabilitiesKHR& capabilities, const rhi::Extent2D& requested);

    vk::raii::SwapchainKHR m_swapchain{nullptr};
    std::vector<vk::Image> m_images{};
    std::vector<vk::raii::ImageView> m_imageViews{};
    std::vector<vk::raii::Semaphore> m_renderFinished{};
    std::vector<bool> m_imageInitialized{};
    vk::Format m_format = vk::Format::eUndefined;
    vk::Extent2D m_extent{0, 0};
    bool m_initialized = false;
};

} // namespace mukki::vulkan
