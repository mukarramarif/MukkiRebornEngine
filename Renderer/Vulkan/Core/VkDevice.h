#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vk_mem_alloc.h>

#include "Renderer/RHI/Device.h"
#include "Renderer/Vulkan/Core/FrameSync.h"
#include "Renderer/Vulkan/Core/VkCommandList.h"
#include "Renderer/Vulkan/Core/VkInstance.h"
#include "Renderer/Vulkan/Core/VkSwapchain.h"

namespace mukki::vulkan
{

class VkDevice final : public rhi::Device
{
public:
    VkDevice() = default;
    ~VkDevice() override;

    VkDevice(const VkDevice&) = delete;
    VkDevice& operator=(const VkDevice&) = delete;

    bool Init(const rhi::DeviceDesc& desc);
    void Shutdown();

    bool BeginFrame() override;
    void EndFrame() override;
    void WaitIdle() override;
    void Resize(const rhi::Extent2D& extent) override;

    [[nodiscard]] rhi::CommandList& GetCommandList() override;
    [[nodiscard]] const rhi::SwapchainInfo& GetSwapchainInfo() const override;

    [[nodiscard]] bool IsInitialized() const { return m_initialized; }
    [[nodiscard]] const VkInstance& GetVkInstance() const { return m_instance; }
    [[nodiscard]] const vk::raii::PhysicalDevice& GetPhysicalDevice() const { return m_physicalDevice; }
    [[nodiscard]] const vk::raii::Device& GetHandle() const { return m_device; }
    [[nodiscard]] const vk::raii::Queue& GetGraphicsQueue() const { return m_graphicsQueue; }
    [[nodiscard]] std::uint32_t GetGraphicsQueueFamily() const { return m_graphicsQueueFamily; }
    [[nodiscard]] VmaAllocator GetAllocator() const { return m_allocator; }

private:
    bool PickPhysicalDevice();
    bool CreateLogicalDevice();
    bool CreateAllocator();
    bool CreateSwapchain();
    bool CreateFrameResources();
    void RecreateSwapchain();
    void WaitForFrame(std::uint32_t frame);

    VkInstance m_instance{};
    vk::raii::PhysicalDevice m_physicalDevice{nullptr};
    vk::raii::Device m_device{nullptr};
    vk::raii::Queue m_graphicsQueue{nullptr};
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    std::unique_ptr<VkSwapchain> m_swapchain{};
    FrameSync m_frameSync{};
    std::vector<std::unique_ptr<VkCommandList>> m_commandLists{};
    rhi::SwapchainInfo m_swapchainInfo{};
    rhi::Extent2D m_requestedExtent{};
    std::uint32_t m_graphicsQueueFamily = 0;
    std::uint32_t m_apiVersion = VK_API_VERSION_1_3;
    std::uint32_t m_framesInFlight = 2;
    std::uint32_t m_frameIndex = 0;
    std::uint32_t m_currentImageIndex = 0;
    bool m_swapchainDirty = false;
    bool m_initialized = false;
};

[[nodiscard]] std::unique_ptr<rhi::Device> CreateVulkanDevice(const rhi::DeviceDesc& desc);

} // namespace mukki::vulkan
