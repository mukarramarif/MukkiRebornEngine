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
#include "Renderer/Vulkan/Core/VkPipeline.h"
#include "Renderer/Vulkan/Core/VkRayTracingDispatch.h"
#include "Renderer/Vulkan/Core/VkShader.h"
#include "Renderer/Vulkan/Core/VkSwapchain.h"
#include "Renderer/Vulkan/Resources/ResourceAllocator.h"
#include "Renderer/Vulkan/Resources/AccelerationStructure.h"
#include "Renderer/Vulkan/Resources/DescriptorSet.h"
#include "Renderer/Vulkan/Resources/Sampler.h"
#include "Renderer/Vulkan/Resources/Texture.h"
#include "Utilities/ProfilingVulkan.h"

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
    [[nodiscard]] std::unique_ptr<rhi::CommandList> CreateSecondaryCommandList() override;
    [[nodiscard]] const rhi::SwapchainInfo& GetSwapchainInfo() const override;
    [[nodiscard]] std::uint32_t GetFrameIndex() const override { return m_frameIndex; }
    [[nodiscard]] std::uint32_t GetFramesInFlight() const override { return m_framesInFlight; }
    [[nodiscard]] bool IsBindlessSupported() const override { return m_descriptorIndexingSupported; }
    [[nodiscard]] bool IsRayTracingSupported() const override { return m_rayTracingSupported; }

    [[nodiscard]] std::unique_ptr<rhi::Buffer> CreateBuffer(
        const rhi::BufferDesc& desc,
        const void* initialData,
        std::uint64_t initialDataSize) override;
    [[nodiscard]] std::unique_ptr<rhi::Texture> CreateTexture(const rhi::TextureDesc& desc) override;
    [[nodiscard]] std::unique_ptr<rhi::Sampler> CreateSampler(const rhi::SamplerDesc& desc) override;
    [[nodiscard]] std::unique_ptr<rhi::Shader> CreateShader(
        rhi::ShaderStage stage,
        const std::filesystem::path& spirvPath) override;
    [[nodiscard]] std::unique_ptr<rhi::Pipeline> CreatePipeline(const rhi::PipelineDesc& desc) override;
    [[nodiscard]] std::unique_ptr<rhi::DescriptorSetLayout> CreateDescriptorSetLayout(
        const std::vector<rhi::DescriptorBinding>& bindings) override;
    [[nodiscard]] std::unique_ptr<rhi::DescriptorSet> CreateDescriptorSet(const rhi::DescriptorSetLayout& layout)
        override;

    [[nodiscard]] std::unique_ptr<rhi::AccelerationStructure> CreateBottomLevelAccelerationStructure(
        const rhi::BottomLevelAccelerationStructureDesc& desc) override;
    [[nodiscard]] std::unique_ptr<rhi::AccelerationStructure> CreateTopLevelAccelerationStructure(
        const rhi::TopLevelAccelerationStructureDesc& desc) override;
    void UpdateTopLevelAccelerationStructure(
        rhi::AccelerationStructure& accelerationStructure,
        std::span<const rhi::AccelerationStructureInstance> instances) override;

    [[nodiscard]] bool IsInitialized() const { return m_initialized; }
    [[nodiscard]] const VkInstance& GetVkInstance() const { return m_instance; }
    [[nodiscard]] const vk::raii::PhysicalDevice& GetPhysicalDevice() const { return m_physicalDevice; }
    [[nodiscard]] const vk::raii::Device& GetHandle() const { return m_device; }
    [[nodiscard]] const vk::raii::Queue& GetGraphicsQueue() const { return m_graphicsQueue; }
    [[nodiscard]] std::uint32_t GetGraphicsQueueFamily() const { return m_graphicsQueueFamily; }
    [[nodiscard]] VmaAllocator GetAllocator() const { return m_allocator; }
    [[nodiscard]] const RayTracingDispatch* GetRayTracingDispatch() const
    {
        return m_rayTracingSupported ? &m_rayTracingDispatch : nullptr;
    }
    [[nodiscard]] const vk::PhysicalDeviceDescriptorIndexingFeatures& GetDescriptorIndexingFeatures() const
    {
        return m_descriptorIndexingFeatures;
    }
    [[nodiscard]] const vk::PhysicalDeviceDescriptorIndexingProperties& GetDescriptorIndexingProperties() const
    {
        return m_descriptorIndexingProperties;
    }

private:
    bool PickPhysicalDevice();
    bool CreateLogicalDevice();
    bool CreateAllocator();
    bool CreateDescriptorPool();
    bool CreateSwapchain();
    bool CreateFrameResources();
    void RecreateSwapchain();
    void WaitForFrame(std::uint32_t frame);
    void LogDeviceFault();

    VkInstance m_instance{};
    vk::raii::PhysicalDevice m_physicalDevice{nullptr};
    vk::raii::Device m_device{nullptr};
    vk::raii::Queue m_graphicsQueue{nullptr};
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkResourceAllocator m_resourceAllocator{};
    vk::raii::DescriptorPool m_descriptorPool{nullptr};
    std::unique_ptr<VkSwapchain> m_swapchain{};
    FrameSync m_frameSync{};
    std::vector<std::unique_ptr<VkCommandList>> m_commandLists{};
    rhi::SwapchainInfo m_swapchainInfo{};
    rhi::Extent2D m_requestedExtent{};
    vk::PhysicalDeviceDescriptorIndexingFeatures m_descriptorIndexingFeatures{};
    vk::PhysicalDeviceDescriptorIndexingProperties m_descriptorIndexingProperties{};
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR m_accelerationStructureFeatures{};
    vk::PhysicalDeviceRayQueryFeaturesKHR m_rayQueryFeatures{};
#ifdef MRE_ENABLE_TRACY
    tracy::VkCtx* m_tracyContext = nullptr;
#endif
    std::uint32_t m_graphicsQueueFamily = 0;
    std::uint32_t m_apiVersion = VK_API_VERSION_1_3;
    std::uint32_t m_framesInFlight = 2;
    std::uint32_t m_frameIndex = 0;
    std::uint32_t m_currentImageIndex = 0;
    std::uint32_t m_nextBindlessIndex = 0;
    bool m_descriptorIndexingSupported = false;
    RayTracingDispatch m_rayTracingDispatch{};
    bool m_rayTracingSupported = false;
    bool m_swapchainDirty = false;
    bool m_initialized = false;
    bool m_deviceFaultSupported = false;
};

[[nodiscard]] std::unique_ptr<rhi::Device> CreateVulkanDevice(const rhi::DeviceDesc& desc);

} // namespace mukki::vulkan
