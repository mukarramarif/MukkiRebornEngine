#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Renderer/RHI/CommandList.h"
#include "Renderer/RHI/Resources.h"
#include "Utilities/ProfilingVulkan.h"

namespace mukki::vulkan
{

class VkTexture;

class VkCommandList final : public rhi::CommandList
{
public:
    VkCommandList() = default;
    ~VkCommandList() override = default;

    VkCommandList(const VkCommandList&) = delete;
    VkCommandList& operator=(const VkCommandList&) = delete;

    void Init(const vk::raii::Device& device, std::uint32_t queueFamilyIndex, bool secondary = false);
    void Shutdown();

    void SetSwapchainTarget(
        const vk::Image& image,
        const vk::ImageView& imageView,
        std::uint32_t imageIndex,
        bool imageInitialized,
        const vk::Extent2D& extent);

    void Begin() override;
    void End() override;
    void BeginSwapchainRendering(const rhi::ClearColor& clearColor, bool clear) override;
    void EndSwapchainRendering() override;
    void BeginRendering(const rhi::RenderingDesc& desc) override;
    void EndRendering() override;

    void BindPipeline(const rhi::Pipeline& pipeline) override;
    void BindDescriptorSet(std::uint32_t setIndex, const rhi::DescriptorSet& descriptorSet) override;
    void BindVertexBuffer(const rhi::Buffer& buffer) override;
    void BindIndexBuffer(const rhi::Buffer& buffer, bool use32BitIndices) override;
    void SetPushConstants(const void* data, std::uint32_t size, rhi::ShaderStageMask stages) override;
    void Draw(std::uint32_t vertexCount, std::uint32_t firstVertex) override;
    void DrawIndexed(std::uint32_t indexCount, std::uint32_t firstIndex) override;
    void Dispatch(std::uint32_t groupCountX, std::uint32_t groupCountY, std::uint32_t groupCountZ) override;
    void ShaderMemoryBarrier(rhi::ShaderStageMask before, rhi::ShaderStageMask after) override;
    void SetDepthBias(float constantFactor, float slopeFactor, float clamp) override;
    void BuildAccelerationStructure(const rhi::AccelerationStructure& accelerationStructure) override;
    void AccelerationStructureBarrier() override;
    void BeginGpuZone(const char* name) override;
    void EndGpuZone() override;

    [[nodiscard]] bool IsSecondary() const override { return m_secondary; }
    void BeginSecondary(const rhi::RenderingDesc& desc) override;
    void ExecuteCommands(const std::vector<const rhi::CommandList*>& secondaryCommandLists) override;

#ifdef MRE_ENABLE_TRACY
    void SetTracyContext(tracy::VkCtx* context) { m_tracyContext = context; }
#endif

    [[nodiscard]] const vk::raii::CommandBuffer& GetHandle() const { return m_commandBuffer; }
    [[nodiscard]] bool WasSwapchainRendered() const { return m_renderedThisFrame; }

private:
    void SetViewportAndScissor(std::uint32_t x, std::uint32_t y, std::uint32_t width, std::uint32_t height);

    vk::raii::CommandPool m_commandPool{nullptr};
    vk::raii::CommandBuffer m_commandBuffer{nullptr};
#ifdef MRE_ENABLE_TRACY
    std::optional<tracy::VkCtxScope> m_gpuZoneScope{};
    tracy::VkCtx* m_tracyContext = nullptr;
#endif
    vk::Image m_swapchainImage{nullptr};
    vk::ImageView m_swapchainImageView{nullptr};
    vk::Extent2D m_swapchainExtent{0, 0};
    vk::ImageLayout m_swapchainImageLayout = vk::ImageLayout::eUndefined;
    vk::PipelineLayout m_boundPipelineLayout{nullptr};
    vk::PipelineBindPoint m_boundPipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    std::vector<const VkTexture*> m_renderingColorTextures{};
    const VkTexture* m_renderingDepthTexture = nullptr;
    std::uint32_t m_swapchainImageIndex = 0;
    bool m_hasBoundPipeline = false;
    bool m_renderedThisFrame = false;
    bool m_recording = false;
    bool m_rendering = false;
    bool m_renderingSwapchain = false;
    bool m_secondary = false;
};

} // namespace mukki::vulkan
