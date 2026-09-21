#pragma once

#include <cstdint>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Renderer/RHI/CommandList.h"
#include "Renderer/RHI/Resources.h"

namespace mukki::vulkan
{

class VkCommandList final : public rhi::CommandList
{
public:
    VkCommandList() = default;
    ~VkCommandList() override = default;

    VkCommandList(const VkCommandList&) = delete;
    VkCommandList& operator=(const VkCommandList&) = delete;

    void Init(const vk::raii::Device& device, std::uint32_t queueFamilyIndex);
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

    void BindPipeline(const rhi::Pipeline& pipeline) override;
    void BindVertexBuffer(const rhi::Buffer& buffer) override;
    void SetPushConstants(const void* data, std::uint32_t size, rhi::ShaderStageMask stages) override;
    void Draw(std::uint32_t vertexCount, std::uint32_t firstVertex) override;

    [[nodiscard]] const vk::raii::CommandBuffer& GetHandle() const { return m_commandBuffer; }
    [[nodiscard]] bool WasSwapchainRendered() const { return m_renderedThisFrame; }

private:
    vk::raii::CommandPool m_commandPool{nullptr};
    vk::raii::CommandBuffer m_commandBuffer{nullptr};
    vk::Image m_swapchainImage{nullptr};
    vk::ImageView m_swapchainImageView{nullptr};
    vk::Extent2D m_swapchainExtent{0, 0};
    vk::ImageLayout m_swapchainImageLayout = vk::ImageLayout::eUndefined;
    vk::PipelineLayout m_boundPipelineLayout{nullptr};
    std::uint32_t m_swapchainImageIndex = 0;
    bool m_hasBoundPipeline = false;
    bool m_renderedThisFrame = false;
    bool m_recording = false;
    bool m_rendering = false;
};

} // namespace mukki::vulkan
