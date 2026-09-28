#pragma once

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Renderer/RHI/Resources.h"

namespace mukki::vulkan
{

class VkPipeline final : public rhi::Pipeline
{
public:
    VkPipeline(const vk::raii::Device& device, const rhi::PipelineDesc& desc);
    ~VkPipeline() override = default;

    [[nodiscard]] vk::Pipeline GetHandle() const { return *m_pipeline; }
    [[nodiscard]] vk::PipelineLayout GetLayout() const { return *m_layout; }
    [[nodiscard]] bool IsCompute() const override { return m_isCompute; }

private:
    vk::raii::PipelineLayout m_layout{nullptr};
    vk::raii::Pipeline m_pipeline{nullptr};
    bool m_isCompute = false;
};

} // namespace mukki::vulkan
