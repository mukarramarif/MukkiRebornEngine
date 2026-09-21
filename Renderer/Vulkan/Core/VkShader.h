#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Renderer/RHI/Resources.h"

namespace mukki::vulkan
{

class VkShader final : public rhi::Shader
{
public:
    VkShader(const vk::raii::Device& device, rhi::ShaderStage stage, const std::vector<std::uint32_t>& code);
    ~VkShader() override = default;

    [[nodiscard]] rhi::ShaderStage GetStage() const override { return m_stage; }
    [[nodiscard]] vk::ShaderModule GetHandle() const { return *m_module; }

    [[nodiscard]] static std::vector<std::uint32_t> LoadSpirv(const std::string& path);

private:
    vk::raii::ShaderModule m_module{nullptr};
    rhi::ShaderStage m_stage = rhi::ShaderStage::Vertex;
};

} // namespace mukki::vulkan
