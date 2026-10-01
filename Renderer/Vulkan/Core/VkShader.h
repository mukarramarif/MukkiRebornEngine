#pragma once

#include <cstdint>
#include <filesystem>
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
    [[nodiscard]] const rhi::ShaderReflection& GetReflection() const override { return m_reflection; }
    [[nodiscard]] vk::ShaderModule GetHandle() const { return *m_module; }

    [[nodiscard]] static std::vector<std::uint32_t> LoadSpirv(const std::filesystem::path& path);

private:
    void Reflect(const std::vector<std::uint32_t>& code);

    vk::raii::ShaderModule m_module{nullptr};
    rhi::ShaderStage m_stage = rhi::ShaderStage::Vertex;
    rhi::ShaderReflection m_reflection{};
};

} // namespace mukki::vulkan
