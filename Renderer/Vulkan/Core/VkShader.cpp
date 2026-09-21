#include "Renderer/Vulkan/Core/VkShader.h"

#include "Utilities/Assert.h"
#include "Utilities/Log.h"
#include "Utilities/PathUtils.h"

#include <filesystem>
#include <fstream>

namespace mukki::vulkan
{

VkShader::VkShader(const vk::raii::Device& device, rhi::ShaderStage stage, const std::vector<std::uint32_t>& code)
    : m_stage(stage)
{
    MRE_ASSERT(!code.empty(), "VkShader requires non-empty SPIR-V code");

    const vk::ShaderModuleCreateInfo createInfo = vk::ShaderModuleCreateInfo{}.setCode(code);
    m_module = vk::raii::ShaderModule{device, createInfo};
}

std::vector<std::uint32_t> VkShader::LoadSpirv(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path))
    {
        MRE_ERROR("Shader file '{}' does not exist", PathToUtf8(path));
        return {};
    }

    const std::uintmax_t fileSize = std::filesystem::file_size(path);
    if (fileSize == 0 || fileSize % sizeof(std::uint32_t) != 0)
    {
        MRE_ERROR("Shader file '{}' has an invalid size ({} bytes)", PathToUtf8(path), fileSize);
        return {};
    }

    std::ifstream stream{path, std::ios::in | std::ios::binary};
    if (!stream.is_open())
    {
        MRE_ERROR("Failed to open shader file '{}'", PathToUtf8(path));
        return {};
    }

    std::vector<std::uint32_t> code(fileSize / sizeof(std::uint32_t));
    stream.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(fileSize));
    if (!stream)
    {
        MRE_ERROR("Failed to read shader file '{}'", PathToUtf8(path));
        return {};
    }

    return code;
}

} // namespace mukki::vulkan
