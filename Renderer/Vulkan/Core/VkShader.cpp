#include "Renderer/Vulkan/Core/VkShader.h"

#include "Utilities/Assert.h"
#include "Utilities/Log.h"

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

std::vector<std::uint32_t> VkShader::LoadSpirv(const std::string& path)
{
    const std::filesystem::path filePath{path};
    if (!std::filesystem::exists(filePath))
    {
        MRE_ERROR("Shader file '{}' does not exist", path);
        return {};
    }

    const std::uintmax_t fileSize = std::filesystem::file_size(filePath);
    if (fileSize == 0 || fileSize % sizeof(std::uint32_t) != 0)
    {
        MRE_ERROR("Shader file '{}' has an invalid size ({} bytes)", path, fileSize);
        return {};
    }

    std::ifstream stream{filePath, std::ios::in | std::ios::binary};
    if (!stream.is_open())
    {
        MRE_ERROR("Failed to open shader file '{}'", path);
        return {};
    }

    std::vector<std::uint32_t> code(fileSize / sizeof(std::uint32_t));
    stream.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(fileSize));
    if (!stream)
    {
        MRE_ERROR("Failed to read shader file '{}'", path);
        return {};
    }

    return code;
}

} // namespace mukki::vulkan
