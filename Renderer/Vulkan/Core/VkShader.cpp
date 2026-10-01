#include "Renderer/Vulkan/Core/VkShader.h"

#include "Utilities/Assert.h"
#include "Utilities/Log.h"
#include "Utilities/PathUtils.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>

#include <spirv-reflect/spirv_reflect.h>

namespace mukki::vulkan
{
namespace
{

rhi::ShaderStageMask ToShaderStageMask(rhi::ShaderStage stage)
{
    switch (stage)
    {
    case rhi::ShaderStage::Vertex:
        return rhi::ShaderStageMask::Vertex;
    case rhi::ShaderStage::Fragment:
        return rhi::ShaderStageMask::Fragment;
    case rhi::ShaderStage::Compute:
        return rhi::ShaderStageMask::Compute;
    }
    return rhi::ShaderStageMask::None;
}

rhi::DescriptorType FromSpvDescriptorType(SpvReflectDescriptorType type)
{
    switch (type)
    {
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
        return rhi::DescriptorType::Sampler;
    case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        return rhi::DescriptorType::CombinedImageSampler;
    case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        return rhi::DescriptorType::SampledImage;
    case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        return rhi::DescriptorType::UniformBuffer;
    case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        return rhi::DescriptorType::StorageBuffer;
    case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
        return rhi::DescriptorType::AccelerationStructure;
    default:
        MRE_ASSERT(false, "Unsupported descriptor type {} in shader reflection", static_cast<int>(type));
        return rhi::DescriptorType::UniformBuffer;
    }
}

rhi::Format FromSpvFormat(SpvReflectFormat format)
{
    switch (format)
    {
    case SPV_REFLECT_FORMAT_R32G32_SFLOAT:
        return rhi::Format::R32G32Sfloat;
    case SPV_REFLECT_FORMAT_R32G32B32_SFLOAT:
        return rhi::Format::R32G32B32Sfloat;
    case SPV_REFLECT_FORMAT_R32G32B32A32_SFLOAT:
        return rhi::Format::R32G32B32A32Sfloat;
    default:
        MRE_ASSERT(false, "Unsupported vertex input format {} in shader reflection", static_cast<int>(format));
        return rhi::Format::Undefined;
    }
}

} // namespace

VkShader::VkShader(const vk::raii::Device& device, rhi::ShaderStage stage, const std::vector<std::uint32_t>& code)
    : m_stage(stage)
{
    MRE_ASSERT(!code.empty(), "VkShader requires non-empty SPIR-V code");

    const vk::ShaderModuleCreateInfo createInfo = vk::ShaderModuleCreateInfo{}.setCode(code);
    m_module = vk::raii::ShaderModule{device, createInfo};

    Reflect(code);
}

void VkShader::Reflect(const std::vector<std::uint32_t>& code)
{
    m_reflection = rhi::ShaderReflection{};
    m_reflection.stage = m_stage;

    SpvReflectShaderModule module{};
    const SpvReflectResult createResult =
        spvReflectCreateShaderModule(code.size() * sizeof(std::uint32_t), code.data(), &module);
    if (createResult != SPV_REFLECT_RESULT_SUCCESS)
    {
        MRE_ERROR(
            "Failed to reflect the shader module (SPIRV-Reflect result {})",
            static_cast<int>(createResult));
        return;
    }

    const rhi::ShaderStageMask stageMask = ToShaderStageMask(m_stage);

    std::uint32_t bindingCount = 0;
    spvReflectEnumerateDescriptorBindings(&module, &bindingCount, nullptr);
    std::vector<SpvReflectDescriptorBinding*> bindings(bindingCount, nullptr);
    spvReflectEnumerateDescriptorBindings(&module, &bindingCount, bindings.data());
    for (const SpvReflectDescriptorBinding* binding : bindings)
    {
        rhi::ShaderReflectionBinding reflected{};
        reflected.set = binding->set;
        reflected.binding = binding->binding;
        reflected.type = FromSpvDescriptorType(binding->descriptor_type);
        reflected.arrayCount = std::max(binding->count, 1u);
        reflected.stages = stageMask;
        m_reflection.bindings.push_back(reflected);
    }

    std::uint32_t pushConstantCount = 0;
    spvReflectEnumeratePushConstantBlocks(&module, &pushConstantCount, nullptr);
    std::vector<SpvReflectBlockVariable*> pushConstants(pushConstantCount, nullptr);
    spvReflectEnumeratePushConstantBlocks(&module, &pushConstantCount, pushConstants.data());
    for (const SpvReflectBlockVariable* block : pushConstants)
    {
        rhi::ShaderReflectionPushConstant reflected{};
        reflected.offset = block->offset;
        reflected.size = block->size;
        reflected.stages = stageMask;
        m_reflection.pushConstants.push_back(reflected);
    }

    if (m_stage == rhi::ShaderStage::Vertex)
    {
        std::uint32_t inputCount = 0;
        spvReflectEnumerateInputVariables(&module, &inputCount, nullptr);
        std::vector<SpvReflectInterfaceVariable*> inputs(inputCount, nullptr);
        spvReflectEnumerateInputVariables(&module, &inputCount, inputs.data());
        for (const SpvReflectInterfaceVariable* input : inputs)
        {
            if (static_cast<std::int32_t>(input->built_in) >= 0 ||
                input->format == SPV_REFLECT_FORMAT_UNDEFINED)
            {
                continue;
            }

            rhi::VertexAttribute attribute{};
            attribute.location = input->location;
            attribute.format = FromSpvFormat(input->format);
            attribute.offset = 0; // offsets are derived by VertexLayout::AddAttribute
            m_reflection.vertexInputs.push_back(attribute);
        }
    }

    spvReflectDestroyShaderModule(&module);
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
