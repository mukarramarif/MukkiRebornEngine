#include "Renderer/RHI/Resources.h"

#include <algorithm>

#include "Utilities/Assert.h"
#include "Utilities/Log.h"

namespace mukki::rhi
{

Buffer::~Buffer() = default;

Texture::~Texture() = default;

Sampler::~Sampler() = default;

DescriptorSetLayout::~DescriptorSetLayout() = default;

DescriptorSet::~DescriptorSet() = default;

Shader::~Shader() = default;

Pipeline::~Pipeline() = default;

void VertexLayout::AddAttribute(std::uint32_t location, Format format)
{
    const std::uint32_t formatSize = GetFormatSize(format);
    MRE_ASSERT(formatSize > 0, "Vertex attribute format must have a known size");

    for ([[maybe_unused]] const VertexAttribute& attribute : m_attributes)
    {
        MRE_ASSERT(attribute.location != location, "Vertex attribute locations must be unique");
    }

    m_attributes.push_back(VertexAttribute{location, format, m_stride});
    m_stride += formatSize;
}

std::vector<DescriptorBinding> MergeShaderReflectionBindings(
    std::span<const Shader* const> shaders,
    std::uint32_t setIndex)
{
    std::vector<DescriptorBinding> merged{};

    for (const Shader* shader : shaders)
    {
        MRE_ASSERT(shader != nullptr, "MergeShaderReflectionBindings requires non-null shaders");
        for (const ShaderReflectionBinding& reflected : shader->GetReflection().bindings)
        {
            if (reflected.set != setIndex)
            {
                continue;
            }

            const auto existing = std::find_if(
                merged.begin(),
                merged.end(),
                [&reflected](const DescriptorBinding& binding) {
                    return binding.binding == reflected.binding;
                });

            if (existing == merged.end())
            {
                DescriptorBinding binding{};
                binding.binding = reflected.binding;
                binding.type = reflected.type;
                binding.stages = reflected.stages;
                binding.arrayCount = std::max(reflected.arrayCount, 1u);
                binding.updateAfterBind = false;
                merged.push_back(binding);
                continue;
            }

            MRE_ASSERT(
                existing->type == reflected.type,
                "Binding {} in set {} is reflected with conflicting descriptor types",
                reflected.binding,
                setIndex);
            existing->stages = static_cast<ShaderStageMask>(
                static_cast<std::uint8_t>(existing->stages) |
                static_cast<std::uint8_t>(reflected.stages));
        }
    }

    std::sort(
        merged.begin(),
        merged.end(),
        [](const DescriptorBinding& left, const DescriptorBinding& right) {
            return left.binding < right.binding;
        });
    return merged;
}

void ApplyShaderReflection(PipelineDesc& desc, std::span<const Shader* const> shaders)
{
    desc.vertexLayout = VertexLayout{};
    desc.pushConstantSize = 0;
    desc.pushConstantStages = ShaderStageMask::None;

    for (const Shader* shader : shaders)
    {
        MRE_ASSERT(shader != nullptr, "ApplyShaderReflection requires non-null shaders");
        const ShaderReflection& reflection = shader->GetReflection();

        if (reflection.stage == ShaderStage::Vertex && desc.vertexLayout.GetAttributes().empty())
        {
            std::vector<VertexAttribute> inputs = reflection.vertexInputs;
            std::sort(
                inputs.begin(),
                inputs.end(),
                [](const VertexAttribute& left, const VertexAttribute& right) {
                    return left.location < right.location;
                });
            for (const VertexAttribute& input : inputs)
            {
                desc.vertexLayout.AddAttribute(input.location, input.format);
            }
        }

        for (const ShaderReflectionPushConstant& range : reflection.pushConstants)
        {
            MRE_ASSERT(range.offset == 0, "Push constant offsets other than zero are not supported");
            desc.pushConstantSize = std::max(desc.pushConstantSize, range.size);
            desc.pushConstantStages = static_cast<ShaderStageMask>(
                static_cast<std::uint8_t>(desc.pushConstantStages) |
                static_cast<std::uint8_t>(range.stages));
        }
    }
}

std::uint32_t GetFormatSize(Format format)
{
    switch (format)
    {
    case Format::R8G8B8A8Unorm:
    case Format::R8G8B8A8Srgb:
    case Format::B8G8R8A8Unorm:
    case Format::B8G8R8A8Srgb:
    case Format::D32Sfloat:
        return 4;
    case Format::R16G16B16A16Sfloat:
    case Format::R32G32Sfloat:
        return 8;
    case Format::R32G32B32Sfloat:
        return 12;
    case Format::R32G32B32A32Sfloat:
        return 16;
    case Format::Undefined:
        return 0;
    }

    MRE_UNREACHABLE();
}

} // namespace mukki::rhi
