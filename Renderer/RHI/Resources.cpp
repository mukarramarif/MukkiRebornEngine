#include "Renderer/RHI/Resources.h"

#include "Utilities/Assert.h"

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
    case Format::Undefined:
        return 0;
    }

    MRE_UNREACHABLE();
}

} // namespace mukki::rhi
