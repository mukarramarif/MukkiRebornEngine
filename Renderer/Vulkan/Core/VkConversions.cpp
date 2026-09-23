#include "Renderer/Vulkan/Core/VkConversions.h"

#include "Utilities/Assert.h"

namespace mukki::vulkan
{

vk::Format ToVkFormat(rhi::Format format)
{
    switch (format)
    {
    case rhi::Format::R8G8B8A8Unorm:
        return vk::Format::eR8G8B8A8Unorm;
    case rhi::Format::R8G8B8A8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case rhi::Format::B8G8R8A8Unorm:
        return vk::Format::eB8G8R8A8Unorm;
    case rhi::Format::B8G8R8A8Srgb:
        return vk::Format::eB8G8R8A8Srgb;
    case rhi::Format::R16G16B16A16Sfloat:
        return vk::Format::eR16G16B16A16Sfloat;
    case rhi::Format::R32G32Sfloat:
        return vk::Format::eR32G32Sfloat;
    case rhi::Format::R32G32B32Sfloat:
        return vk::Format::eR32G32B32Sfloat;
    case rhi::Format::D32Sfloat:
        return vk::Format::eD32Sfloat;
    case rhi::Format::Undefined:
        return vk::Format::eUndefined;
    }

    MRE_UNREACHABLE();
}

rhi::Format FromVkFormat(vk::Format format)
{
    switch (format)
    {
    case vk::Format::eR8G8B8A8Unorm:
        return rhi::Format::R8G8B8A8Unorm;
    case vk::Format::eR8G8B8A8Srgb:
        return rhi::Format::R8G8B8A8Srgb;
    case vk::Format::eB8G8R8A8Unorm:
        return rhi::Format::B8G8R8A8Unorm;
    case vk::Format::eB8G8R8A8Srgb:
        return rhi::Format::B8G8R8A8Srgb;
    case vk::Format::eR16G16B16A16Sfloat:
        return rhi::Format::R16G16B16A16Sfloat;
    case vk::Format::eR32G32Sfloat:
        return rhi::Format::R32G32Sfloat;
    case vk::Format::eR32G32B32Sfloat:
        return rhi::Format::R32G32B32Sfloat;
    case vk::Format::eD32Sfloat:
        return rhi::Format::D32Sfloat;
    default:
        return rhi::Format::Undefined;
    }
}

vk::PresentModeKHR ToVkPresentMode(rhi::PresentMode mode)
{
    switch (mode)
    {
    case rhi::PresentMode::Fifo:
        return vk::PresentModeKHR::eFifo;
    case rhi::PresentMode::Mailbox:
        return vk::PresentModeKHR::eMailbox;
    case rhi::PresentMode::Immediate:
        return vk::PresentModeKHR::eImmediate;
    }

    MRE_UNREACHABLE();
}

vk::BufferUsageFlags ToVkBufferUsage(rhi::BufferUsage usage)
{
    switch (usage)
    {
    case rhi::BufferUsage::Vertex:
        return vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst;
    case rhi::BufferUsage::Index:
        return vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst;
    case rhi::BufferUsage::Uniform:
        return vk::BufferUsageFlagBits::eUniformBuffer;
    case rhi::BufferUsage::Storage:
        return vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
    case rhi::BufferUsage::Staging:
        return vk::BufferUsageFlagBits::eTransferSrc;
    }

    MRE_UNREACHABLE();
}

vk::ShaderStageFlagBits ToVkShaderStage(rhi::ShaderStage stage)
{
    switch (stage)
    {
    case rhi::ShaderStage::Vertex:
        return vk::ShaderStageFlagBits::eVertex;
    case rhi::ShaderStage::Fragment:
        return vk::ShaderStageFlagBits::eFragment;
    case rhi::ShaderStage::Compute:
        return vk::ShaderStageFlagBits::eCompute;
    }

    MRE_UNREACHABLE();
}

vk::ShaderStageFlags ToVkShaderStageMask(rhi::ShaderStageMask mask)
{
    vk::ShaderStageFlags flags{};
    const auto value = static_cast<std::uint8_t>(mask);
    if ((value & static_cast<std::uint8_t>(rhi::ShaderStageMask::Vertex)) != 0)
    {
        flags |= vk::ShaderStageFlagBits::eVertex;
    }
    if ((value & static_cast<std::uint8_t>(rhi::ShaderStageMask::Fragment)) != 0)
    {
        flags |= vk::ShaderStageFlagBits::eFragment;
    }
    if ((value & static_cast<std::uint8_t>(rhi::ShaderStageMask::Compute)) != 0)
    {
        flags |= vk::ShaderStageFlagBits::eCompute;
    }
    return flags;
}

vk::CullModeFlags ToVkCullMode(rhi::CullMode mode)
{
    switch (mode)
    {
    case rhi::CullMode::None:
        return vk::CullModeFlagBits::eNone;
    case rhi::CullMode::Front:
        return vk::CullModeFlagBits::eFront;
    case rhi::CullMode::Back:
        return vk::CullModeFlagBits::eBack;
    }

    MRE_UNREACHABLE();
}

vk::DescriptorType ToVkDescriptorType(rhi::DescriptorType type)
{
    switch (type)
    {
    case rhi::DescriptorType::UniformBuffer:
        return vk::DescriptorType::eUniformBuffer;
    case rhi::DescriptorType::CombinedImageSampler:
        return vk::DescriptorType::eCombinedImageSampler;
    case rhi::DescriptorType::SampledImage:
        return vk::DescriptorType::eSampledImage;
    case rhi::DescriptorType::Sampler:
        return vk::DescriptorType::eSampler;
    }

    MRE_UNREACHABLE();
}

} // namespace mukki::vulkan
