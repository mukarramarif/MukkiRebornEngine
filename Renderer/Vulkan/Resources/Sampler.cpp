#include "Renderer/Vulkan/Resources/Sampler.h"

#include "Utilities/Assert.h"

namespace mukki::vulkan
{
namespace
{

vk::Filter ToVkFilter(rhi::Filter filter)
{
    return filter == rhi::Filter::Linear ? vk::Filter::eLinear : vk::Filter::eNearest;
}

vk::SamplerAddressMode ToVkAddressMode(rhi::AddressMode mode)
{
    switch (mode)
    {
    case rhi::AddressMode::Repeat:
        return vk::SamplerAddressMode::eRepeat;
    case rhi::AddressMode::ClampToEdge:
        return vk::SamplerAddressMode::eClampToEdge;
    case rhi::AddressMode::MirroredRepeat:
        return vk::SamplerAddressMode::eMirroredRepeat;
    }

    MRE_UNREACHABLE();
}

vk::CompareOp ToVkCompareOp(rhi::CompareOp op)
{
    switch (op)
    {
    case rhi::CompareOp::Never:
        return vk::CompareOp::eNever;
    case rhi::CompareOp::Less:
        return vk::CompareOp::eLess;
    case rhi::CompareOp::Equal:
        return vk::CompareOp::eEqual;
    case rhi::CompareOp::LessEqual:
        return vk::CompareOp::eLessOrEqual;
    case rhi::CompareOp::Greater:
        return vk::CompareOp::eGreater;
    case rhi::CompareOp::NotEqual:
        return vk::CompareOp::eNotEqual;
    case rhi::CompareOp::GreaterEqual:
        return vk::CompareOp::eGreaterOrEqual;
    case rhi::CompareOp::Always:
        return vk::CompareOp::eAlways;
    }

    MRE_UNREACHABLE();
}

} // namespace

VkSampler::VkSampler(const vk::raii::Device& device, const rhi::SamplerDesc& desc)
{
    const vk::SamplerCreateInfo createInfo = vk::SamplerCreateInfo{}
                                                  .setMagFilter(ToVkFilter(desc.magFilter))
                                                  .setMinFilter(ToVkFilter(desc.minFilter))
                                                  .setMipmapMode(vk::SamplerMipmapMode::eNearest)
                                                  .setAddressModeU(ToVkAddressMode(desc.addressModeU))
                                                  .setAddressModeV(ToVkAddressMode(desc.addressModeV))
                                                  .setAddressModeW(ToVkAddressMode(desc.addressModeU))
                                                  .setAnisotropyEnable(false)
                                                  .setCompareEnable(desc.compareEnable)
                                                  .setCompareOp(ToVkCompareOp(desc.compareOp))
                                                  .setBorderColor(vk::BorderColor::eFloatOpaqueBlack)
                                                  .setUnnormalizedCoordinates(false);

    m_sampler = vk::raii::Sampler{device, createInfo};
}

} // namespace mukki::vulkan
