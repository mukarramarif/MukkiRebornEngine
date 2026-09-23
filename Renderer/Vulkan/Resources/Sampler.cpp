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
                                                  .setCompareEnable(false)
                                                  .setBorderColor(vk::BorderColor::eFloatOpaqueBlack)
                                                  .setUnnormalizedCoordinates(false);

    m_sampler = vk::raii::Sampler{device, createInfo};
}

} // namespace mukki::vulkan
