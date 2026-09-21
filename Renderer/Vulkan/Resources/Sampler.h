#pragma once

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Renderer/RHI/Resources.h"

namespace mukki::vulkan
{

class VkSampler final : public rhi::Sampler
{
public:
    VkSampler(const vk::raii::Device& device, const rhi::SamplerDesc& desc);

    [[nodiscard]] vk::Sampler GetHandle() const { return *m_sampler; }

private:
    vk::raii::Sampler m_sampler{nullptr};
};

} // namespace mukki::vulkan
