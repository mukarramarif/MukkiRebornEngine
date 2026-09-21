#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Renderer/RHI/Resources.h"

namespace mukki::vulkan
{

class VkDescriptorSetLayout final : public rhi::DescriptorSetLayout
{
public:
    VkDescriptorSetLayout(const vk::raii::Device& device, const std::vector<rhi::DescriptorBinding>& bindings);

    [[nodiscard]] vk::DescriptorSetLayout GetHandle() const { return *m_layout; }

private:
    vk::raii::DescriptorSetLayout m_layout{nullptr};
};

class VkDescriptorSet final : public rhi::DescriptorSet
{
public:
    VkDescriptorSet(
        const vk::raii::Device& device,
        const vk::raii::DescriptorPool& pool,
        const VkDescriptorSetLayout& layout);

    void BindBuffer(
        std::uint32_t binding,
        const rhi::Buffer& buffer,
        std::uint64_t offset,
        std::uint64_t range) override;
    void BindTexture(std::uint32_t binding, const rhi::Texture& texture, const rhi::Sampler& sampler) override;

    [[nodiscard]] vk::DescriptorSet GetHandle() const { return *m_set; }

private:
    const vk::raii::Device* m_device = nullptr;
    vk::raii::DescriptorSet m_set{nullptr};
};

} // namespace mukki::vulkan
