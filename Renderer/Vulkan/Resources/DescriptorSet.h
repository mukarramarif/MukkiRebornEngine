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
    [[nodiscard]] const std::vector<rhi::DescriptorBinding>& GetBindings() const { return m_bindings; }

private:
    vk::raii::DescriptorSetLayout m_layout{nullptr};
    std::vector<rhi::DescriptorBinding> m_bindings{};
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
    void BindTexture(std::uint32_t binding, const rhi::Texture& texture, std::uint32_t arrayIndex) override;
    void BindSampler(std::uint32_t binding, const rhi::Sampler& sampler) override;
    void BindSampler(std::uint32_t binding, const rhi::Sampler& sampler, std::uint32_t arrayIndex) override;
    void BindStorageImage(std::uint32_t binding, const rhi::Texture& texture) override;
    void BindStorageBuffer(
        std::uint32_t binding,
        const rhi::Buffer& buffer,
        std::uint32_t arrayIndex) override;
    void BindAccelerationStructure(
        std::uint32_t binding,
        const rhi::AccelerationStructure& accelerationStructure) override;

    [[nodiscard]] vk::DescriptorSet GetHandle() const { return *m_set; }

private:
    [[nodiscard]] rhi::DescriptorType GetBindingType(std::uint32_t binding) const;

    const vk::raii::Device* m_device = nullptr;
    vk::raii::DescriptorSet m_set{nullptr};
    std::vector<rhi::DescriptorBinding> m_bindings{};
};

} // namespace mukki::vulkan
