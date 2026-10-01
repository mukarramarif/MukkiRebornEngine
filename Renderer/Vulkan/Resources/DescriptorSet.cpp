#include "Renderer/Vulkan/Resources/DescriptorSet.h"

#include <array>

#include "Renderer/Vulkan/Core/VkConversions.h"
#include "Renderer/Vulkan/Resources/AccelerationStructure.h"
#include "Renderer/Vulkan/Resources/Buffer.h"
#include "Renderer/Vulkan/Resources/Sampler.h"
#include "Renderer/Vulkan/Resources/Texture.h"
#include "Utilities/Assert.h"

#include <utility>

namespace mukki::vulkan
{

VkDescriptorSetLayout::VkDescriptorSetLayout(
    const vk::raii::Device& device,
    const std::vector<rhi::DescriptorBinding>& bindings)
    : m_bindings(bindings)
{
    MRE_ASSERT(!bindings.empty(), "Descriptor set layout requires at least one binding");

    std::vector<vk::DescriptorSetLayoutBinding> nativeBindings{};
    nativeBindings.reserve(bindings.size());

    std::vector<vk::DescriptorBindingFlags> bindingFlags{};
    bindingFlags.reserve(bindings.size());

    bool hasUpdateAfterBind = false;

    for (const rhi::DescriptorBinding& binding : bindings)
    {
        nativeBindings.push_back(vk::DescriptorSetLayoutBinding{}
                                     .setBinding(binding.binding)
                                     .setDescriptorType(ToVkDescriptorType(binding.type))
                                     .setDescriptorCount(binding.arrayCount)
                                     .setStageFlags(ToVkShaderStageMask(binding.stages)));

        vk::DescriptorBindingFlags flags{};
        if (binding.updateAfterBind)
        {
            flags |= vk::DescriptorBindingFlagBits::eUpdateAfterBind;
            flags |= vk::DescriptorBindingFlagBits::ePartiallyBound;
            hasUpdateAfterBind = true;
        }
        bindingFlags.push_back(flags);
    }

    const vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{bindingFlags};

    vk::DescriptorSetLayoutCreateInfo createInfo = vk::DescriptorSetLayoutCreateInfo{}.setBindings(nativeBindings);
    if (hasUpdateAfterBind)
    {
        createInfo.pNext = &bindingFlagsInfo;
        createInfo.flags |= vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
    }

    m_layout = vk::raii::DescriptorSetLayout{device, createInfo};
}

VkDescriptorSet::VkDescriptorSet(
    const vk::raii::Device& device,
    const vk::raii::DescriptorPool& pool,
    const VkDescriptorSetLayout& layout)
    : m_device(&device)
    , m_bindings(layout.GetBindings())
{
    const vk::DescriptorSetLayout layoutHandle = layout.GetHandle();
    const vk::DescriptorSetAllocateInfo allocateInfo = vk::DescriptorSetAllocateInfo{}
                                                           .setDescriptorPool(*pool)
                                                           .setSetLayouts(layoutHandle);

    std::vector<vk::raii::DescriptorSet> descriptorSets = device.allocateDescriptorSets(allocateInfo);
    MRE_ASSERT(descriptorSets.size() == 1, "Expected exactly one descriptor set");
    m_set = std::move(descriptorSets.front());
}

void VkDescriptorSet::BindBuffer(
    std::uint32_t binding,
    const rhi::Buffer& buffer,
    std::uint64_t offset,
    std::uint64_t range)
{
    MRE_ASSERT(m_device != nullptr, "VkDescriptorSet requires a valid device");
    MRE_ASSERT(m_set != nullptr, "VkDescriptorSet requires a valid descriptor set");

    const rhi::DescriptorType bindingType = GetBindingType(binding);
    MRE_ASSERT(
        bindingType == rhi::DescriptorType::UniformBuffer || bindingType == rhi::DescriptorType::StorageBuffer,
        "BindBuffer requires a uniform or storage buffer binding");

    const auto& vkBuffer = static_cast<const VkBuffer&>(buffer);
    const vk::DescriptorBufferInfo bufferInfo = vk::DescriptorBufferInfo{}
                                                    .setBuffer(vkBuffer.GetHandle())
                                                    .setOffset(offset)
                                                    .setRange(range == 0 ? VK_WHOLE_SIZE : range);

    const vk::WriteDescriptorSet write = vk::WriteDescriptorSet{}
                                             .setDstSet(*m_set)
                                             .setDstBinding(binding)
                                             .setDescriptorType(ToVkDescriptorType(bindingType))
                                             .setBufferInfo(bufferInfo);

    m_device->updateDescriptorSets(write, {});
}

rhi::DescriptorType VkDescriptorSet::GetBindingType(std::uint32_t binding) const
{
    for (const rhi::DescriptorBinding& candidate : m_bindings)
    {
        if (candidate.binding == binding)
        {
            return candidate.type;
        }
    }

    MRE_ASSERT(false, "Descriptor set layout has no binding {}", binding);
    return rhi::DescriptorType::UniformBuffer;
}

void VkDescriptorSet::BindTexture(
    std::uint32_t binding,
    const rhi::Texture& texture,
    const rhi::Sampler& sampler)
{
    MRE_ASSERT(m_device != nullptr, "VkDescriptorSet requires a valid device");
    MRE_ASSERT(m_set != nullptr, "VkDescriptorSet requires a valid descriptor set");

    const auto& vkTexture = static_cast<const VkTexture&>(texture);
    const auto& vkSampler = static_cast<const VkSampler&>(sampler);
    MRE_ASSERT(
        vkTexture.GetLayout() == vk::ImageLayout::eShaderReadOnlyOptimal,
        "Textures must be in a shader readable layout before binding");

    const vk::DescriptorImageInfo imageInfo = vk::DescriptorImageInfo{}
                                                  .setSampler(vkSampler.GetHandle())
                                                  .setImageView(vkTexture.GetImageView())
                                                  .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    const vk::WriteDescriptorSet write = vk::WriteDescriptorSet{}
                                             .setDstSet(*m_set)
                                             .setDstBinding(binding)
                                             .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                                             .setImageInfo(imageInfo);

    m_device->updateDescriptorSets(write, {});
}

void VkDescriptorSet::BindTexture(std::uint32_t binding, const rhi::Texture& texture, std::uint32_t arrayIndex)
{
    MRE_ASSERT(m_device != nullptr, "VkDescriptorSet requires a valid device");
    MRE_ASSERT(m_set != nullptr, "VkDescriptorSet requires a valid descriptor set");

    const auto& vkTexture = static_cast<const VkTexture&>(texture);
    MRE_ASSERT(
        vkTexture.GetLayout() == vk::ImageLayout::eShaderReadOnlyOptimal,
        "Textures must be in a shader readable layout before binding");

    const vk::DescriptorImageInfo imageInfo = vk::DescriptorImageInfo{}
                                                  .setImageView(vkTexture.GetImageView())
                                                  .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    const vk::WriteDescriptorSet write = vk::WriteDescriptorSet{}
                                             .setDstSet(*m_set)
                                             .setDstBinding(binding)
                                             .setDstArrayElement(arrayIndex)
                                             .setDescriptorType(vk::DescriptorType::eSampledImage)
                                             .setImageInfo(imageInfo);

    m_device->updateDescriptorSets(write, {});
}

void VkDescriptorSet::BindSampler(std::uint32_t binding, const rhi::Sampler& sampler)
{
    MRE_ASSERT(m_device != nullptr, "VkDescriptorSet requires a valid device");
    MRE_ASSERT(m_set != nullptr, "VkDescriptorSet requires a valid descriptor set");

    const auto& vkSampler = static_cast<const VkSampler&>(sampler);
    const vk::DescriptorImageInfo imageInfo = vk::DescriptorImageInfo{}.setSampler(vkSampler.GetHandle());

    const vk::WriteDescriptorSet write = vk::WriteDescriptorSet{}
                                             .setDstSet(*m_set)
                                             .setDstBinding(binding)
                                             .setDescriptorType(vk::DescriptorType::eSampler)
                                             .setImageInfo(imageInfo);

    m_device->updateDescriptorSets(write, {});
}

void VkDescriptorSet::BindAccelerationStructure(
    std::uint32_t binding,
    const rhi::AccelerationStructure& accelerationStructure)
{
    MRE_ASSERT(m_device != nullptr, "VkDescriptorSet requires a valid device");
    MRE_ASSERT(m_set != nullptr, "VkDescriptorSet requires a valid descriptor set");

    const auto& vkAccelerationStructure = static_cast<const VkAccelerationStructure&>(accelerationStructure);
    const std::array<vk::AccelerationStructureKHR, 1> structures{vkAccelerationStructure.GetStructure()};
    const vk::WriteDescriptorSetAccelerationStructureKHR accelerationStructureInfo =
        vk::WriteDescriptorSetAccelerationStructureKHR{}.setAccelerationStructures(structures);

    const vk::WriteDescriptorSet write = vk::WriteDescriptorSet{}
                                             .setDstSet(*m_set)
                                             .setDstBinding(binding)
                                             .setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR)
                                             .setDescriptorCount(1)
                                             .setPNext(&accelerationStructureInfo);

    m_device->updateDescriptorSets(write, {});
}

void VkDescriptorSet::BindSampler(std::uint32_t binding, const rhi::Sampler& sampler, std::uint32_t arrayIndex)
{
    MRE_ASSERT(m_device != nullptr, "VkDescriptorSet requires a valid device");
    MRE_ASSERT(m_set != nullptr, "VkDescriptorSet requires a valid descriptor set");

    const auto& vkSampler = static_cast<const VkSampler&>(sampler);
    const vk::DescriptorImageInfo imageInfo = vk::DescriptorImageInfo{}.setSampler(vkSampler.GetHandle());

    const vk::WriteDescriptorSet write = vk::WriteDescriptorSet{}
                                             .setDstSet(*m_set)
                                             .setDstBinding(binding)
                                             .setDstArrayElement(arrayIndex)
                                             .setDescriptorType(vk::DescriptorType::eSampler)
                                             .setImageInfo(imageInfo);

    m_device->updateDescriptorSets(write, {});
}

} // namespace mukki::vulkan
