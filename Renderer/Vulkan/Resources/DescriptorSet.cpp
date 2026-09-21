#include "Renderer/Vulkan/Resources/DescriptorSet.h"

#include "Renderer/Vulkan/Core/VkConversions.h"
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
{
    MRE_ASSERT(!bindings.empty(), "Descriptor set layout requires at least one binding");

    std::vector<vk::DescriptorSetLayoutBinding> nativeBindings{};
    nativeBindings.reserve(bindings.size());
    for (const rhi::DescriptorBinding& binding : bindings)
    {
        const vk::DescriptorType type = binding.type == rhi::DescriptorType::UniformBuffer
                                            ? vk::DescriptorType::eUniformBuffer
                                            : vk::DescriptorType::eCombinedImageSampler;
        nativeBindings.push_back(vk::DescriptorSetLayoutBinding{}
                                     .setBinding(binding.binding)
                                     .setDescriptorType(type)
                                     .setDescriptorCount(1)
                                     .setStageFlags(ToVkShaderStageMask(binding.stages)));
    }

    const vk::DescriptorSetLayoutCreateInfo createInfo =
        vk::DescriptorSetLayoutCreateInfo{}.setBindings(nativeBindings);
    m_layout = vk::raii::DescriptorSetLayout{device, createInfo};
}

VkDescriptorSet::VkDescriptorSet(
    const vk::raii::Device& device,
    const vk::raii::DescriptorPool& pool,
    const VkDescriptorSetLayout& layout)
    : m_device(&device)
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

    const auto& vkBuffer = static_cast<const VkBuffer&>(buffer);
    const vk::DescriptorBufferInfo bufferInfo = vk::DescriptorBufferInfo{}
                                                    .setBuffer(vkBuffer.GetHandle())
                                                    .setOffset(offset)
                                                    .setRange(range == 0 ? VK_WHOLE_SIZE : range);

    const vk::WriteDescriptorSet write = vk::WriteDescriptorSet{}
                                             .setDstSet(*m_set)
                                             .setDstBinding(binding)
                                             .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                                             .setBufferInfo(bufferInfo);

    m_device->updateDescriptorSets(write, {});
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

} // namespace mukki::vulkan
