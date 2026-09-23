#include "Renderer/Vulkan/Core/VkConversions.h"

#include <gtest/gtest.h>

namespace
{

TEST(VkConversions, FormatsRoundTrip)
{
    using mukki::rhi::Format;

    const Format formats[] = {
        Format::R8G8B8A8Unorm,
        Format::R8G8B8A8Srgb,
        Format::B8G8R8A8Unorm,
        Format::B8G8R8A8Srgb,
        Format::R16G16B16A16Sfloat,
        Format::D32Sfloat,
    };

    for (const Format format : formats)
    {
        const vk::Format vkFormat = mukki::vulkan::ToVkFormat(format);
        EXPECT_NE(vkFormat, vk::Format::eUndefined);
        EXPECT_EQ(mukki::vulkan::FromVkFormat(vkFormat), format);
    }
}

TEST(VkConversions, UndefinedMapsToUndefined)
{
    EXPECT_EQ(mukki::vulkan::ToVkFormat(mukki::rhi::Format::Undefined), vk::Format::eUndefined);
    EXPECT_EQ(mukki::vulkan::FromVkFormat(vk::Format::eR32G32B32A32Sfloat), mukki::rhi::Format::Undefined);
}

TEST(VkConversions, PresentModes)
{
    EXPECT_EQ(mukki::vulkan::ToVkPresentMode(mukki::rhi::PresentMode::Fifo), vk::PresentModeKHR::eFifo);
    EXPECT_EQ(mukki::vulkan::ToVkPresentMode(mukki::rhi::PresentMode::Mailbox), vk::PresentModeKHR::eMailbox);
    EXPECT_EQ(mukki::vulkan::ToVkPresentMode(mukki::rhi::PresentMode::Immediate), vk::PresentModeKHR::eImmediate);
}

TEST(VkConversions, DescriptorTypes)
{
    using mukki::rhi::DescriptorType;

    EXPECT_EQ(mukki::vulkan::ToVkDescriptorType(DescriptorType::UniformBuffer), vk::DescriptorType::eUniformBuffer);
    EXPECT_EQ(
        mukki::vulkan::ToVkDescriptorType(DescriptorType::CombinedImageSampler),
        vk::DescriptorType::eCombinedImageSampler);
    EXPECT_EQ(mukki::vulkan::ToVkDescriptorType(DescriptorType::SampledImage), vk::DescriptorType::eSampledImage);
    EXPECT_EQ(mukki::vulkan::ToVkDescriptorType(DescriptorType::Sampler), vk::DescriptorType::eSampler);
}

} // namespace
