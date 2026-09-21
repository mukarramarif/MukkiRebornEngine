#include "Renderer/RHI/Resources.h"

#include <gtest/gtest.h>

namespace
{

TEST(Resources, FormatSizes)
{
    using mukki::rhi::Format;

    EXPECT_EQ(mukki::rhi::GetFormatSize(Format::R8G8B8A8Unorm), 4u);
    EXPECT_EQ(mukki::rhi::GetFormatSize(Format::R8G8B8A8Srgb), 4u);
    EXPECT_EQ(mukki::rhi::GetFormatSize(Format::B8G8R8A8Unorm), 4u);
    EXPECT_EQ(mukki::rhi::GetFormatSize(Format::B8G8R8A8Srgb), 4u);
    EXPECT_EQ(mukki::rhi::GetFormatSize(Format::R16G16B16A16Sfloat), 8u);
    EXPECT_EQ(mukki::rhi::GetFormatSize(Format::R32G32Sfloat), 8u);
    EXPECT_EQ(mukki::rhi::GetFormatSize(Format::R32G32B32Sfloat), 12u);
    EXPECT_EQ(mukki::rhi::GetFormatSize(Format::D32Sfloat), 4u);
    EXPECT_EQ(mukki::rhi::GetFormatSize(Format::Undefined), 0u);
}

TEST(Resources, VertexLayoutOffsetsAndStride)
{
    mukki::rhi::VertexLayout layout{};
    layout.AddAttribute(0, mukki::rhi::Format::R32G32Sfloat);
    layout.AddAttribute(1, mukki::rhi::Format::R32G32B32Sfloat);

    const auto& attributes = layout.GetAttributes();
    ASSERT_EQ(attributes.size(), 2u);
    EXPECT_EQ(attributes[0].location, 0u);
    EXPECT_EQ(attributes[0].offset, 0u);
    EXPECT_EQ(attributes[1].location, 1u);
    EXPECT_EQ(attributes[1].offset, 8u);
    EXPECT_EQ(layout.GetStride(), 20u);
}

TEST(Resources, EmptyVertexLayoutHasZeroStride)
{
    const mukki::rhi::VertexLayout layout{};
    EXPECT_TRUE(layout.GetAttributes().empty());
    EXPECT_EQ(layout.GetStride(), 0u);
}

} // namespace
