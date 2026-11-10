#include "Renderer/Lighting/Subsurface.h"

#include <cmath>

#include <gtest/gtest.h>

namespace
{

std::size_t LutOffset(std::uint32_t x, std::uint32_t y)
{
    return (static_cast<std::size_t>(y) * mukki::kSubsurfaceLutWidth + x) * 4;
}

TEST(SubsurfaceLut, HasTheExpectedSize)
{
    std::vector<float> pixels{};
    mukki::BuildSubsurfaceLut(pixels);
    EXPECT_EQ(
        pixels.size(),
        static_cast<std::size_t>(mukki::kSubsurfaceLutWidth) * mukki::kSubsurfaceLutHeight * 4);
}

TEST(SubsurfaceLut, CenterTapIsUnity)
{
    std::vector<float> pixels{};
    mukki::BuildSubsurfaceLut(pixels);

    for (std::uint32_t y = 0; y < mukki::kSubsurfaceLutHeight; ++y)
    {
        const std::size_t offset = LutOffset(0, y);
        EXPECT_NEAR(pixels[offset + 0], 1.0f, 1.0e-5f);
        EXPECT_NEAR(pixels[offset + 1], 1.0f, 1.0e-5f);
        EXPECT_NEAR(pixels[offset + 2], 1.0f, 1.0e-5f);
    }
}

TEST(SubsurfaceLut, FallsOffMonotonicallyAlongDistance)
{
    std::vector<float> pixels{};
    mukki::BuildSubsurfaceLut(pixels);

    for (std::uint32_t y = 0; y < mukki::kSubsurfaceLutHeight; ++y)
    {
        float previous = 2.0f;
        for (std::uint32_t x = 0; x < mukki::kSubsurfaceLutWidth; ++x)
        {
            const float weight = pixels[LutOffset(x, y)];
            EXPECT_LE(weight, previous + 1.0e-6f);
            previous = weight;
        }
    }
}

// Red scatters over the longest radius, blue over the shortest, so for any off-center tap the red
// weight is the largest and the blue weight the smallest.
TEST(SubsurfaceLut, ChannelsUseDifferentScatterRadii)
{
    std::vector<float> pixels{};
    mukki::BuildSubsurfaceLut(pixels);

    const std::uint32_t middleBucket = mukki::kSubsurfaceLutHeight / 2;
    for (std::uint32_t x = 1; x < mukki::kSubsurfaceLutWidth; ++x)
    {
        const std::size_t offset = LutOffset(x, middleBucket);
        EXPECT_GT(pixels[offset + 0], pixels[offset + 1]);
        EXPECT_GT(pixels[offset + 1], pixels[offset + 2]);
    }
}

// A larger scattering radius keeps more energy at the same normalized distance.
TEST(SubsurfaceLut, LargerRadiusBucketsFallOffSlower)
{
    std::vector<float> pixels{};
    mukki::BuildSubsurfaceLut(pixels);

    const std::uint32_t lastColumn = mukki::kSubsurfaceLutWidth - 1;
    float previous = -1.0f;
    for (std::uint32_t y = 0; y < mukki::kSubsurfaceLutHeight; ++y)
    {
        const float weight = pixels[LutOffset(lastColumn, y)];
        EXPECT_GE(weight, previous - 1.0e-6f);
        previous = weight;
    }
}

} // namespace
