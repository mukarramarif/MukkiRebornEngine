#include "Renderer/Clouds/CloudNoise.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <glm/glm.hpp>

namespace mukki
{
namespace
{

TEST(CloudNoise, PerlinTilesSeamlessly)
{
    constexpr std::uint32_t kPeriod = 4;
    for (float z = 0.0f; z < 1.0f; z += 0.25f)
    {
        for (float y = 0.0f; y < 1.0f; y += 0.25f)
        {
            for (float x = 0.0f; x < 1.0f; x += 0.25f)
            {
                const glm::vec3 point{x, y, z};
                const float value = TileablePerlin(point, kPeriod, 7u);
                const float shifted =
                    TileablePerlin(point + glm::vec3{static_cast<float>(kPeriod), 0.0f, 0.0f}, kPeriod, 7u);
                EXPECT_NEAR(value, shifted, 1e-5f);
            }
        }
    }
}

TEST(CloudNoise, WorleyTilesSeamlessly)
{
    constexpr std::uint32_t kPeriod = 4;
    const glm::vec3 point{0.3f, 1.7f, 2.9f};
    const float value = TileableWorley(point, kPeriod, 13u);
    EXPECT_NEAR(value, TileableWorley(point + glm::vec3{0.0f, static_cast<float>(kPeriod), 0.0f}, kPeriod, 13u), 1e-5f);
    EXPECT_NEAR(value, TileableWorley(point + glm::vec3{static_cast<float>(kPeriod), 0.0f, 0.0f}, kPeriod, 13u), 1e-5f);
}

TEST(CloudNoise, OutputsStayInRange)
{
    for (std::uint32_t seed = 0; seed < 3; ++seed)
    {
        for (float z = 0.0f; z < 3.0f; z += 0.37f)
        {
            for (float y = 0.0f; y < 3.0f; y += 0.41f)
            {
                for (float x = 0.0f; x < 3.0f; x += 0.53f)
                {
                    const glm::vec3 point{x, y, z};
                    EXPECT_GE(TileablePerlin(point, 4, seed), 0.0f);
                    EXPECT_LE(TileablePerlin(point, 4, seed), 1.0f);
                    EXPECT_GE(TileableWorley(point, 4, seed), 0.0f);
                    EXPECT_LE(TileableWorley(point, 4, seed), 1.0f);
                    EXPECT_GE(TileableWorleyFbm(point, 4, seed, 3), 0.0f);
                    EXPECT_LE(TileableWorleyFbm(point, 4, seed, 3), 1.0f);
                }
            }
        }
    }
}

TEST(CloudNoise, SeedsChangeTheNoise)
{
    EXPECT_NE(
        TileablePerlin(glm::vec3{0.25f, 0.5f, 0.75f}, 4, 1u),
        TileablePerlin(glm::vec3{0.25f, 0.5f, 0.75f}, 4, 2u));
    EXPECT_EQ(
        TileablePerlin(glm::vec3{0.25f, 0.5f, 0.75f}, 4, 1u),
        TileablePerlin(glm::vec3{0.25f, 0.5f, 0.75f}, 4, 1u));
}

TEST(CloudNoise, BaseShapeHasVariation)
{
    constexpr std::uint32_t kResolution = 16;
    std::vector<std::uint8_t> pixels{};
    GenerateCloudBaseShape(kResolution, 5u, 0, kResolution, pixels);

    EXPECT_EQ(pixels.size(), static_cast<std::size_t>(kResolution) * kResolution * kResolution * 4);

    std::uint8_t minValue = 255;
    std::uint8_t maxValue = 0;
    for (std::size_t index = 0; index < pixels.size(); index += 4)
    {
        minValue = std::min(minValue, pixels[index]);
        maxValue = std::max(maxValue, pixels[index]);
    }
    EXPECT_LT(minValue, maxValue);
    EXPECT_LT(minValue, 128);
    EXPECT_GT(maxValue, 128);
}

TEST(CloudNoise, WeatherMapHasRegionalVariation)
{
    constexpr std::uint32_t kResolution = 32;
    std::vector<std::uint8_t> pixels{};
    GenerateCloudWeather(kResolution, 0.4f, 3u, 0, kResolution, pixels);

    ASSERT_EQ(pixels.size(), static_cast<std::size_t>(kResolution) * kResolution * 4);

    std::uint8_t minValue = 255;
    std::uint8_t maxValue = 0;
    std::uint64_t sum = 0;
    for (std::size_t index = 0; index < pixels.size(); index += 4)
    {
        minValue = std::min(minValue, pixels[index]);
        maxValue = std::max(maxValue, pixels[index]);
        sum += pixels[index];
    }

    const double average = static_cast<double>(sum) / static_cast<double>(pixels.size() / 4);
    EXPECT_LT(minValue, 128);
    EXPECT_GT(maxValue, 128);
    EXPECT_GT(average, 100.0); // centred around 0.5 (127.5)
    EXPECT_LT(average, 155.0);
}

} // namespace
} // namespace mukki
