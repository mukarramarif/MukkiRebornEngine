#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "Renderer/GlobalIllumination/GiTypes.h"
#include "Renderer/Lighting/Light.h"

namespace mukki
{
namespace
{

Light MakeLight(LightType type, float intensity = 1.0f, bool castShadows = true)
{
    Light light{};
    light.type = type;
    light.intensity = intensity;
    light.castShadows = castShadows;
    return light;
}

TEST(GiLightSelection, KeepsOnlyPunctualLights)
{
    const std::array<Light, 5> lights{
        MakeLight(LightType::Directional),
        MakeLight(LightType::Point),
        MakeLight(LightType::Spot),
        MakeLight(LightType::Point),
        MakeLight(LightType::Directional)};

    const std::vector<GpuLight> selected = SelectGiLights(lights, kMaxGiLights);
    ASSERT_EQ(selected.size(), 3u);
    EXPECT_FLOAT_EQ(selected[0].params.x, static_cast<float>(LightType::Point));
    EXPECT_FLOAT_EQ(selected[1].params.x, static_cast<float>(LightType::Spot));
    EXPECT_FLOAT_EQ(selected[2].params.x, static_cast<float>(LightType::Point));
}

TEST(GiLightSelection, HonorsTheCountCap)
{
    std::vector<Light> lights(20, MakeLight(LightType::Point));

    const std::vector<GpuLight> selected = SelectGiLights(lights, 4);
    EXPECT_EQ(selected.size(), 4u);
}

TEST(GiLightSelection, PacksCastShadowsIntoTheUnusedParamsSlot)
{
    const std::array<Light, 2> lights{
        MakeLight(LightType::Point, 1.0f, true),
        MakeLight(LightType::Spot, 1.0f, false)};

    const std::vector<GpuLight> selected = SelectGiLights(lights, kMaxGiLights);
    ASSERT_EQ(selected.size(), 2u);
    EXPECT_FLOAT_EQ(selected[0].params.w, 1.0f);
    EXPECT_FLOAT_EQ(selected[1].params.w, 0.0f);
}

TEST(GiLightSelection, ZeroCapSelectsNothing)
{
    const std::array<Light, 1> lights{MakeLight(LightType::Point)};
    EXPECT_TRUE(SelectGiLights(lights, 0).empty());
}

// The probe pass shades hits with these packed values, so a spot light must arrive with its
// normalized direction, cone cosines, color and intensity intact.
TEST(GiLightSelection, PacksSpotDataUsedByTheProbeShading)
{
    Light light{};
    light.type = LightType::Spot;
    light.position = glm::vec3{1.0f, 2.0f, 3.0f};
    light.direction = glm::vec3{0.0f, -2.0f, 0.0f};
    light.color = glm::vec3{0.25f, 0.5f, 0.75f};
    light.intensity = 7.0f;
    light.range = 33.0f;
    light.innerConeAngle = 0.3f;
    light.outerConeAngle = 0.6f;

    const std::array<Light, 1> lights{light};
    const std::vector<GpuLight> selected = SelectGiLights(lights, kMaxGiLights);
    ASSERT_EQ(selected.size(), 1u);

    const GpuLight& packed = selected[0];
    EXPECT_FLOAT_EQ(packed.positionRange.x, 1.0f);
    EXPECT_FLOAT_EQ(packed.positionRange.y, 2.0f);
    EXPECT_FLOAT_EQ(packed.positionRange.z, 3.0f);
    EXPECT_FLOAT_EQ(packed.positionRange.w, 33.0f);
    EXPECT_NEAR(packed.directionIntensity.x, 0.0f, 1.0e-6f);
    EXPECT_NEAR(packed.directionIntensity.y, -1.0f, 1.0e-6f);
    EXPECT_NEAR(packed.directionIntensity.z, 0.0f, 1.0e-6f);
    EXPECT_FLOAT_EQ(packed.directionIntensity.w, 7.0f);
    EXPECT_FLOAT_EQ(packed.color.x, 0.25f);
    EXPECT_FLOAT_EQ(packed.color.y, 0.5f);
    EXPECT_FLOAT_EQ(packed.color.z, 0.75f);
    EXPECT_FLOAT_EQ(packed.params.x, static_cast<float>(LightType::Spot));
    EXPECT_NEAR(packed.params.y, std::cos(0.3f), 1.0e-5f);
    EXPECT_NEAR(packed.params.z, std::cos(0.6f), 1.0e-5f);
}

} // namespace
} // namespace mukki
