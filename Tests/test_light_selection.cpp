#include <array>
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

} // namespace
} // namespace mukki
