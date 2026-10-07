#include <gtest/gtest.h>

#include <glm/vec3.hpp>

#include "Renderer/GlobalIllumination/GiTypes.h"

namespace mukki
{
namespace
{

TEST(GiProbes, UniformProbeReturnsConstantIrradiance)
{
    GpuGiProbe probe{};
    probe.sh0 = glm::vec4{2.0f, 1.0f, 0.5f, 1.0f};

    for (const glm::vec3 normal : {glm::vec3{0.0f, 1.0f, 0.0f}, glm::vec3{1.0f, 0.0f, 0.0f}})
    {
        const glm::vec3 irradiance = EvaluateProbeIrradiance(probe, normal);
        EXPECT_FLOAT_EQ(irradiance.r, 2.0f);
        EXPECT_FLOAT_EQ(irradiance.g, 1.0f);
        EXPECT_FLOAT_EQ(irradiance.b, 0.5f);
    }
}

TEST(GiProbes, DirectionalTermsFollowTheNormal)
{
    GpuGiProbe probe{};
    probe.shX = glm::vec4{1.0f, 0.0f, 0.0f, 0.0f};

    EXPECT_FLOAT_EQ(EvaluateProbeIrradiance(probe, glm::vec3{1.0f, 0.0f, 0.0f}).r, 1.0f);
    EXPECT_FLOAT_EQ(EvaluateProbeIrradiance(probe, glm::vec3{0.0f, 1.0f, 0.0f}).r, 0.0f);
    EXPECT_FLOAT_EQ(EvaluateProbeIrradiance(probe, glm::vec3{-1.0f, 0.0f, 0.0f}).r, -1.0f);
}

} // namespace
} // namespace mukki
