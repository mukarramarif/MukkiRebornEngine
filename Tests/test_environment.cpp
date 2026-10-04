#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <glm/glm.hpp>

#include "Renderer/Lighting/Environment.h"

namespace mukki
{
namespace
{

TEST(Environment, GradientSkyMatchesItsColors)
{
    EnvironmentSettings settings{};
    settings.zenithColor = glm::vec3{0.2f, 0.4f, 0.8f};
    settings.horizonColor = glm::vec3{0.9f, 0.8f, 0.7f};
    settings.groundColor = glm::vec3{0.1f, 0.1f, 0.1f};
    settings.intensity = 1.0f;

    const glm::vec3 zenith = EvaluateSkyRadiance(settings, glm::vec3{0.0f, 1.0f, 0.0f});
    EXPECT_NEAR(zenith.r, 0.2f, 1e-4f);
    EXPECT_NEAR(zenith.g, 0.4f, 1e-4f);
    EXPECT_NEAR(zenith.b, 0.8f, 1e-4f);

    const glm::vec3 horizon = EvaluateSkyRadiance(settings, glm::vec3{1.0f, 0.0f, 0.0f});
    EXPECT_NEAR(horizon.r, 0.9f, 1e-4f);
    EXPECT_NEAR(horizon.g, 0.8f, 1e-4f);
    EXPECT_NEAR(horizon.b, 0.7f, 1e-4f);

    const glm::vec3 ground = EvaluateSkyRadiance(settings, glm::vec3{0.0f, -1.0f, 0.0f});
    EXPECT_NEAR(ground.r, 0.1f, 1e-4f);
    EXPECT_NEAR(ground.g, 0.1f, 1e-4f);
    EXPECT_NEAR(ground.b, 0.1f, 1e-4f);
}

TEST(Environment, ConstantSkyShEqualsRadianceOverPi)
{
    EnvironmentSettings settings{};
    settings.zenithColor = glm::vec3{0.5f, 0.5f, 0.5f};
    settings.horizonColor = glm::vec3{0.5f, 0.5f, 0.5f};
    settings.groundColor = glm::vec3{0.5f, 0.5f, 0.5f};
    settings.intensity = 1.0f;

    const EnvironmentSh sh = ComputeEnvironmentSh(settings);

    // A constant environment of radiance L has irradiance / pi = L (the band factors fold the
    // Lambert convolution), and only the L0 band is non-zero. Evaluating the basis against a
    // normal returns the coefficient times Y_00.
    constexpr float kY00 = 0.282095f;
    EXPECT_NEAR(sh.bands[0].r * kY00, 0.5f, 1e-3f);
    EXPECT_NEAR(sh.bands[0].g * kY00, 0.5f, 1e-3f);
    EXPECT_NEAR(sh.bands[0].b * kY00, 0.5f, 1e-3f);
    for (std::size_t band = 1; band < 9; ++band)
    {
        EXPECT_NEAR(sh.bands[band].r, 0.0f, 1e-3f);
        EXPECT_NEAR(sh.bands[band].g, 0.0f, 1e-3f);
        EXPECT_NEAR(sh.bands[band].b, 0.0f, 1e-3f);
    }
}

TEST(Environment, ShIrradianceFavorsTheBrightSide)
{
    EnvironmentSettings settings{};
    settings.zenithColor = glm::vec3{1.0f, 1.0f, 1.0f};
    settings.horizonColor = glm::vec3{0.0f, 0.0f, 0.0f};
    settings.groundColor = glm::vec3{0.0f, 0.0f, 0.0f};
    settings.intensity = 1.0f;

    const EnvironmentSh sh = ComputeEnvironmentSh(settings);

    const auto evaluate = [&sh](const glm::vec3& direction) {
        glm::vec3 result{0.0f};
        const float x = direction.x;
        const float y = direction.y;
        const float z = direction.z;
        const float basis[9] = {
            0.282095f,
            0.488603f * x,
            0.488603f * y,
            0.488603f * z,
            1.092548f * x * y,
            1.092548f * y * z,
            0.315392f * (3.0f * z * z - 1.0f),
            1.092548f * x * z,
            0.546274f * (x * x - y * y)};
        for (std::size_t band = 0; band < 9; ++band)
        {
            result += glm::vec3{sh.bands[band]} * basis[band];
        }
        return result;
    };

    const glm::vec3 up = evaluate(glm::vec3{0.0f, 1.0f, 0.0f});
    const glm::vec3 down = evaluate(glm::vec3{0.0f, -1.0f, 0.0f});
    EXPECT_GT(up.r, down.r);
    EXPECT_GT(up.r, 0.1f);
}

TEST(Environment, BrdfLutStaysInRange)
{
    std::vector<float> pixels{};
    BuildEnvironmentBrdfLut(pixels, 16);
    ASSERT_EQ(pixels.size(), 16u * 16u * 2u);

    for (std::size_t index = 0; index < pixels.size(); ++index)
    {
        ASSERT_TRUE(std::isfinite(pixels[index])) << "index " << index;
        EXPECT_GE(pixels[index], 0.0f);
        EXPECT_LE(pixels[index], 1.0f);
    }

    // Smooth surfaces (bottom row, low roughness) behave close to a single Fresnel bounce.
    EXPECT_GT(pixels[(0u * 16u + 8u) * 2u + 1u], pixels[(15u * 16u + 8u) * 2u + 1u]);
}

} // namespace
} // namespace mukki
