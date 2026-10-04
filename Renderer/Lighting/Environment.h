#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace mukki
{

// Parametric sky used as the image based lighting source. The gradient depends only on the world
// direction; the sun disc is part of the visible background but is excluded from the SH/prefilter
// so the analytic directional light is not double counted.
//
// When `environmentMap` points to a Radiance .hdr file it replaces the gradient sky everywhere
// (SH irradiance, prefiltered specular, background and GI sky rays). Intensity scales both.
struct EnvironmentSettings
{
    bool enabled = true;
    float intensity = 1.0f;
    glm::vec3 zenithColor{0.16f, 0.28f, 0.52f};
    glm::vec3 horizonColor{0.62f, 0.70f, 0.80f};
    glm::vec3 groundColor{0.18f, 0.16f, 0.14f};
    float sunDiscIntensity = 12.0f;
    float sunAngularRadiusDegrees = 0.6f;
    std::filesystem::path environmentMap{};
};

// SH L2 coefficients of the sky radiance with the Lambert band factors folded in, so evaluating
// the basis against a normal gives irradiance / pi (multiply by albedo for outgoing diffuse).
struct EnvironmentSh
{
    glm::vec4 bands[9]{};
};

// Matches Environment.cpp / Renderer/Shaders/sky.slang.
[[nodiscard]] glm::vec3 EvaluateSkyRadiance(
    const EnvironmentSettings& settings,
    const glm::vec3& direction);

// Numerically projects the sky gradient onto SH L2 (2048 directions, no sun disc).
[[nodiscard]] EnvironmentSh ComputeEnvironmentSh(const EnvironmentSettings& settings);

// Projects a linear equirectangular image (row 0 = top, `width * height * 3` floats) onto SH L2.
// The same direction convention as the shader is used:
//   phi = atan2(z, x) = (u - 0.5) * 2*pi, theta = acos(y) = v * pi.
[[nodiscard]] EnvironmentSh ComputeEnvironmentSh(
    const float* equirectRgb,
    std::uint32_t width,
    std::uint32_t height,
    float intensity);

// Split-sum BRDF integration LUT: RG32F, x = scale, y = bias for `f0 * x + y`.
void BuildEnvironmentBrdfLut(std::vector<float>& pixels, std::uint32_t size);

} // namespace mukki
