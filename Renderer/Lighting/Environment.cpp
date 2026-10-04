#include "Renderer/Lighting/Environment.h"

#include <algorithm>
#include <cmath>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>

namespace mukki
{
namespace
{

constexpr float kPi = 3.14159265358979323846f;

// Must match EvaluateSkySh in Renderer/Shaders/sky.slang.
float EvaluateShBasis(std::size_t band, const glm::vec3& direction)
{
    const float x = direction.x;
    const float y = direction.y;
    const float z = direction.z;

    switch (band)
    {
    case 0:
        return 0.282095f;
    case 1:
        return 0.488603f * x;
    case 2:
        return 0.488603f * y;
    case 3:
        return 0.488603f * z;
    case 4:
        return 1.092548f * x * y;
    case 5:
        return 1.092548f * y * z;
    case 6:
        return 0.315392f * (3.0f * z * z - 1.0f);
    case 7:
        return 1.092548f * x * z;
    default:
        return 0.546274f * (x * x - y * y);
    }
}

// Lambert convolution of the SH bands: (A_l / pi) with A_0 = pi, A_1 = 2pi/3, A_2 = pi/4.
float LambertBandFactor(std::size_t band)
{
    if (band == 0)
    {
        return 1.0f;
    }
    if (band <= 3)
    {
        return 2.0f / 3.0f;
    }
    return 0.25f;
}

float RadicalInverse(std::uint32_t index, std::uint32_t base)
{
    float result = 0.0f;
    float fraction = 1.0f;
    while (index > 0)
    {
        fraction /= static_cast<float>(base);
        result += fraction * static_cast<float>(index % base);
        index /= base;
    }
    return result;
}

// Standard split-sum importance sampling of the GGX normal distribution.
glm::vec3 ImportanceSampleGgx(const glm::vec2& xi, float roughness)
{
    const float a = roughness * roughness;
    const float phi = 2.0f * kPi * xi.x;
    const float cosTheta = std::sqrt((1.0f - xi.y) / (1.0f + (a * a - 1.0f) * xi.y));
    const float sinTheta = std::sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0f));
    return glm::vec3{std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta};
}

float GeometrySchlickGgxIbl(float nDotX, float roughness)
{
    const float r = roughness + 1.0f;
    const float k = (r * r) / 8.0f;
    return nDotX / std::max(nDotX * (1.0f - k) + k, 1e-4f);
}

} // namespace

glm::vec3 EvaluateSkyRadiance(const EnvironmentSettings& settings, const glm::vec3& direction)
{
    const float up = std::clamp(direction.y, 0.0f, 1.0f);
    const float down = std::clamp(-direction.y, 0.0f, 1.0f);
    const glm::vec3 upper = glm::mix(settings.horizonColor, settings.zenithColor, std::sqrt(up));
    const glm::vec3 lower = glm::mix(settings.horizonColor, settings.groundColor, std::sqrt(down));
    return (direction.y >= 0.0f ? upper : lower) * settings.intensity;
}

EnvironmentSh ComputeEnvironmentSh(const EnvironmentSettings& settings)
{
    constexpr std::uint32_t kZenithSteps = 32;
    constexpr std::uint32_t kAzimuthSteps = 64;

    EnvironmentSh result{};
    for (std::uint32_t band = 0; band < 9; ++band)
    {
        result.bands[band] = glm::vec4{0.0f};
    }

    for (std::uint32_t zenith = 0; zenith < kZenithSteps; ++zenith)
    {
        const float theta = (static_cast<float>(zenith) + 0.5f) / kZenithSteps * kPi;
        const float cosTheta = std::cos(theta);
        const float sinTheta = std::sin(theta);
        const float sampleWeight = sinTheta * (kPi / kZenithSteps) * (2.0f * kPi / kAzimuthSteps);

        for (std::uint32_t azimuth = 0; azimuth < kAzimuthSteps; ++azimuth)
        {
            const float phi = (static_cast<float>(azimuth) + 0.5f) / kAzimuthSteps * 2.0f * kPi;
            const glm::vec3 direction{
                sinTheta * std::cos(phi),
                cosTheta,
                sinTheta * std::sin(phi)};
            const glm::vec3 radiance = EvaluateSkyRadiance(settings, direction);
            if (radiance == glm::vec3{0.0f})
            {
                continue;
            }

            for (std::uint32_t band = 0; band < 9; ++band)
            {
                const float value = EvaluateShBasis(band, direction) * sampleWeight;
                result.bands[band] += glm::vec4{radiance * value, 0.0f};
            }
        }
    }

    for (std::uint32_t band = 0; band < 9; ++band)
    {
        result.bands[band] *= LambertBandFactor(band);
    }
    return result;
}

EnvironmentSh ComputeEnvironmentSh(
    const float* equirectRgb,
    std::uint32_t width,
    std::uint32_t height,
    float intensity)
{
    EnvironmentSh result{};
    for (std::uint32_t band = 0; band < 9; ++band)
    {
        result.bands[band] = glm::vec4{0.0f};
    }

    if (equirectRgb == nullptr || width == 0 || height == 0)
    {
        return result;
    }

    const float dTheta = kPi / static_cast<float>(height);
    const float dPhi = 2.0f * kPi / static_cast<float>(width);
    for (std::uint32_t y = 0; y < height; ++y)
    {
        // Row 0 is the top of the image (theta = 0, +Y).
        const float theta = (static_cast<float>(y) + 0.5f) * dTheta;
        const float cosTheta = std::cos(theta);
        const float sinTheta = std::sin(theta);
        const float sampleWeight = sinTheta * dTheta * dPhi;

        for (std::uint32_t x = 0; x < width; ++x)
        {
            const float phi = ((static_cast<float>(x) + 0.5f) / static_cast<float>(width) - 0.5f) * 2.0f * kPi;
            const glm::vec3 direction{
                sinTheta * std::cos(phi),
                cosTheta,
                sinTheta * std::sin(phi)};
            const std::size_t offset = (static_cast<std::size_t>(y) * width + x) * 3;
            const glm::vec3 radiance{
                equirectRgb[offset + 0],
                equirectRgb[offset + 1],
                equirectRgb[offset + 2]};
            if (radiance == glm::vec3{0.0f})
            {
                continue;
            }

            for (std::uint32_t band = 0; band < 9; ++band)
            {
                const float value = EvaluateShBasis(band, direction) * sampleWeight;
                result.bands[band] += glm::vec4{radiance * (value * intensity), 0.0f};
            }
        }
    }

    for (std::uint32_t band = 0; band < 9; ++band)
    {
        result.bands[band] *= LambertBandFactor(band);
    }
    return result;
}

void BuildEnvironmentBrdfLut(std::vector<float>& pixels, std::uint32_t size)
{
    constexpr std::uint32_t kSampleCount = 128;

    pixels.assign(static_cast<std::size_t>(size) * size * 2, 0.0f);
    for (std::uint32_t y = 0; y < size; ++y)
    {
        for (std::uint32_t x = 0; x < size; ++x)
        {
            const float nDotV = (static_cast<float>(x) + 0.5f) / static_cast<float>(size);
            const float roughness = (static_cast<float>(y) + 0.5f) / static_cast<float>(size);

            const glm::vec3 view{
                std::sqrt(std::max(1.0f - nDotV * nDotV, 0.0f)),
                0.0f,
                nDotV};

            float scale = 0.0f;
            float bias = 0.0f;
            for (std::uint32_t sampleIndex = 0; sampleIndex < kSampleCount; ++sampleIndex)
            {
                const glm::vec2 xi{
                    RadicalInverse(sampleIndex + 1, 2),
                    RadicalInverse(sampleIndex + 1, 3)};
                const glm::vec3 halfVector = ImportanceSampleGgx(xi, roughness);
                const glm::vec3 light = 2.0f * glm::dot(view, halfVector) * halfVector - view;
                if (light.z <= 0.0f)
                {
                    continue;
                }

                const float nDotL = light.z;
                const float nDotH = std::max(halfVector.z, 0.0f);
                const float vDotH = std::max(glm::dot(view, halfVector), 0.0f);
                const float geometry =
                    GeometrySchlickGgxIbl(nDotV, roughness) * GeometrySchlickGgxIbl(nDotL, roughness);
                const float visibility =
                    geometry * vDotH / std::max(nDotH * nDotV, 1e-5f);
                const float fresnel = std::pow(1.0f - vDotH, 5.0f);
                scale += (1.0f - fresnel) * visibility;
                bias += fresnel * visibility;
            }

            const float inverseSamples = 1.0f / static_cast<float>(kSampleCount);
            const std::size_t offset = (static_cast<std::size_t>(y) * size + x) * 2;
            pixels[offset + 0] = scale * inverseSamples;
            pixels[offset + 1] = bias * inverseSamples;
        }
    }
}

} // namespace mukki
