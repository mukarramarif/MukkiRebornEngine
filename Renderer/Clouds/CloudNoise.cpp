#include "Renderer/Clouds/CloudNoise.h"

#include <algorithm>
#include <cmath>

#include <glm/common.hpp>
#include <glm/geometric.hpp>

namespace mukki
{
namespace
{

std::uint32_t Wrap(std::int32_t value, std::uint32_t period)
{
    const std::int32_t signedPeriod = static_cast<std::int32_t>(period);
    std::int32_t result = value % signedPeriod;
    if (result < 0)
    {
        result += signedPeriod;
    }
    return static_cast<std::uint32_t>(result);
}

std::uint32_t Hash3(std::int32_t x, std::int32_t y, std::int32_t z, std::uint32_t seed)
{
    std::uint32_t hash = seed;
    hash ^= static_cast<std::uint32_t>(x) * 0x8DA6B343u;
    hash ^= static_cast<std::uint32_t>(y) * 0xD8163841u;
    hash ^= static_cast<std::uint32_t>(z) * 0xCB1AB31Fu;
    hash ^= hash >> 15;
    hash *= 0x2C1B3C6Du;
    hash ^= hash >> 12;
    hash *= 0x297A2D39u;
    hash ^= hash >> 15;
    return hash;
}

constexpr float HashToUnit(std::uint32_t hash)
{
    return static_cast<float>(hash & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
}

float Quintic(float value)
{
    return value * value * value * (value * (value * 6.0f - 15.0f) + 10.0f);
}

float Lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

glm::vec3 GradientFromHash(std::uint32_t hash)
{
    // 12 edge gradients of a cube (classic Perlin set).
    switch (hash % 12u)
    {
    case 0:
        return glm::vec3{1.0f, 1.0f, 0.0f};
    case 1:
        return glm::vec3{-1.0f, 1.0f, 0.0f};
    case 2:
        return glm::vec3{1.0f, -1.0f, 0.0f};
    case 3:
        return glm::vec3{-1.0f, -1.0f, 0.0f};
    case 4:
        return glm::vec3{1.0f, 0.0f, 1.0f};
    case 5:
        return glm::vec3{-1.0f, 0.0f, 1.0f};
    case 6:
        return glm::vec3{1.0f, 0.0f, -1.0f};
    case 7:
        return glm::vec3{-1.0f, 0.0f, -1.0f};
    case 8:
        return glm::vec3{0.0f, 1.0f, 1.0f};
    case 9:
        return glm::vec3{0.0f, -1.0f, 1.0f};
    case 10:
        return glm::vec3{0.0f, 1.0f, -1.0f};
    default:
        return glm::vec3{0.0f, -1.0f, -1.0f};
    }
}

std::uint8_t ToByte(float value)
{
    return static_cast<std::uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

} // namespace

float TileablePerlin(const glm::vec3& position, std::uint32_t period, std::uint32_t seed)
{
    if (period == 0)
    {
        return 0.5f;
    }

    const glm::vec3 lattice = glm::floor(position);
    const glm::vec3 fraction = position - lattice;
    const glm::vec3 smooth{
        Quintic(fraction.x),
        Quintic(fraction.y),
        Quintic(fraction.z)};

    const std::int32_t baseX = static_cast<std::int32_t>(lattice.x);
    const std::int32_t baseY = static_cast<std::int32_t>(lattice.y);
    const std::int32_t baseZ = static_cast<std::int32_t>(lattice.z);

    float result = 0.0f;
    for (std::int32_t corner = 0; corner < 8; ++corner)
    {
        const std::int32_t offsetX = corner & 1;
        const std::int32_t offsetY = (corner >> 1) & 1;
        const std::int32_t offsetZ = (corner >> 2) & 1;

        const glm::vec3 gradient = GradientFromHash(Hash3(
            static_cast<std::int32_t>(Wrap(baseX + offsetX, period)),
            static_cast<std::int32_t>(Wrap(baseY + offsetY, period)),
            static_cast<std::int32_t>(Wrap(baseZ + offsetZ, period)),
            seed));
        const glm::vec3 delta = fraction - glm::vec3{
                                                static_cast<float>(offsetX),
                                                static_cast<float>(offsetY),
                                                static_cast<float>(offsetZ)};

        const float weightX = offsetX == 0 ? (1.0f - smooth.x) : smooth.x;
        const float weightY = offsetY == 0 ? (1.0f - smooth.y) : smooth.y;
        const float weightZ = offsetZ == 0 ? (1.0f - smooth.z) : smooth.z;
        result += glm::dot(gradient, delta) * weightX * weightY * weightZ;
    }

    // Classic Perlin range is roughly [-1, 1] (0.707 worst case per axis pair); remap with a
    // slightly generous scale so the result stays inside [0, 1].
    return std::clamp(result * 0.9f + 0.5f, 0.0f, 1.0f);
}

float TileableWorley(const glm::vec3& position, std::uint32_t period, std::uint32_t seed)
{
    if (period == 0)
    {
        return 1.0f;
    }

    const glm::vec3 cell = glm::floor(position);
    const glm::vec3 fraction = position - cell;
    const std::int32_t baseX = static_cast<std::int32_t>(cell.x);
    const std::int32_t baseY = static_cast<std::int32_t>(cell.y);
    const std::int32_t baseZ = static_cast<std::int32_t>(cell.z);

    float minDistance = 1.0f;
    for (std::int32_t offsetZ = -1; offsetZ <= 1; ++offsetZ)
    {
        for (std::int32_t offsetY = -1; offsetY <= 1; ++offsetY)
        {
            for (std::int32_t offsetX = -1; offsetX <= 1; ++offsetX)
            {
                // One hash per cell, cheaply expanded to three feature components.
                const std::uint32_t hash = Hash3(
                    static_cast<std::int32_t>(Wrap(baseX + offsetX, period)),
                    static_cast<std::int32_t>(Wrap(baseY + offsetY, period)),
                    static_cast<std::int32_t>(Wrap(baseZ + offsetZ, period)),
                    seed);
                const std::uint32_t hashX = hash * 0x9E3779B9u + 0x85EBCA6Bu;
                const std::uint32_t hashY = (hash ^ (hash >> 13)) * 0xC2B2AE35u;

                const glm::vec3 feature{
                    static_cast<float>(offsetX) + HashToUnit(hash),
                    static_cast<float>(offsetY) + HashToUnit(hashX),
                    static_cast<float>(offsetZ) + HashToUnit(hashY)};
                const glm::vec3 delta = feature - fraction;
                minDistance = std::min(minDistance, glm::dot(delta, delta));
            }
        }
    }

    return std::clamp(std::sqrt(minDistance), 0.0f, 1.0f);
}

float TileableWorleyFbm(
    const glm::vec3& position,
    std::uint32_t period,
    std::uint32_t seed,
    std::uint32_t octaves)
{
    float result = 0.0f;
    float amplitude = 0.625f;
    float normalization = 0.0f;
    std::uint32_t frequency = 1;
    for (std::uint32_t octave = 0; octave < octaves; ++octave)
    {
        result += amplitude *
                  TileableWorley(position * static_cast<float>(frequency), period * frequency, seed + octave);
        normalization += amplitude;
        amplitude *= 0.5f;
        frequency *= 2;
    }
    return normalization > 0.0f ? result / normalization : 0.0f;
}

namespace
{

float PerlinWorley(const glm::vec3& position, std::uint32_t period, std::uint32_t seed)
{
    const float perlin = TileablePerlin(position, period, seed);
    const float worley = TileableWorleyFbm(position, period, seed, 3);
    const float minimum = 1.0f - worley;
    if (worley <= 1e-4f)
    {
        return perlin;
    }
    return std::clamp((perlin - minimum) / std::max(worley, 1e-4f), 0.0f, 1.0f);
}

// Curl of a 3D vector potential built from three tileable Perlin fields.
glm::vec3 CurlNoise(const glm::vec3& position, std::uint32_t period, std::uint32_t seed)
{
    const float epsilon = 1.0f / static_cast<float>(period);
    const glm::vec3 dx{epsilon, 0.0f, 0.0f};
    const glm::vec3 dy{0.0f, epsilon, 0.0f};
    const glm::vec3 dz{0.0f, 0.0f, epsilon};

    const auto potential = [&](const glm::vec3& sample, std::uint32_t offsetSeed) {
        return glm::vec3{
            TileablePerlin(sample, period, seed + offsetSeed + 0u),
            TileablePerlin(sample, period, seed + offsetSeed + 11u),
            TileablePerlin(sample, period, seed + offsetSeed + 23u)};
    };

    const glm::vec3 px = potential(position + dx, 0u);
    const glm::vec3 mx = potential(position - dx, 0u);
    const glm::vec3 py = potential(position + dy, 0u);
    const glm::vec3 my = potential(position - dy, 0u);
    const glm::vec3 pz = potential(position + dz, 0u);
    const glm::vec3 mz = potential(position - dz, 0u);

    const glm::vec3 derivativeX = (px - mx) / (2.0f * epsilon);
    const glm::vec3 derivativeY = (py - my) / (2.0f * epsilon);
    const glm::vec3 derivativeZ = (pz - mz) / (2.0f * epsilon);

    return glm::vec3{
        derivativeY.z - derivativeZ.y,
        derivativeZ.x - derivativeX.z,
        derivativeX.y - derivativeY.x};
}

} // namespace

void GenerateCloudBaseShape(
    std::uint32_t resolution,
    std::uint32_t seed,
    std::uint32_t zBegin,
    std::uint32_t zEnd,
    std::vector<std::uint8_t>& outPixels)
{
    outPixels.resize(static_cast<std::size_t>(resolution) * resolution * resolution * 4);
    zEnd = std::min(zEnd, resolution);

    for (std::uint32_t z = zBegin; z < zEnd; ++z)
    {
        for (std::uint32_t y = 0; y < resolution; ++y)
        {
            for (std::uint32_t x = 0; x < resolution; ++x)
            {
                const glm::vec3 unit{
                    (static_cast<float>(x) + 0.5f) / static_cast<float>(resolution),
                    (static_cast<float>(y) + 0.5f) / static_cast<float>(resolution),
                    (static_cast<float>(z) + 0.5f) / static_cast<float>(resolution)};

                // One Worley octave per channel; the shader weights g/b/a as an FBM.
                const float base = PerlinWorley(unit * 4.0f, 4, seed);
                const float worleyMedium = TileableWorley(unit * 8.0f, 8, seed + 101u);
                const float worleyHigh = TileableWorley(unit * 16.0f, 16, seed + 211u);
                const float worleyLow = TileableWorley(unit * 32.0f, 32, seed + 307u);

                const std::size_t index =
                    ((static_cast<std::size_t>(z) * resolution + y) * resolution + x) * 4;
                outPixels[index + 0] = ToByte(base);
                outPixels[index + 1] = ToByte(worleyMedium);
                outPixels[index + 2] = ToByte(worleyHigh);
                outPixels[index + 3] = ToByte(worleyLow);
            }
        }
    }
}

void GenerateCloudDetail(
    std::uint32_t resolution,
    std::uint32_t seed,
    std::uint32_t zBegin,
    std::uint32_t zEnd,
    std::vector<std::uint8_t>& outPixels)
{
    outPixels.resize(static_cast<std::size_t>(resolution) * resolution * resolution * 4);
    zEnd = std::min(zEnd, resolution);

    for (std::uint32_t z = zBegin; z < zEnd; ++z)
    {
        for (std::uint32_t y = 0; y < resolution; ++y)
        {
            for (std::uint32_t x = 0; x < resolution; ++x)
            {
                const glm::vec3 unit{
                    (static_cast<float>(x) + 0.5f) / static_cast<float>(resolution),
                    (static_cast<float>(y) + 0.5f) / static_cast<float>(resolution),
                    (static_cast<float>(z) + 0.5f) / static_cast<float>(resolution)};

                const float detailLow = TileableWorley(unit * 4.0f, 4, seed + 401u);
                const float detailMedium = TileableWorley(unit * 8.0f, 8, seed + 509u);
                const float detailHigh = TileableWorley(unit * 16.0f, 16, seed + 601u);

                const std::size_t index =
                    ((static_cast<std::size_t>(z) * resolution + y) * resolution + x) * 4;
                outPixels[index + 0] = ToByte(detailLow);
                outPixels[index + 1] = ToByte(detailMedium);
                outPixels[index + 2] = ToByte(detailHigh);
                outPixels[index + 3] = 255u;
            }
        }
    }
}

void GenerateCloudCurl(
    std::uint32_t resolution,
    std::uint32_t seed,
    std::uint32_t yBegin,
    std::uint32_t yEnd,
    std::vector<std::uint8_t>& outPixels)
{
    outPixels.resize(static_cast<std::size_t>(resolution) * resolution * 4);
    yEnd = std::min(yEnd, resolution);

    constexpr std::uint32_t kPeriod = 8;
    for (std::uint32_t y = yBegin; y < yEnd; ++y)
    {
        for (std::uint32_t x = 0; x < resolution; ++x)
        {
            const glm::vec3 position{
                ((static_cast<float>(x) + 0.5f) / static_cast<float>(resolution)) *
                    static_cast<float>(kPeriod),
                ((static_cast<float>(y) + 0.5f) / static_cast<float>(resolution)) *
                    static_cast<float>(kPeriod),
                0.5f * static_cast<float>(kPeriod)};

            const glm::vec3 curl = CurlNoise(position, kPeriod, seed);
            const std::size_t index = (static_cast<std::size_t>(y) * resolution + x) * 4;
            outPixels[index + 0] = ToByte(curl.x * 0.5f + 0.5f);
            outPixels[index + 1] = ToByte(curl.y * 0.5f + 0.5f);
            outPixels[index + 2] = ToByte(curl.z * 0.5f + 0.5f);
            outPixels[index + 3] = 255u;
        }
    }
}

void GenerateCloudWeather(
    std::uint32_t resolution,
    float variation,
    std::uint32_t seed,
    std::uint32_t yBegin,
    std::uint32_t yEnd,
    std::vector<std::uint8_t>& outPixels)
{
    outPixels.resize(static_cast<std::size_t>(resolution) * resolution * 4);
    yEnd = std::min(yEnd, resolution);

    const float amplitude = std::max(variation, 0.0f);
    for (std::uint32_t y = yBegin; y < yEnd; ++y)
    {
        for (std::uint32_t x = 0; x < resolution; ++x)
        {
            const glm::vec3 unit{
                (static_cast<float>(x) + 0.5f) / static_cast<float>(resolution),
                (static_cast<float>(y) + 0.5f) / static_cast<float>(resolution),
                0.5f};

            // Regional variation centred at 0.5; the sliders offset coverage/type/altitude in the
            // shader so no texture regeneration is needed when they move.
            const float coverageNoise =
                (TileablePerlin(unit * 4.0f, 4, seed) * 0.65f +
                 TileablePerlin(unit * 8.0f, 8, seed + 71u) * 0.35f) -
                0.5f;
            const float typeNoise = TileablePerlin(unit * 2.0f, 2, seed + 131u) - 0.5f;
            const float altitudeNoise = TileablePerlin(unit * 3.0f, 3, seed + 197u) - 0.5f;

            const std::size_t index = (static_cast<std::size_t>(y) * resolution + x) * 4;
            outPixels[index + 0] = ToByte(0.5f + coverageNoise * amplitude * 2.0f);
            outPixels[index + 1] = ToByte(0.5f + typeNoise * amplitude * 2.0f);
            outPixels[index + 2] = ToByte(0.5f + altitudeNoise * amplitude * 2.0f);
            outPixels[index + 3] = 255u;
        }
    }
}

} // namespace mukki
