#include "Terrain/TerrainNoise.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace mukki
{
namespace
{

std::uint64_t SplitMix64(std::uint64_t& state)
{
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t result = state;
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9ull;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EBull;
    return result ^ (result >> 31);
}

float Quintic(float value)
{
    return value * value * value * (value * (value * 6.0f - 15.0f) + 10.0f);
}

float Lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

// Dot product with one of the eight 2D lattice gradients.
float DotGradient(std::uint32_t hash, float x, float y)
{
    switch (hash & 7u)
    {
    case 0:
        return x;
    case 1:
        return -x;
    case 2:
        return y;
    case 3:
        return -y;
    case 4:
        return x + y;
    case 5:
        return x - y;
    case 6:
        return -x + y;
    default:
        return -x - y;
    }
}

} // namespace

PerlinNoise2D::PerlinNoise2D(std::uint32_t seed)
{
    std::uint8_t shuffle[256]{};
    for (std::uint32_t index = 0; index < 256u; ++index)
    {
        shuffle[index] = static_cast<std::uint8_t>(index);
    }

    std::uint64_t state = static_cast<std::uint64_t>(seed) * 0x9E3779B97F4A7C15ull + 0xD1B54A32D192ED03ull;
    for (std::int32_t index = 255; index > 0; --index)
    {
        const std::uint32_t other =
            static_cast<std::uint32_t>(SplitMix64(state) % static_cast<std::uint64_t>(index + 1));
        std::swap(
            shuffle[static_cast<std::size_t>(index)],
            shuffle[static_cast<std::size_t>(other)]);
    }

    for (std::uint32_t index = 0; index < 256u; ++index)
    {
        m_permutation[index] = shuffle[index];
        m_permutation[index + 256u] = shuffle[index];
    }
}

std::uint32_t PerlinNoise2D::Hash(std::int32_t x, std::int32_t y) const
{
    const std::uint32_t wrappedX = static_cast<std::uint32_t>(x) & 255u;
    const std::uint32_t wrappedY = static_cast<std::uint32_t>(y) & 255u;
    const std::size_t index = static_cast<std::size_t>(wrappedX) + m_permutation[wrappedY];
    return m_permutation[index];
}

float PerlinNoise2D::Sample(float x, float y) const
{
    const float latticeX = std::floor(x);
    const float latticeY = std::floor(y);
    const std::int32_t baseX = static_cast<std::int32_t>(latticeX);
    const std::int32_t baseY = static_cast<std::int32_t>(latticeY);

    const float fractionX = x - latticeX;
    const float fractionY = y - latticeY;
    const float smoothX = Quintic(fractionX);
    const float smoothY = Quintic(fractionY);

    const float corner00 = DotGradient(Hash(baseX, baseY), fractionX, fractionY);
    const float corner10 = DotGradient(Hash(baseX + 1, baseY), fractionX - 1.0f, fractionY);
    const float corner01 = DotGradient(Hash(baseX, baseY + 1), fractionX, fractionY - 1.0f);
    const float corner11 =
        DotGradient(Hash(baseX + 1, baseY + 1), fractionX - 1.0f, fractionY - 1.0f);

    const float bottom = Lerp(corner00, corner10, smoothX);
    const float top = Lerp(corner01, corner11, smoothX);
    // Scale so the diagonal gradients keep the result inside [-1, 1].
    return std::clamp(Lerp(bottom, top, smoothY) * 0.70710678f, -1.0f, 1.0f);
}

float PerlinNoise2D::Fbm(
    float x,
    float y,
    std::uint32_t octaves,
    float lacunarity,
    float persistence) const
{
    if (octaves == 0u)
    {
        return 0.5f;
    }

    const float safeLacunarity = std::max(lacunarity, 1.0f);
    const float safePersistence = std::clamp(persistence, 0.0f, 1.0f);

    float result = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float normalization = 0.0f;
    for (std::uint32_t octave = 0; octave < octaves; ++octave)
    {
        result += Sample(x * frequency, y * frequency) * amplitude;
        normalization += amplitude;
        amplitude *= safePersistence;
        frequency *= safeLacunarity;
    }

    if (normalization <= 0.0f)
    {
        return 0.5f;
    }
    return std::clamp(result / normalization * 0.5f + 0.5f, 0.0f, 1.0f);
}

float PerlinNoise2D::RidgedFbm(
    float x,
    float y,
    std::uint32_t octaves,
    float lacunarity,
    float persistence,
    float sharpness) const
{
    if (octaves == 0u)
    {
        return 0.0f;
    }

    const float safeLacunarity = std::max(lacunarity, 1.0f);
    const float safePersistence = std::clamp(persistence, 0.0f, 1.0f);
    const float safeSharpness = std::max(sharpness, 0.01f);

    float result = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    float normalization = 0.0f;
    float weight = 1.0f;
    for (std::uint32_t octave = 0; octave < octaves; ++octave)
    {
        const float signal =
            std::pow(1.0f - std::abs(Sample(x * frequency, y * frequency)), safeSharpness) * weight;
        result += signal * amplitude;
        normalization += amplitude;
        weight = std::clamp(signal * 2.0f, 0.0f, 1.0f);
        amplitude *= safePersistence;
        frequency *= safeLacunarity;
    }

    if (normalization <= 0.0f)
    {
        return 0.0f;
    }
    return std::clamp(result / normalization, 0.0f, 1.0f);
}

float HashToUnit2D(std::int32_t x, std::int32_t y, std::uint32_t seed)
{
    std::uint32_t hash = seed;
    hash ^= static_cast<std::uint32_t>(x) * 0x8DA6B343u;
    hash ^= static_cast<std::uint32_t>(y) * 0xD8163841u;
    hash ^= hash >> 15;
    hash *= 0x2C1B3C6Du;
    hash ^= hash >> 12;
    hash *= 0x297A2D39u;
    hash ^= hash >> 15;
    return static_cast<float>(hash & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
}

} // namespace mukki
