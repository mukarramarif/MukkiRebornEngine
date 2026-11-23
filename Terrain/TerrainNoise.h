#pragma once

#include <cstdint>

namespace mukki
{

// Deterministic seeded 2D Perlin noise. The permutation table is shuffled from the seed with a
// fixed algorithm, so the same seed produces the same terrain on every platform.
class PerlinNoise2D
{
public:
    explicit PerlinNoise2D(std::uint32_t seed);

    // Classic gradient noise, approximately in [-1, 1].
    [[nodiscard]] float Sample(float x, float y) const;

    // Fractal Brownian motion, normalised to [0, 1].
    [[nodiscard]] float Fbm(
        float x,
        float y,
        std::uint32_t octaves,
        float lacunarity,
        float persistence) const;

    // Ridged multifractal in [0, 1]: |noise| is inverted so the field peaks along the noise
    // zero crossings, and each octave is weighted by the previous one to keep the crests sharp.
    [[nodiscard]] float RidgedFbm(
        float x,
        float y,
        std::uint32_t octaves,
        float lacunarity,
        float persistence,
        float sharpness) const;

private:
    [[nodiscard]] std::uint32_t Hash(std::int32_t x, std::int32_t y) const;

    std::uint8_t m_permutation[512]{};
};

// Fast seedable hash mapped to [0, 1). Used for per-vertex colour variation and droplet spawns.
[[nodiscard]] float HashToUnit2D(std::int32_t x, std::int32_t y, std::uint32_t seed);

} // namespace mukki
