#pragma once

#include <cstdint>
#include <vector>

#include <glm/vec3.hpp>

namespace mukki
{

// Tileable procedural noise used by the volumetric cloud volume. The construction follows the
// Nubis/Decima papers: a Perlin-Worley base shape, Worley FBM detail octaves and a curl field for
// wind distortion. Coordinates are in lattice space; `period` is the lattice period so a texture
// sampled over [0, period) tiles seamlessly.
[[nodiscard]] float TileablePerlin(
    const glm::vec3& position,
    std::uint32_t period,
    std::uint32_t seed);
[[nodiscard]] float TileableWorley(
    const glm::vec3& position,
    std::uint32_t period,
    std::uint32_t seed);
[[nodiscard]] float TileableWorleyFbm(
    const glm::vec3& position,
    std::uint32_t period,
    std::uint32_t seed,
    std::uint32_t octaves);

// RGBA8 base shape: r = Perlin-Worley, g/b/a = Worley FBM at increasing frequencies.
// Generates the z-slice range [zBegin, zEnd) of `resolution^3` RGBA pixels.
void GenerateCloudBaseShape(
    std::uint32_t resolution,
    std::uint32_t seed,
    std::uint32_t zBegin,
    std::uint32_t zEnd,
    std::vector<std::uint8_t>& outPixels);

// RGBA8 high-frequency detail: r/g/b = Worley FBM octaves at increasing frequencies.
void GenerateCloudDetail(
    std::uint32_t resolution,
    std::uint32_t seed,
    std::uint32_t zBegin,
    std::uint32_t zEnd,
    std::vector<std::uint8_t>& outPixels);

// RGBA8 curl field sampled at z = 0.5 (remapped to [0,1]); rgb = curl vector.
void GenerateCloudCurl(
    std::uint32_t resolution,
    std::uint32_t seed,
    std::uint32_t yBegin,
    std::uint32_t yEnd,
    std::vector<std::uint8_t>& outPixels);

// RGBA8 weather map with regional variation around 0.5: r coverage, g cloud type, b altitude.
// The UI sliders are applied on top of this in the shader, so the texture never needs regenerating.
void GenerateCloudWeather(
    std::uint32_t resolution,
    float variation,
    std::uint32_t seed,
    std::uint32_t yBegin,
    std::uint32_t yEnd,
    std::vector<std::uint8_t>& outPixels);

} // namespace mukki
