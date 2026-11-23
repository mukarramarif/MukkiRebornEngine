#pragma once

#include <cstdint>
#include <vector>

#include <glm/vec3.hpp>

namespace mukki
{

struct TerrainSettings;

// Recomputes unit vertex normals for a heightfield in metres. Heights are row major
// (z * (resolution + 1) + x) and the grid spans [0, resolution] cells.
void ComputeTerrainNormals(
    const std::vector<float>& heights,
    std::uint32_t resolution,
    float cellSizeMeters,
    std::vector<glm::vec3>& outNormals);

// Slope in [0, 1], defined as 1 - normal.y (0 is flat, 1 is vertical).
[[nodiscard]] float TerrainSlope(const glm::vec3& normal);

// Droplet-based hydraulic erosion (Beyer's particle model). Heights are expected to be normalised
// to roughly [0, 1]; the random stream is seeded from settings.seed so the result is deterministic.
// Returns the number of simulated droplets.
std::uint64_t ApplyHydraulicErosion(
    std::vector<float>& heights,
    std::uint32_t resolution,
    const TerrainSettings& settings);

// Talus-angle thermal erosion: material on slopes steeper than the repose angle slides to the
// lowest neighbour. Heights are in metres. Returns the number of full iterations applied.
std::uint64_t ApplyThermalErosion(
    std::vector<float>& heights,
    std::uint32_t resolution,
    float cellSizeMeters,
    float talusAngleDegrees,
    std::uint32_t iterations,
    float rate);

} // namespace mukki
