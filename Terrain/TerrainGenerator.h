#pragma once

#include <cstdint>
#include <vector>

#include "Terrain/TerrainTypes.h"

namespace mukki
{

// Generates a deterministic landscape heightfield: domain-warped ridged multifractal mountains
// blended with FBM greenland, hydraulic droplet erosion, thermal talus relaxation, flattened
// water bodies and biome classification. The same settings always produce the same terrain.
[[nodiscard]] TerrainHeightfield GenerateTerrain(const TerrainSettings& settings);

// Classifies one vertex. `heightMeters` is measured from zero, `slope` is 1 - normal.y and
// `moisture` is in [0, 1]. Thresholds are stored as fractions of `settings.maxHeightMeters`.
[[nodiscard]] TerrainBiome ClassifyTerrainBiome(
    float heightMeters,
    float slope,
    float moisture,
    const TerrainSettings& settings);

// RGBA8 hill-shaded preview of the biome colours, one texel per grid vertex. This is only meant
// for the editor UI; the terrain model uses the unshaded `TerrainHeightfield::albedo` instead.
[[nodiscard]] std::vector<std::uint8_t> BuildTerrainPreview(const TerrainHeightfield& heightfield);

} // namespace mukki
