#include "Terrain/TerrainTypes.h"

namespace mukki
{

const char* TerrainBiomeName(TerrainBiome biome)
{
    switch (biome)
    {
    case TerrainBiome::Water:
        return "water";
    case TerrainBiome::Beach:
        return "beach";
    case TerrainBiome::Grassland:
        return "grassland";
    case TerrainBiome::Forest:
        return "forest";
    case TerrainBiome::Rock:
        return "rock";
    case TerrainBiome::Snow:
        return "snow";
    }
    return "grassland";
}

} // namespace mukki
