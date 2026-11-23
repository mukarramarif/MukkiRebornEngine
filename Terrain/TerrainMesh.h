#pragma once

#include "Assets/GltfLoader.h"
#include "Terrain/TerrainTypes.h"

namespace mukki
{

// Builds a world-space (metres, Y up, centred on the origin) glTF model from a generated
// heightfield. The primitive is indexed and carries the biome albedo as an sRGB base colour
// texture, so it can be uploaded through the regular model pipeline.
[[nodiscard]] GltfModel BuildTerrainModel(const TerrainHeightfield& heightfield);

} // namespace mukki
