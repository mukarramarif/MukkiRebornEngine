#include "Renderer/Lighting/ClusterGrid.h"

#include "Utilities/Assert.h"

namespace mukki
{

ClusterGrid ComputeClusterGrid(std::uint32_t width, std::uint32_t height)
{
    MRE_ASSERT(width > 0 && height > 0, "Cluster grid dimensions must be greater than zero");

    ClusterGrid grid{};
    grid.x = (width + kClusterTileSize - 1) / kClusterTileSize;
    grid.y = (height + kClusterTileSize - 1) / kClusterTileSize;
    grid.z = kClusterZSliceCount;
    grid.count = grid.x * grid.y * grid.z;
    return grid;
}

} // namespace mukki
