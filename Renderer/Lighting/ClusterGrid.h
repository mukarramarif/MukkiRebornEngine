#pragma once

#include <cstdint>

namespace mukki
{

// Cluster grid used by the light culling pass. The grid is rebuilt when the swapchain is resized.
inline constexpr std::uint32_t kClusterTileSize = 32;
inline constexpr std::uint32_t kClusterZSliceCount = 24;
inline constexpr std::uint32_t kMaxLightsPerCluster = 64;
inline constexpr std::uint32_t kMaxLights = 256;

struct ClusterGrid
{
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t z = 0;
    std::uint32_t count = 0;
};

[[nodiscard]] ClusterGrid ComputeClusterGrid(std::uint32_t width, std::uint32_t height);

} // namespace mukki
