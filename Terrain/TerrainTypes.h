#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <glm/vec3.hpp>

namespace mukki
{

enum class TerrainBiome : std::uint8_t
{
    Water = 0,
    Beach,
    Grassland,
    Forest,
    Rock,
    Snow,
};

inline constexpr std::uint32_t kTerrainBiomeCount = 6;
inline constexpr std::uint32_t kMinTerrainResolution = 8;
inline constexpr std::uint32_t kMaxTerrainResolution = 1024;

[[nodiscard]] const char* TerrainBiomeName(TerrainBiome biome);

// Parameters for the procedural terrain generator. Height thresholds are fractions of
// `maxHeightMeters`; frequencies are in cycles across the whole terrain side.
struct TerrainSettings
{
    std::uint32_t seed = 1337u;
    // Heightfield cells per side. The vertex grid is (resolution + 1) x (resolution + 1).
    std::uint32_t resolution = 256u;
    // World-space side length in metres.
    float sizeMeters = 2048.0f;
    // Height of the highest peak in metres (roughly; erosion can shave the summit).
    float maxHeightMeters = 700.0f;

    // Continent shape: low-frequency FBM selects land, mountains and sea.
    float continentFrequency = 1.6f;
    float mountainFrequency = 5.5f;
    std::uint32_t octaves = 6u;
    float lacunarity = 2.0f;
    float persistence = 0.5f;
    // Ridged multifractal exponent; larger values sharpen the crests into narrow ridges.
    float ridgeSharpness = 1.6f;
    // Continent value above which mountains start replacing the rolling lowlands.
    float mountainCoverage = 0.43f;
    // Domain warp strength in noise units.
    float domainWarp = 0.35f;

    // Hydraulic degradation (droplet simulation, heights normalised to [0, 1]).
    bool hydraulicErosion = true;
    // Droplets per heightfield cell.
    float hydraulicDropletDensity = 0.4f;
    std::uint32_t hydraulicLifetime = 32u;
    std::uint32_t hydraulicBrushRadius = 3u;
    float hydraulicInertia = 0.05f;
    float hydraulicCapacity = 4.0f;
    float hydraulicMinCapacity = 0.01f;
    float hydraulicErodeSpeed = 0.3f;
    float hydraulicDepositSpeed = 0.3f;
    float hydraulicEvaporateSpeed = 0.02f;
    float hydraulicGravity = 4.0f;

    // Thermal degradation (talus angle relaxation, heights in metres).
    bool thermalErosion = true;
    std::uint32_t thermalIterations = 12u;
    float thermalTalusAngleDegrees = 32.0f;
    float thermalRate = 0.5f;

    // Biome classification thresholds.
    float waterLevel = 0.04f;
    float beachBand = 0.022f;
    float snowLine = 0.62f;
    float treeLine = 0.52f;
    float rockSlope = 0.42f;
    float forestMoisture = 0.56f;
};

struct TerrainGenerationStats
{
    double generationSeconds = 0.0;
    std::uint64_t hydraulicDroplets = 0;
    std::uint64_t thermalIterations = 0;
    float minHeightMeters = 0.0f;
    float maxHeightMeters = 0.0f;
    float meanHeightMeters = 0.0f;
    float waterLevelMeters = 0.0f;
    float maxSlopeDegrees = 0.0f;
    std::array<float, kTerrainBiomeCount> biomeFractions{};
};

struct TerrainHeightfield
{
    TerrainSettings settings{};
    std::uint32_t resolution = 0;
    float cellSizeMeters = 0.0f;
    float waterLevelMeters = 0.0f;
    // (resolution + 1)^2 heights in metres, row major: z * side + x.
    std::vector<float> heights{};
    std::vector<glm::vec3> normals{};
    // Per-vertex moisture in [0, 1], used by the biome classification.
    std::vector<float> moisture{};
    std::vector<TerrainBiome> biomes{};
    // RGBA8 biome albedo with one texel per grid vertex.
    std::vector<std::uint8_t> albedo{};
    TerrainGenerationStats stats{};

    [[nodiscard]] std::uint32_t SideVertices() const { return resolution + 1u; }
    [[nodiscard]] std::uint32_t VertexCount() const
    {
        return SideVertices() * SideVertices();
    }
    [[nodiscard]] std::size_t Index(std::uint32_t x, std::uint32_t z) const
    {
        return static_cast<std::size_t>(z) * SideVertices() + x;
    }
    [[nodiscard]] bool Empty() const { return heights.empty(); }
};

} // namespace mukki
