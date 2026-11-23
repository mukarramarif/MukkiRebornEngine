#include "Terrain/TerrainErosion.h"
#include "Terrain/TerrainGenerator.h"
#include "Terrain/TerrainMesh.h"
#include "Terrain/TerrainNoise.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

namespace mukki
{
namespace
{

TerrainSettings MakeTestSettings(std::uint32_t resolution = 48u)
{
    TerrainSettings settings{};
    settings.seed = 4242u;
    settings.resolution = resolution;
    settings.sizeMeters = 600.0f;
    settings.maxHeightMeters = 280.0f;
    settings.hydraulicDropletDensity = 0.15f;
    settings.thermalIterations = 3u;
    settings.waterLevel = 0.12f;
    settings.snowLine = 0.60f;
    return settings;
}

std::size_t GridIndex(std::uint32_t resolution, std::uint32_t x, std::uint32_t z)
{
    return static_cast<std::size_t>(z) * (resolution + 1u) + x;
}

float MaximumSlopeDegrees(
    const std::vector<float>& heights,
    std::uint32_t resolution,
    float cellSizeMeters)
{
    std::vector<glm::vec3> normals{};
    ComputeTerrainNormals(heights, resolution, cellSizeMeters, normals);
    float maximum = 0.0f;
    for (const glm::vec3& normal : normals)
    {
        maximum = std::max(maximum, std::acos(std::clamp(normal.y, -1.0f, 1.0f)) * 57.2957795f);
    }
    return maximum;
}

TEST(TerrainNoise, IsDeterministicAndSeedDependent)
{
    const PerlinNoise2D first{91u};
    const PerlinNoise2D second{91u};
    const PerlinNoise2D other{92u};

    bool differs = false;
    for (float y = 0.0f; y < 4.0f; y += 0.37f)
    {
        for (float x = 0.0f; x < 4.0f; x += 0.41f)
        {
            const float value = first.Sample(x, y);
            EXPECT_EQ(value, second.Sample(x, y));
            differs = differs || value != other.Sample(x, y);
            EXPECT_GE(value, -1.0f);
            EXPECT_LE(value, 1.0f);
        }
    }
    EXPECT_TRUE(differs);
}

TEST(TerrainNoise, FractalLayersStayNormalised)
{
    const PerlinNoise2D noise{17u};
    for (float y = 0.0f; y < 3.0f; y += 0.29f)
    {
        for (float x = 0.0f; x < 3.0f; x += 0.31f)
        {
            const float fbm = noise.Fbm(x, y, 6u, 2.0f, 0.5f);
            const float ridged = noise.RidgedFbm(x, y, 6u, 2.0f, 0.5f, 1.6f);
            EXPECT_GE(fbm, 0.0f);
            EXPECT_LE(fbm, 1.0f);
            EXPECT_GE(ridged, 0.0f);
            EXPECT_LE(ridged, 1.0f);
        }
    }
}

TEST(TerrainGenerator, IsDeterministicForTheSameSettings)
{
    const TerrainSettings settings = MakeTestSettings();
    const TerrainHeightfield first = GenerateTerrain(settings);
    const TerrainHeightfield second = GenerateTerrain(settings);

    ASSERT_FALSE(first.Empty());
    ASSERT_EQ(first.heights.size(), second.heights.size());
    EXPECT_EQ(first.resolution, settings.resolution);
    EXPECT_EQ(first.normals.size(), first.heights.size());
    EXPECT_EQ(first.biomes.size(), first.heights.size());
    EXPECT_EQ(first.albedo.size(), first.heights.size() * 4u);
    for (std::size_t index = 0; index < first.heights.size(); ++index)
    {
        EXPECT_EQ(first.heights[index], second.heights[index]);
    }
}

TEST(TerrainGenerator, DifferentSeedsChangeTheLandscape)
{
    TerrainSettings settings = MakeTestSettings(32u);
    const TerrainHeightfield first = GenerateTerrain(settings);
    settings.seed += 1u;
    const TerrainHeightfield second = GenerateTerrain(settings);

    bool differs = false;
    for (std::size_t index = 0; index < first.heights.size() && !differs; ++index)
    {
        differs = first.heights[index] != second.heights[index];
    }
    EXPECT_TRUE(differs);
}

TEST(TerrainGenerator, ProducesFiniteMountainsAndGreenland)
{
    const TerrainSettings settings = MakeTestSettings();
    const TerrainHeightfield heightfield = GenerateTerrain(settings);

    ASSERT_FALSE(heightfield.Empty());
    EXPECT_GT(heightfield.stats.maxHeightMeters, settings.maxHeightMeters * 0.4f);
    EXPECT_GT(heightfield.stats.maxSlopeDegrees, 20.0f);
    EXPECT_GT(heightfield.stats.generationSeconds, 0.0);
    EXPECT_GT(heightfield.stats.hydraulicDroplets, 0u);
    EXPECT_GT(heightfield.stats.thermalIterations, 0u);

    for (std::size_t index = 0; index < heightfield.heights.size(); ++index)
    {
        EXPECT_TRUE(std::isfinite(heightfield.heights[index]));
        EXPECT_GE(heightfield.heights[index], heightfield.waterLevelMeters - 1e-3f);
        EXPECT_LE(heightfield.heights[index], settings.maxHeightMeters + 1e-3f);
        const glm::vec3& normal = heightfield.normals[index];
        EXPECT_NEAR(glm::length(normal), 1.0f, 1e-4f);
    }

    // The heightfield should contain water, vegetation and high rocky/snowy terrain.
    EXPECT_GT(heightfield.stats.biomeFractions[static_cast<std::size_t>(TerrainBiome::Water)], 0.0f);
    const float greenland =
        heightfield.stats.biomeFractions[static_cast<std::size_t>(TerrainBiome::Grassland)] +
        heightfield.stats.biomeFractions[static_cast<std::size_t>(TerrainBiome::Forest)];
    const float highland =
        heightfield.stats.biomeFractions[static_cast<std::size_t>(TerrainBiome::Rock)] +
        heightfield.stats.biomeFractions[static_cast<std::size_t>(TerrainBiome::Snow)];
    EXPECT_GT(greenland, 0.0f);
    EXPECT_GT(highland, 0.0f);

    float biomeSum = 0.0f;
    for (const float fraction : heightfield.stats.biomeFractions)
    {
        biomeSum += fraction;
    }
    EXPECT_NEAR(biomeSum, 1.0f, 1e-3f);
}

TEST(TerrainGenerator, ClampsResolutionIntoSafeBounds)
{
    TerrainSettings settings = MakeTestSettings(1u);
    const TerrainHeightfield heightfield = GenerateTerrain(settings);
    EXPECT_EQ(heightfield.resolution, kMinTerrainResolution);
    EXPECT_EQ(heightfield.heights.size(), (kMinTerrainResolution + 1u) * (kMinTerrainResolution + 1u));
}

TEST(TerrainBiomes, ClassificationRespectsHeightSlopeAndMoisture)
{
    TerrainSettings settings{};
    settings.maxHeightMeters = 100.0f;
    settings.waterLevel = 0.10f;
    settings.beachBand = 0.05f;
    settings.snowLine = 0.70f;
    settings.treeLine = 0.50f;
    settings.rockSlope = 0.40f;
    settings.forestMoisture = 0.50f;

    EXPECT_EQ(ClassifyTerrainBiome(5.0f, 0.0f, 0.5f, settings), TerrainBiome::Water);
    EXPECT_EQ(ClassifyTerrainBiome(12.0f, 0.1f, 0.5f, settings), TerrainBiome::Beach);
    EXPECT_EQ(ClassifyTerrainBiome(30.0f, 0.1f, 0.9f, settings), TerrainBiome::Forest);
    EXPECT_EQ(ClassifyTerrainBiome(30.0f, 0.1f, 0.1f, settings), TerrainBiome::Grassland);
    EXPECT_EQ(ClassifyTerrainBiome(55.0f, 0.1f, 0.5f, settings), TerrainBiome::Rock);
    EXPECT_EQ(ClassifyTerrainBiome(75.0f, 0.1f, 0.5f, settings), TerrainBiome::Snow);
    EXPECT_EQ(ClassifyTerrainBiome(30.0f, 0.8f, 0.9f, settings), TerrainBiome::Rock);
}

TEST(TerrainErosion, ThermalRelaxationLowersSteepSlopes)
{
    constexpr std::uint32_t kResolution = 16u;
    constexpr float kCellSize = 4.0f;
    std::vector<float> heights(
        static_cast<std::size_t>(kResolution + 1u) * (kResolution + 1u),
        0.0f);

    const std::size_t center = GridIndex(kResolution, kResolution / 2u, kResolution / 2u);
    heights[center] = 80.0f;
    const float initialSum = std::accumulate(heights.begin(), heights.end(), 0.0f);
    const float initialSlope = MaximumSlopeDegrees(heights, kResolution, kCellSize);

    const std::uint64_t iterations =
        ApplyThermalErosion(heights, kResolution, kCellSize, 25.0f, 40u, 0.5f);
    EXPECT_GT(iterations, 0u);
    EXPECT_LT(heights[center], 80.0f);

    const float finalSum = std::accumulate(heights.begin(), heights.end(), 0.0f);
    const float finalSlope = MaximumSlopeDegrees(heights, kResolution, kCellSize);
    EXPECT_NEAR(finalSum, initialSum, 1.0f);
    EXPECT_LT(finalSlope, initialSlope);
}

TEST(TerrainErosion, HydraulicErosionReshapesTheHeightfield)
{
    TerrainSettings settings = MakeTestSettings(32u);
    settings.thermalErosion = false;
    settings.hydraulicErosion = false;
    const TerrainHeightfield uneroded = GenerateTerrain(settings);

    settings.hydraulicErosion = true;
    const TerrainHeightfield eroded = GenerateTerrain(settings);

    EXPECT_EQ(uneroded.stats.hydraulicDroplets, 0u);
    EXPECT_GT(eroded.stats.hydraulicDroplets, 0u);

    bool hasEroded = false;
    bool hasDeposited = false;
    for (std::size_t index = 0; index < uneroded.heights.size(); ++index)
    {
        const float difference = eroded.heights[index] - uneroded.heights[index];
        hasEroded = hasEroded || difference < -1e-5f;
        hasDeposited = hasDeposited || difference > 1e-5f;
    }
    EXPECT_TRUE(hasEroded);
    EXPECT_TRUE(hasDeposited);
}

TEST(TerrainMesh, BuildsWorldSpaceGeometryAndMaterial)
{
    constexpr std::uint32_t kResolution = 8u;
    TerrainHeightfield heightfield{};
    heightfield.settings.sizeMeters = 400.0f;
    heightfield.settings.maxHeightMeters = 100.0f;
    heightfield.resolution = kResolution;
    heightfield.cellSizeMeters = 50.0f;
    const std::size_t vertexCount =
        static_cast<std::size_t>(kResolution + 1u) * (kResolution + 1u);
    heightfield.heights.assign(vertexCount, 12.0f);
    heightfield.normals.assign(vertexCount, glm::vec3{0.0f, 1.0f, 0.0f});
    heightfield.moisture.assign(vertexCount, 0.5f);
    heightfield.biomes.assign(vertexCount, TerrainBiome::Grassland);
    heightfield.albedo.assign(vertexCount * 4u, 128u);

    const GltfModel model = BuildTerrainModel(heightfield);

    ASSERT_EQ(model.primitives.size(), 1u);
    ASSERT_EQ(model.nodes.size(), 1u);
    ASSERT_EQ(model.materials.size(), 1u);
    ASSERT_EQ(model.images.size(), 1u);
    ASSERT_EQ(model.textures.size(), 1u);
    EXPECT_EQ(model.meshPrimitives.size(), 1u);

    const GltfPrimitive& primitive = model.primitives[0];
    EXPECT_EQ(primitive.vertices.size(), vertexCount);
    EXPECT_EQ(primitive.indices.size(), static_cast<std::size_t>(kResolution) * kResolution * 6u);
    EXPECT_EQ(primitive.materialIndex, 0);
    EXPECT_EQ(model.vertexCount, vertexCount);
    EXPECT_EQ(model.triangleCount, static_cast<std::uint64_t>(kResolution) * kResolution * 2u);
    EXPECT_EQ(model.materials[0].baseColorTexture, 0);
    EXPECT_EQ(model.textures[0].imageIndex, 0);
    EXPECT_EQ(model.images[0].width, kResolution + 1u);
    EXPECT_EQ(model.images[0].height, kResolution + 1u);
    EXPECT_EQ(model.images[0].pixels.size(), vertexCount * 4u);

    for (const GltfVertex& vertex : primitive.vertices)
    {
        EXPECT_NEAR(vertex.normal.y, 1.0f, 1e-5f);
        EXPECT_GE(vertex.uv.x, 0.0f);
        EXPECT_LE(vertex.uv.x, 1.0f);
        EXPECT_GE(vertex.uv.y, 0.0f);
        EXPECT_LE(vertex.uv.y, 1.0f);
        EXPECT_FLOAT_EQ(vertex.weights[0], 1.0f);
    }

    EXPECT_NEAR(model.minBounds.x, -200.0f, 1e-3f);
    EXPECT_NEAR(model.maxBounds.x, 200.0f, 1e-3f);
    EXPECT_NEAR(model.minBounds.z, -200.0f, 1e-3f);
    EXPECT_NEAR(model.maxBounds.z, 200.0f, 1e-3f);
    EXPECT_NEAR(model.minBounds.y, 12.0f, 1e-3f);

    // Upward facing triangles: the cross product of the first triangle points along +Y.
    const glm::vec3 first =
        primitive.vertices[primitive.indices[0]].position;
    const glm::vec3 second =
        primitive.vertices[primitive.indices[1]].position;
    const glm::vec3 third =
        primitive.vertices[primitive.indices[2]].position;
    EXPECT_GT(glm::cross(second - first, third - first).y, 0.0f);
}

TEST(TerrainPreview, MatchesTheHeightfieldResolution)
{
    const TerrainHeightfield heightfield = GenerateTerrain(MakeTestSettings(16u));
    const std::vector<std::uint8_t> preview = BuildTerrainPreview(heightfield);
    ASSERT_EQ(preview.size(), heightfield.heights.size() * 4u);
    for (std::size_t index = 3; index < preview.size(); index += 4)
    {
        EXPECT_EQ(preview[index], 255u);
    }
}

} // namespace
} // namespace mukki
