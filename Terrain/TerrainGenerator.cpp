#include "Terrain/TerrainGenerator.h"

#include "Terrain/TerrainErosion.h"
#include "Terrain/TerrainNoise.h"
#include "Utilities/Assert.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

namespace mukki
{
namespace
{

constexpr float kPi = 3.14159265358979f;

float SmoothStep(float edge0, float edge1, float value)
{
    if (edge0 >= edge1)
    {
        return value >= edge1 ? 1.0f : 0.0f;
    }
    const float t = std::clamp((value - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float Lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

std::uint8_t ToByte(float value)
{
    return static_cast<std::uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

TerrainSettings SanitizeSettings(const TerrainSettings& source)
{
    TerrainSettings settings = source;
    settings.resolution = std::clamp(settings.resolution, kMinTerrainResolution, kMaxTerrainResolution);
    settings.sizeMeters = std::max(settings.sizeMeters, 16.0f);
    settings.maxHeightMeters = std::max(settings.maxHeightMeters, 1.0f);
    settings.continentFrequency = std::clamp(settings.continentFrequency, 0.05f, 32.0f);
    settings.mountainFrequency = std::clamp(settings.mountainFrequency, 0.05f, 64.0f);
    settings.octaves = std::clamp(settings.octaves, 1u, 12u);
    settings.lacunarity = std::clamp(settings.lacunarity, 1.01f, 4.0f);
    settings.persistence = std::clamp(settings.persistence, 0.0f, 0.95f);
    settings.ridgeSharpness = std::clamp(settings.ridgeSharpness, 0.1f, 8.0f);
    settings.mountainCoverage = std::clamp(settings.mountainCoverage, 0.0f, 0.95f);
    settings.domainWarp = std::clamp(settings.domainWarp, 0.0f, 2.0f);

    settings.hydraulicDropletDensity = std::clamp(settings.hydraulicDropletDensity, 0.0f, 4.0f);
    settings.hydraulicLifetime = std::clamp(settings.hydraulicLifetime, 1u, 128u);
    settings.hydraulicBrushRadius = std::clamp(settings.hydraulicBrushRadius, 1u, 12u);
    settings.hydraulicInertia = std::clamp(settings.hydraulicInertia, 0.0f, 1.0f);
    settings.hydraulicCapacity = std::max(settings.hydraulicCapacity, 0.0f);
    settings.hydraulicMinCapacity = std::max(settings.hydraulicMinCapacity, 0.0f);
    settings.hydraulicErodeSpeed = std::clamp(settings.hydraulicErodeSpeed, 0.0f, 1.0f);
    settings.hydraulicDepositSpeed = std::clamp(settings.hydraulicDepositSpeed, 0.0f, 1.0f);
    settings.hydraulicEvaporateSpeed = std::clamp(settings.hydraulicEvaporateSpeed, 0.0f, 1.0f);
    settings.hydraulicGravity = std::max(settings.hydraulicGravity, 0.0f);

    settings.thermalIterations = std::clamp(settings.thermalIterations, 0u, 100u);
    settings.thermalTalusAngleDegrees = std::clamp(settings.thermalTalusAngleDegrees, 0.0f, 70.0f);
    settings.thermalRate = std::clamp(settings.thermalRate, 0.0f, 1.0f);

    settings.waterLevel = std::clamp(settings.waterLevel, 0.0f, 0.95f);
    settings.beachBand = std::clamp(settings.beachBand, 0.0f, 0.4f);
    settings.snowLine = std::clamp(settings.snowLine, 0.05f, 1.0f);
    settings.treeLine = std::clamp(settings.treeLine, 0.0f, settings.snowLine);
    settings.rockSlope = std::clamp(settings.rockSlope, 0.02f, 0.99f);
    settings.forestMoisture = std::clamp(settings.forestMoisture, 0.0f, 1.0f);
    return settings;
}

glm::vec3 BiomeColor(
    TerrainBiome biome,
    float normalizedHeight,
    float moisture,
    float variation)
{
    const float variation01 = std::clamp(variation * 0.5f + 0.5f, 0.0f, 1.0f);
    switch (biome)
    {
    case TerrainBiome::Water:
        return glm::mix(
            glm::vec3{0.025f, 0.11f, 0.22f},
            glm::vec3{0.070f, 0.23f, 0.32f},
            variation01);

    case TerrainBiome::Beach:
        return glm::mix(
            glm::vec3{0.72f, 0.66f, 0.47f},
            glm::vec3{0.84f, 0.78f, 0.58f},
            variation01);

    case TerrainBiome::Grassland:
    {
        const glm::vec3 dry{0.33f, 0.47f, 0.19f};
        const glm::vec3 lush{0.16f, 0.37f, 0.12f};
        return glm::mix(dry, lush, std::clamp(moisture * 0.8f + variation01 * 0.2f, 0.0f, 1.0f));
    }

    case TerrainBiome::Forest:
        return glm::mix(
            glm::vec3{0.065f, 0.21f, 0.075f},
            glm::vec3{0.13f, 0.32f, 0.11f},
            variation01);

    case TerrainBiome::Rock:
    {
        const glm::vec3 stone = glm::mix(
            glm::vec3{0.29f, 0.27f, 0.25f},
            glm::vec3{0.47f, 0.45f, 0.42f},
            variation01);
        // Moss and lichen take over the lower rock faces where moisture is available.
        const float moss = (1.0f - normalizedHeight) * moisture * 0.3f;
        return glm::mix(stone, glm::vec3{0.24f, 0.31f, 0.17f}, std::clamp(moss, 0.0f, 1.0f));
    }

    case TerrainBiome::Snow:
        return glm::mix(
            glm::vec3{0.80f, 0.85f, 0.90f},
            glm::vec3{0.94f, 0.96f, 0.99f},
            variation01);
    }

    return glm::vec3{0.2f, 0.4f, 0.15f};
}

} // namespace

TerrainHeightfield GenerateTerrain(const TerrainSettings& sourceSettings)
{
    const auto startTime = std::chrono::steady_clock::now();

    TerrainHeightfield heightfield{};
    const TerrainSettings settings = SanitizeSettings(sourceSettings);
    heightfield.settings = settings;
    heightfield.resolution = settings.resolution;
    heightfield.cellSizeMeters = settings.sizeMeters / static_cast<float>(settings.resolution);
    heightfield.waterLevelMeters = settings.waterLevel * settings.maxHeightMeters;
    heightfield.stats.waterLevelMeters = heightfield.waterLevelMeters;

    const std::size_t vertexCount =
        static_cast<std::size_t>(heightfield.SideVertices()) * heightfield.SideVertices();
    heightfield.heights.assign(vertexCount, 0.0f);
    heightfield.moisture.assign(vertexCount, 0.5f);
    heightfield.biomes.assign(vertexCount, TerrainBiome::Grassland);
    heightfield.albedo.assign(vertexCount * 4u, 255u);

    const std::uint32_t seed = settings.seed;
    const PerlinNoise2D continentNoise{seed + 0x1B873593u};
    const PerlinNoise2D hillNoise{seed + 0x0E9B4A91u};
    const PerlinNoise2D mountainNoise{seed + 0x2C1B3C6Du};
    const PerlinNoise2D warpNoise{seed + 0x68E31DA4u};
    const PerlinNoise2D detailNoise{seed + 0x05BD1C7Fu};
    const PerlinNoise2D moistureNoise{seed + 0x7A9D02E5u};

    const float resolutionFloat = static_cast<float>(settings.resolution);
    const float warpScale = settings.continentFrequency * 1.6f;
    const float hillScale = settings.continentFrequency * 2.7f;

    for (std::uint32_t z = 0; z <= settings.resolution; ++z)
    {
        for (std::uint32_t x = 0; x <= settings.resolution; ++x)
        {
            const float u = static_cast<float>(x) / resolutionFloat;
            const float v = static_cast<float>(z) / resolutionFloat;

            // Warp the mountain sampling position so ridges bend and braid.
            const float warpX =
                (warpNoise.Fbm(
                     u * warpScale + 5.2f,
                     v * warpScale + 1.3f,
                     3u,
                     2.0f,
                     0.5f) -
                 0.5f) *
                settings.domainWarp * 2.0f;
            const float warpZ =
                (warpNoise.Fbm(
                     u * warpScale + 9.7f,
                     v * warpScale + 3.8f,
                     3u,
                     2.0f,
                     0.5f) -
                 0.5f) *
                settings.domainWarp * 2.0f;

            const float continent = continentNoise.Fbm(
                u * settings.continentFrequency + 0.37f,
                v * settings.continentFrequency + 5.11f,
                4u,
                2.0f,
                0.5f);
            const float hills = hillNoise.Fbm(
                u * hillScale + 3.3f,
                v * hillScale + 8.9f,
                5u,
                settings.lacunarity,
                settings.persistence);
            const float mountain = mountainNoise.RidgedFbm(
                (u + warpX) * settings.mountainFrequency + 2.1f,
                (v + warpZ) * settings.mountainFrequency + 7.4f,
                settings.octaves,
                settings.lacunarity,
                settings.persistence,
                settings.ridgeSharpness);

            const float seaMask = SmoothStep(0.28f, 0.60f, continent);
            const float mountainMask = SmoothStep(
                settings.mountainCoverage,
                std::min(settings.mountainCoverage + 0.32f, 1.0f),
                continent);

            const float lowland = Lerp(-0.045f, 0.07f + hills * 0.26f, seaMask);
            const float mountainSurface = 0.26f + mountain * 0.74f;
            float elevation = Lerp(lowland, mountainSurface, mountainMask);

            // Broad undulation and a little high-frequency roughness keep the greenland from
            // looking like a smooth analytic surface.
            elevation -=
                (continentNoise.Fbm(
                     u * 0.9f + 13.2f,
                     v * 0.9f + 17.4f,
                     2u,
                     2.0f,
                     0.5f) -
                 0.5f) *
                0.10f * seaMask;
            elevation +=
                (detailNoise.Fbm(
                     u * 14.0f + 7.9f,
                     v * 14.0f + 1.7f,
                     3u,
                     2.0f,
                     0.5f) -
                 0.5f) *
                0.025f * seaMask;
            elevation = std::clamp(elevation, -0.12f, 1.05f);

            const std::size_t index = heightfield.Index(x, z);
            heightfield.heights[index] = elevation;

            const float elevationNormalized = std::clamp(elevation, 0.0f, 1.0f);
            const float baseMoisture = moistureNoise.Fbm(
                u * 2.2f + 11.1f,
                v * 2.2f + 6.7f,
                4u,
                2.0f,
                0.5f);
            heightfield.moisture[index] = std::clamp(
                baseMoisture * 0.85f + (1.0f - elevationNormalized) * 0.10f + 0.02f,
                0.0f,
                1.0f);
        }
    }

    if (settings.hydraulicErosion)
    {
        heightfield.stats.hydraulicDroplets = ApplyHydraulicErosion(
            heightfield.heights,
            settings.resolution,
            settings);
    }

    // Rescale the eroded field so the highest summit lands on maxHeightMeters. This keeps the
    // biome thresholds meaningful even when erosion shaves the peaks.
    float maximum = 0.001f;
    for (const float height : heightfield.heights)
    {
        maximum = std::max(maximum, height);
    }
    const float heightScale = settings.maxHeightMeters / maximum;
    for (float& height : heightfield.heights)
    {
        height *= heightScale;
    }

    if (settings.thermalErosion)
    {
        heightfield.stats.thermalIterations = ApplyThermalErosion(
            heightfield.heights,
            settings.resolution,
            heightfield.cellSizeMeters,
            settings.thermalTalusAngleDegrees,
            settings.thermalIterations,
            settings.thermalRate);
    }

    // Flatten everything below the water line so seas and lakes read as level water surfaces.
    for (float& height : heightfield.heights)
    {
        height = std::max(height, heightfield.waterLevelMeters);
    }

    ComputeTerrainNormals(
        heightfield.heights,
        settings.resolution,
        heightfield.cellSizeMeters,
        heightfield.normals);

    float minHeight = heightfield.heights.empty() ? 0.0f : heightfield.heights[0];
    float maxHeight = heightfield.heights.empty() ? 0.0f : heightfield.heights[0];
    double heightSum = 0.0;
    float maxSlopeDegrees = 0.0f;

    for (std::uint32_t z = 0; z <= settings.resolution; ++z)
    {
        for (std::uint32_t x = 0; x <= settings.resolution; ++x)
        {
            const std::size_t index = heightfield.Index(x, z);
            const float height = heightfield.heights[index];
            const glm::vec3& normal = heightfield.normals[index];
            const float slope = TerrainSlope(normal);
            const TerrainBiome biome =
                ClassifyTerrainBiome(height, slope, heightfield.moisture[index], settings);

            heightfield.biomes[index] = biome;
            const std::size_t biomeIndex = static_cast<std::size_t>(biome);
            heightfield.stats.biomeFractions[biomeIndex] += 1.0f;

            const float variation =
                HashToUnit2D(
                    static_cast<std::int32_t>(x),
                    static_cast<std::int32_t>(z),
                    seed ^ 0x5BD1E995u) *
                    2.0f -
                1.0f;
            const float normalizedHeight = std::clamp(
                height / std::max(settings.maxHeightMeters, 1e-3f),
                0.0f,
                1.0f);
            const glm::vec3 color = BiomeColor(
                biome,
                normalizedHeight,
                heightfield.moisture[index],
                variation);
            heightfield.albedo[index * 4u + 0u] = ToByte(color.r);
            heightfield.albedo[index * 4u + 1u] = ToByte(color.g);
            heightfield.albedo[index * 4u + 2u] = ToByte(color.b);
            heightfield.albedo[index * 4u + 3u] = 255u;

            minHeight = std::min(minHeight, height);
            maxHeight = std::max(maxHeight, height);
            heightSum += static_cast<double>(height);
            maxSlopeDegrees = std::max(
                maxSlopeDegrees,
                std::acos(std::clamp(normal.y, -1.0f, 1.0f)) * 180.0f / kPi);
        }
    }

    const float vertexCountFloat = static_cast<float>(std::max<std::size_t>(vertexCount, 1u));
    heightfield.stats.minHeightMeters = minHeight;
    heightfield.stats.maxHeightMeters = maxHeight;
    heightfield.stats.meanHeightMeters =
        static_cast<float>(heightSum / static_cast<double>(std::max<std::size_t>(vertexCount, 1u)));
    heightfield.stats.maxSlopeDegrees = maxSlopeDegrees;
    for (float& fraction : heightfield.stats.biomeFractions)
    {
        fraction /= vertexCountFloat;
    }

    const auto endTime = std::chrono::steady_clock::now();
    heightfield.stats.generationSeconds =
        std::chrono::duration<double>(endTime - startTime).count();
    return heightfield;
}

TerrainBiome ClassifyTerrainBiome(
    float heightMeters,
    float slope,
    float moisture,
    const TerrainSettings& settings)
{
    const float maxHeight = std::max(settings.maxHeightMeters, 1e-3f);
    const float water = settings.waterLevel * maxHeight;
    const float beach = water + settings.beachBand * maxHeight;
    const float tree = settings.treeLine * maxHeight;
    const float snow = settings.snowLine * maxHeight;

    if (heightMeters <= water + 1e-3f)
    {
        return TerrainBiome::Water;
    }
    if (slope >= settings.rockSlope)
    {
        return TerrainBiome::Rock;
    }
    if (heightMeters <= beach)
    {
        return TerrainBiome::Beach;
    }
    if (heightMeters >= snow)
    {
        return TerrainBiome::Snow;
    }
    if (heightMeters >= tree)
    {
        return TerrainBiome::Rock;
    }
    if (moisture >= settings.forestMoisture)
    {
        return TerrainBiome::Forest;
    }
    return TerrainBiome::Grassland;
}

std::vector<std::uint8_t> BuildTerrainPreview(const TerrainHeightfield& heightfield)
{
    if (heightfield.Empty())
    {
        return {};
    }

    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(heightfield.VertexCount()) * 4u,
        255u);
    const glm::vec3 sunDirection = glm::normalize(glm::vec3{-0.55f, 0.75f, -0.35f});

    for (std::uint32_t z = 0; z <= heightfield.resolution; ++z)
    {
        for (std::uint32_t x = 0; x <= heightfield.resolution; ++x)
        {
            const std::size_t index = heightfield.Index(x, z);
            const glm::vec3& normal = heightfield.normals[index];
            const float light = 0.35f + 0.65f * std::clamp(glm::dot(normal, sunDirection), 0.0f, 1.0f);
            for (std::size_t channel = 0; channel < 3u; ++channel)
            {
                const float albedo =
                    static_cast<float>(heightfield.albedo[index * 4u + channel]) / 255.0f;
                pixels[index * 4u + channel] = ToByte(albedo * light);
            }
            pixels[index * 4u + 3u] = 255u;
        }
    }

    return pixels;
}

} // namespace mukki
