#include "Terrain/TerrainErosion.h"

#include "Terrain/TerrainTypes.h"
#include "Utilities/Assert.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>

namespace mukki
{
namespace
{

constexpr std::uint64_t kMaxHydraulicDroplets = 250000u;

class SplitMixRandom
{
public:
    explicit SplitMixRandom(std::uint64_t seed) : m_state(seed) {}

    [[nodiscard]] std::uint64_t Next()
    {
        m_state += 0x9E3779B97F4A7C15ull;
        std::uint64_t result = m_state;
        result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9ull;
        result = (result ^ (result >> 27)) * 0x94D049BB133111EBull;
        return result ^ (result >> 31);
    }

    [[nodiscard]] float Unit()
    {
        return static_cast<float>(Next() >> 40) / 16777216.0f;
    }

private:
    std::uint64_t m_state = 0;
};

struct ErosionBrushEntry
{
    std::int32_t dx = 0;
    std::int32_t dz = 0;
    float weight = 0.0f;
};

struct ErosionBrush
{
    std::vector<ErosionBrushEntry> entries{};
};

ErosionBrush BuildErosionBrush(std::uint32_t radius)
{
    ErosionBrush brush{};
    const std::int32_t signedRadius = static_cast<std::int32_t>(radius);
    float totalWeight = 0.0f;

    for (std::int32_t dz = -signedRadius; dz <= signedRadius; ++dz)
    {
        for (std::int32_t dx = -signedRadius; dx <= signedRadius; ++dx)
        {
            const float distance = std::sqrt(
                static_cast<float>(dx * dx + dz * dz));
            if (distance > static_cast<float>(signedRadius))
            {
                continue;
            }

            const float weight = 1.0f - distance / static_cast<float>(std::max(radius, 1u));
            if (weight <= 0.0f)
            {
                continue;
            }

            brush.entries.push_back(ErosionBrushEntry{dx, dz, weight});
            totalWeight += weight;
        }
    }

    if (totalWeight > 0.0f)
    {
        for (ErosionBrushEntry& entry : brush.entries)
        {
            entry.weight /= totalWeight;
        }
    }
    return brush;
}

float SampleHeight(
    const std::vector<float>& heights,
    std::uint32_t resolution,
    float x,
    float z)
{
    const float maxCoordinate = static_cast<float>(resolution);
    const float clampedX = std::clamp(x, 0.0f, maxCoordinate);
    const float clampedZ = std::clamp(z, 0.0f, maxCoordinate);

    const std::uint32_t x0 = std::min(
        static_cast<std::uint32_t>(clampedX),
        resolution > 0u ? resolution - 1u : 0u);
    const std::uint32_t z0 = std::min(
        static_cast<std::uint32_t>(clampedZ),
        resolution > 0u ? resolution - 1u : 0u);
    const std::uint32_t x1 = std::min(x0 + 1u, resolution);
    const std::uint32_t z1 = std::min(z0 + 1u, resolution);

    const float tx = clampedX - static_cast<float>(x0);
    const float tz = clampedZ - static_cast<float>(z0);

    const std::size_t side = static_cast<std::size_t>(resolution) + 1u;
    const float h00 = heights[static_cast<std::size_t>(z0) * side + x0];
    const float h10 = heights[static_cast<std::size_t>(z0) * side + x1];
    const float h01 = heights[static_cast<std::size_t>(z1) * side + x0];
    const float h11 = heights[static_cast<std::size_t>(z1) * side + x1];

    const float bottom = h00 + (h10 - h00) * tx;
    const float top = h01 + (h11 - h01) * tx;
    return bottom + (top - bottom) * tz;
}

glm::vec2 HeightGradient(
    const std::vector<float>& heights,
    std::uint32_t resolution,
    float x,
    float z)
{
    const float left = SampleHeight(heights, resolution, x - 1.0f, z);
    const float right = SampleHeight(heights, resolution, x + 1.0f, z);
    const float down = SampleHeight(heights, resolution, x, z - 1.0f);
    const float up = SampleHeight(heights, resolution, x, z + 1.0f);
    return glm::vec2{(right - left) * 0.5f, (up - down) * 0.5f};
}

void DepositBilinear(
    std::vector<float>& heights,
    std::uint32_t resolution,
    float x,
    float z,
    float amount)
{
    if (amount <= 0.0f)
    {
        return;
    }

    const float maxCoordinate = static_cast<float>(resolution);
    const float clampedX = std::clamp(x, 0.0f, maxCoordinate);
    const float clampedZ = std::clamp(z, 0.0f, maxCoordinate);
    const std::uint32_t x0 = std::min(
        static_cast<std::uint32_t>(clampedX),
        resolution > 0u ? resolution - 1u : 0u);
    const std::uint32_t z0 = std::min(
        static_cast<std::uint32_t>(clampedZ),
        resolution > 0u ? resolution - 1u : 0u);
    const std::uint32_t x1 = std::min(x0 + 1u, resolution);
    const std::uint32_t z1 = std::min(z0 + 1u, resolution);
    const float tx = clampedX - static_cast<float>(x0);
    const float tz = clampedZ - static_cast<float>(z0);

    const std::size_t side = static_cast<std::size_t>(resolution) + 1u;
    heights[static_cast<std::size_t>(z0) * side + x0] += amount * (1.0f - tx) * (1.0f - tz);
    heights[static_cast<std::size_t>(z0) * side + x1] += amount * tx * (1.0f - tz);
    heights[static_cast<std::size_t>(z1) * side + x0] += amount * (1.0f - tx) * tz;
    heights[static_cast<std::size_t>(z1) * side + x1] += amount * tx * tz;
}

void ErodeBrushAt(
    std::vector<float>& heights,
    std::uint32_t resolution,
    const ErosionBrush& brush,
    std::uint32_t nodeX,
    std::uint32_t nodeZ,
    float amount)
{
    if (amount <= 0.0f || brush.entries.empty())
    {
        return;
    }

    const std::int32_t signedResolution = static_cast<std::int32_t>(resolution);
    const std::int32_t baseX = static_cast<std::int32_t>(nodeX);
    const std::int32_t baseZ = static_cast<std::int32_t>(nodeZ);
    const std::size_t side = static_cast<std::size_t>(resolution) + 1u;

    float validWeight = 0.0f;
    for (const ErosionBrushEntry& entry : brush.entries)
    {
        const std::int32_t x = baseX + entry.dx;
        const std::int32_t z = baseZ + entry.dz;
        if (x < 0 || x > signedResolution || z < 0 || z > signedResolution)
        {
            continue;
        }
        validWeight += entry.weight;
    }

    if (validWeight <= 0.0f)
    {
        return;
    }

    for (const ErosionBrushEntry& entry : brush.entries)
    {
        const std::int32_t x = baseX + entry.dx;
        const std::int32_t z = baseZ + entry.dz;
        if (x < 0 || x > signedResolution || z < 0 || z > signedResolution)
        {
            continue;
        }

        const float scaledWeight = entry.weight / validWeight;
        heights[static_cast<std::size_t>(z) * side + static_cast<std::size_t>(x)] -=
            amount * scaledWeight;
    }
}

} // namespace

void ComputeTerrainNormals(
    const std::vector<float>& heights,
    std::uint32_t resolution,
    float cellSizeMeters,
    std::vector<glm::vec3>& outNormals)
{
    const std::size_t side = static_cast<std::size_t>(resolution) + 1u;
    MRE_ASSERT(resolution > 0u, "Terrain normals need at least one cell");
    MRE_ASSERT(cellSizeMeters > 0.0f, "Terrain cell size must be positive");
    if (heights.size() != side * side)
    {
        outNormals.clear();
        return;
    }

    outNormals.assign(heights.size(), glm::vec3{0.0f, 1.0f, 0.0f});
    const float heightScale = 2.0f * cellSizeMeters;
    for (std::uint32_t z = 0; z <= resolution; ++z)
    {
        for (std::uint32_t x = 0; x <= resolution; ++x)
        {
            const std::uint32_t left = x > 0u ? x - 1u : x;
            const std::uint32_t right = x < resolution ? x + 1u : x;
            const std::uint32_t down = z > 0u ? z - 1u : z;
            const std::uint32_t up = z < resolution ? z + 1u : z;

            const float heightLeft = heights[static_cast<std::size_t>(z) * side + left];
            const float heightRight = heights[static_cast<std::size_t>(z) * side + right];
            const float heightDown = heights[static_cast<std::size_t>(down) * side + x];
            const float heightUp = heights[static_cast<std::size_t>(up) * side + x];

            const glm::vec3 normal{
                heightLeft - heightRight,
                heightScale,
                heightDown - heightUp};
            const float lengthSquared = glm::dot(normal, normal);
            outNormals[static_cast<std::size_t>(z) * side + x] =
                lengthSquared > 0.0f ? normal / std::sqrt(lengthSquared)
                                     : glm::vec3{0.0f, 1.0f, 0.0f};
        }
    }
}

float TerrainSlope(const glm::vec3& normal)
{
    return std::clamp(1.0f - normal.y, 0.0f, 1.0f);
}

std::uint64_t ApplyHydraulicErosion(
    std::vector<float>& heights,
    std::uint32_t resolution,
    const TerrainSettings& settings)
{
    const std::size_t side = static_cast<std::size_t>(resolution) + 1u;
    if (resolution < 2u || heights.size() != side * side)
    {
        return 0;
    }

    const float density = std::max(settings.hydraulicDropletDensity, 0.0f);
    const float erosionRate = std::clamp(settings.hydraulicErodeSpeed, 0.0f, 1.0f);
    const float depositionRate = std::clamp(settings.hydraulicDepositSpeed, 0.0f, 1.0f);
    const float inertia = std::clamp(settings.hydraulicInertia, 0.0f, 1.0f);
    const float evaporate = std::clamp(settings.hydraulicEvaporateSpeed, 0.0f, 1.0f);
    const float capacityFactor = std::max(settings.hydraulicCapacity, 0.0f);
    const float minCapacity = std::max(settings.hydraulicMinCapacity, 0.0f);
    const float gravity = std::max(settings.hydraulicGravity, 0.0f);
    const std::uint32_t lifetime = std::max(settings.hydraulicLifetime, 1u);

    if (density <= 0.0f)
    {
        return 0;
    }

    const double cellCount = static_cast<double>(resolution) * static_cast<double>(resolution);
    const std::uint64_t requested =
        static_cast<std::uint64_t>(std::llround(cellCount * static_cast<double>(density)));
    const std::uint64_t dropletCount = std::min(requested, kMaxHydraulicDroplets);
    if (dropletCount == 0u)
    {
        return 0;
    }

    std::uint32_t brushRadius = std::min(
        settings.hydraulicBrushRadius,
        std::max(resolution / 2u, 1u));
    brushRadius = std::max(brushRadius, 1u);
    const ErosionBrush brush = BuildErosionBrush(brushRadius);

    SplitMixRandom random{
        static_cast<std::uint64_t>(settings.seed) * 0x9E3779B97F4A7C15ull + 0x243F6A8885A308D3ull};
    const float maxCoordinate = static_cast<float>(resolution);
    const float margin = std::min(
        static_cast<float>(brushRadius) + 1.0f,
        maxCoordinate * 0.25f);
    const float spawnRange = std::max(maxCoordinate - margin * 2.0f, 1.0f);

    std::uint64_t simulated = 0;
    for (std::uint64_t droplet = 0; droplet < dropletCount; ++droplet)
    {
        float positionX = margin + random.Unit() * spawnRange;
        float positionZ = margin + random.Unit() * spawnRange;
        glm::vec2 direction{0.0f, 0.0f};
        float speed = 1.0f;
        float water = 1.0f;
        float sediment = 0.0f;
        ++simulated;

        for (std::uint32_t step = 0; step < lifetime; ++step)
        {
            const glm::vec2 gradient = HeightGradient(heights, resolution, positionX, positionZ);
            direction = direction * inertia - gradient * (1.0f - inertia);
            const float directionLengthSquared = glm::dot(direction, direction);
            if (directionLengthSquared <= 1e-8f)
            {
                break;
            }
            direction /= std::sqrt(directionLengthSquared);

            const float nextX = positionX + direction.x;
            const float nextZ = positionZ + direction.y;
            if (nextX < 0.0f || nextX > maxCoordinate || nextZ < 0.0f || nextZ > maxCoordinate)
            {
                break;
            }

            const float oldHeight = SampleHeight(heights, resolution, positionX, positionZ);
            const float newHeight = SampleHeight(heights, resolution, nextX, nextZ);
            const float heightDelta = newHeight - oldHeight;
            const float capacity = std::max(
                -heightDelta * speed * water * capacityFactor,
                minCapacity);

            if (sediment > capacity || heightDelta > 0.0f)
            {
                const float deposit = heightDelta > 0.0f
                                          ? std::min(heightDelta, sediment)
                                          : (sediment - capacity) * depositionRate;
                sediment -= deposit;
                DepositBilinear(heights, resolution, positionX, positionZ, deposit);
            }
            else
            {
                const float erode = std::min(
                    (capacity - sediment) * erosionRate,
                    -heightDelta);
                const std::uint32_t nodeX = static_cast<std::uint32_t>(
                    std::clamp(std::floor(positionX), 0.0f, maxCoordinate));
                const std::uint32_t nodeZ = static_cast<std::uint32_t>(
                    std::clamp(std::floor(positionZ), 0.0f, maxCoordinate));
                ErodeBrushAt(heights, resolution, brush, nodeX, nodeZ, erode);
                sediment += erode;
            }

            speed = std::sqrt(std::max(0.0f, speed * speed + -heightDelta * gravity));
            water *= (1.0f - evaporate);
            positionX = nextX;
            positionZ = nextZ;
            if (water < 0.01f)
            {
                break;
            }
        }
    }

    return simulated;
}

std::uint64_t ApplyThermalErosion(
    std::vector<float>& heights,
    std::uint32_t resolution,
    float cellSizeMeters,
    float talusAngleDegrees,
    std::uint32_t iterations,
    float rate)
{
    const std::size_t side = static_cast<std::size_t>(resolution) + 1u;
    if (resolution < 2u || heights.size() != side * side || iterations == 0u ||
        cellSizeMeters <= 0.0f)
    {
        return 0;
    }

    const float clampedAngle = std::clamp(talusAngleDegrees, 0.0f, 89.0f);
    const float talus = std::tan(clampedAngle * 3.14159265358979f / 180.0f) *
                        std::max(cellSizeMeters, 1e-4f);
    if (talus <= 0.0f)
    {
        return 0;
    }

    const float moveRate = std::clamp(rate, 0.0f, 1.0f);
    const std::int32_t offsets[8][2] = {
        {-1, -1},
        {0, -1},
        {1, -1},
        {-1, 0},
        {1, 0},
        {-1, 1},
        {0, 1},
        {1, 1},
    };

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        // Two checkerboard phases per iteration keep the update order from biasing the flow.
        for (std::uint32_t phase = 0; phase < 2u; ++phase)
        {
            for (std::uint32_t z = 1; z < resolution; ++z)
            {
                for (std::uint32_t x = 1; x < resolution; ++x)
                {
                    if (((x + z) & 1u) != phase)
                    {
                        continue;
                    }

                    const std::size_t center = static_cast<std::size_t>(z) * side + x;
                    float lowestHeight = heights[center];
                    std::size_t lowestIndex = center;
                    for (const auto& offset : offsets)
                    {
                        const std::size_t neighbor = static_cast<std::size_t>(
                                                         static_cast<std::int32_t>(z) + offset[1]) *
                                                         side +
                                                     static_cast<std::size_t>(
                                                         static_cast<std::int32_t>(x) + offset[0]);
                        if (heights[neighbor] < lowestHeight)
                        {
                            lowestHeight = heights[neighbor];
                            lowestIndex = neighbor;
                        }
                    }

                    const float difference = heights[center] - lowestHeight;
                    if (difference <= talus)
                    {
                        continue;
                    }

                    const float amount = (difference - talus) * moveRate * 0.5f;
                    heights[center] -= amount;
                    heights[lowestIndex] += amount;
                }
            }
        }
    }

    return iterations;
}

} // namespace mukki
