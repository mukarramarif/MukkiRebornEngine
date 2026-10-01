#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "Renderer/FrameView.h"
#include "Renderer/Lighting/Light.h"

namespace mukki
{

inline constexpr std::uint32_t kShadowCascadeCount = 4;
inline constexpr std::uint32_t kShadowResolution = 1024;
inline constexpr std::uint32_t kMaxSpotShadowLights = 4;
inline constexpr std::uint32_t kSpotShadowResolution = 1024;
inline constexpr std::uint32_t kMaxPointShadowLights = 2;
inline constexpr std::uint32_t kPointShadowFaceCount = 6;
inline constexpr std::uint32_t kPointShadowTileSize = 512;
inline constexpr std::uint32_t kPointShadowAtlasColumns = 3;
inline constexpr std::uint32_t kPointShadowAtlasRows = 2 * kMaxPointShadowLights;
inline constexpr std::uint32_t kPointShadowAtlasWidth =
    kPointShadowAtlasColumns * kPointShadowTileSize;
inline constexpr std::uint32_t kPointShadowAtlasHeight =
    kPointShadowAtlasRows * kPointShadowTileSize;
inline constexpr float kShadowNearPlane = 0.05f;

// Shadow matrices live in one flat array per frame: cascades first, then spot maps, then the six
// faces per point cube.
inline constexpr std::uint32_t kSpotShadowOffset = kShadowCascadeCount;
inline constexpr std::uint32_t kPointShadowOffset = kSpotShadowOffset + kMaxSpotShadowLights;
inline constexpr std::uint32_t kMaxShadowViews =
    kPointShadowOffset + kMaxPointShadowLights * kPointShadowFaceCount;

struct ShadowSettings
{
    bool enabled = true;
    std::uint32_t cascadeCount = kShadowCascadeCount;
    std::uint32_t resolution = kShadowResolution;
    float distance = 100.0f;   // world units covered by the last cascade
    float lambda = 0.95f;      // 0 = uniform splits, 1 = logarithmic splits
    float depthBias = 1.25f;   // constant depth bias applied in the shadow pass
    float normalBias = 0.03f;  // world-space offset along the surface normal when sampling
    // Optional hardware ray traced sun shadows. When enabled the lighting pass resolves the sun
    // visibility from a traced mask instead of the cascaded shadow map.
    bool rtSunShadows = true;
    std::uint32_t rtShadowSamples = 4;
    float rtSunAngularRadiusDegrees = 0.5f;
};

struct ShadowCascade
{
    glm::mat4 lightViewProjection{1.0f};
    float splitNear = 0.0f;
    float splitFar = 0.0f;
};

// Builds a stable light-space projection per cascade for the given camera view. `lightDirection`
// points from the light into the scene and is normalized internally.
[[nodiscard]] std::vector<ShadowCascade> ComputeShadowCascades(
    const FrameView& view,
    const glm::vec3& lightDirection,
    const ShadowSettings& settings);

// Perspective projection covering the spot light cone. `direction` points from the light into the
// scene.
[[nodiscard]] glm::mat4 ComputeSpotShadowMatrix(const Light& light, float nearPlane);

// Six perspective projections for a point light, ordered +X, -X, +Y, -Y, +Z, -Z. The lighting
// shader selects the face from the major axis of the light-to-fragment direction.
[[nodiscard]] std::array<glm::mat4, kPointShadowFaceCount> ComputePointShadowMatrices(
    const Light& light,
    float nearPlane);

} // namespace mukki
