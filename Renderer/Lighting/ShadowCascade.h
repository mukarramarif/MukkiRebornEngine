#pragma once

#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "Renderer/FrameView.h"

namespace mukki
{

inline constexpr std::uint32_t kShadowCascadeCount = 4;
inline constexpr std::uint32_t kShadowResolution = 1024;

struct ShadowSettings
{
    bool enabled = true;
    std::uint32_t cascadeCount = kShadowCascadeCount;
    std::uint32_t resolution = kShadowResolution;
    float distance = 100.0f;   // world units covered by the last cascade
    float lambda = 0.95f;      // 0 = uniform splits, 1 = logarithmic splits
    float depthBias = 1.25f;   // constant depth bias applied in the shadow pass
    float normalBias = 0.03f;  // world-space offset along the surface normal when sampling
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

} // namespace mukki
