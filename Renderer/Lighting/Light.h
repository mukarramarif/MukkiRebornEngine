#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace mukki
{

enum class LightType : std::uint32_t
{
    Directional = 0,
    Point = 1,
    Spot = 2,
};

struct Light
{
    LightType type = LightType::Point;
    glm::vec3 position{0.0f};
    glm::vec3 direction{0.0f, -1.0f, 0.0f};
    glm::vec3 color{1.0f};
    float intensity = 1.0f;
    float range = 10.0f;
    float innerConeAngle = 0.0f;
    float outerConeAngle = 0.7853981634f;
    bool castShadows = true; // only used by the first directional light with shadows enabled
};

// GPU layout shared with the shaders. Every member is a float4 so the std430 offsets match the
// Slang struct without any padding surprises.
struct GpuLight
{
    glm::vec4 positionRange{0.0f};
    glm::vec4 directionIntensity{0.0f};
    glm::vec4 color{1.0f};
    glm::vec4 params{0.0f}; // x: type, y: inner cone cosine, z: outer cone cosine
};

[[nodiscard]] GpuLight ToGpuLight(const Light& light);

// Punctual lights (point and spot) packed for the ray traced GI light buffer. `GpuLight::params.w`
// carries the cast-shadows flag because this buffer is separate from the clustered light buffer.
[[nodiscard]] std::vector<GpuLight> SelectGiLights(
    std::span<const Light> lights,
    std::uint32_t maxCount);

} // namespace mukki
