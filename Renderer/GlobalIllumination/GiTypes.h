#pragma once

#include <cstdint>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace mukki
{

// Maximum number of GI probes uploaded to the GPU.
inline constexpr std::uint32_t kMaxGiProbes = 64;

struct GiSettings
{
    bool enabled = true;
    float resolutionScale = 0.5f;      // half resolution tracing
    float intensity = 1.0f;
    float maxDistance = 40.0f;
    float feedbackIntensity = 1.0f;    // multi-bounce energy fed back from the previous frame
    float specularMinRoughness = 0.15f;
    float temporalAlphaMin = 0.01f;
    float phiDepth = 1.0f;
    float phiNormal = 32.0f;
    float phiLuminance = 4.0f;
    float phiHitDistance = 4.0f;
    std::uint32_t atrousIterations = 4;
    bool sunShadowRay = true;
    std::uint32_t sunShadowSamples = 4;      // rays traced per hit point for sun visibility
    float sunAngularRadiusDegrees = 0.5f;    // jitter cone; produces soft shadows
    std::uint32_t traceSamples = 2;          // diffuse/specular path pairs traced per pixel
    // Probe mode: probes marked in the scene accumulate ray traced irradiance and the lighting
    // pass blends them instead of running the per-pixel trace.
    bool useProbes = true;
    std::uint32_t probeRays = 64;
    float probeHysteresis = 0.9f; // fraction of the previous probe irradiance kept per update
};

// Ray traced glossy reflections. They reuse the GI trace resolution and the GI a-trous passes.
struct ReflectionSettings
{
    bool enabled = true;
    float intensity = 1.0f;
    float maxRoughness = 0.45f;    // surfaces above this roughness do not trace reflections
    float maxDistance = 60.0f;
    std::uint32_t samples = 2;     // GGX VNDF rays per pixel (reuses the GI temporal alpha)
};

// Matches Renderer/Shaders/gi_*.slang.
struct GiUniform
{
    glm::mat4 inverseViewProjection{1.0f};
    glm::mat4 previousViewProjection{1.0f};
    glm::mat4 viewProjection{1.0f};
    glm::vec4 screenSize{0.0f}; // x: width, y: height
    glm::vec4 halfSize{0.0f};   // x: width, y: height
    glm::vec4 cameraPosition{0.0f};
    glm::vec4 skyColor{0.0f};              // rgb: color, w: intensity
    glm::vec4 sunDirectionIntensity{0.0f}; // xyz: direction toward the light, w: intensity
    glm::vec4 sunColor{1.0f};
    glm::vec4 params{0.0f};  // x: maxDistance, y: frameCounter, z: specularMinRoughness, w: feedbackIntensity
    glm::vec4 params2{0.0f}; // x: temporalAlphaMin, y: phiDepth, z: phiNormal, w: phiLuminance
    glm::vec4 params3{0.0f}; // x: phiHitDistance, y: intensity, z: sunShadowRay, w: resetHistory
    glm::vec4 params4{0.0f}; // x: sunShadowSamples, y: sunAngularRadius (radians), z: traceSamples, w: rtSunMask
    glm::vec4 params5{0.0f}; // x: probeRays, y: probeHysteresis, z: probeCount, w: reflectionResetHistory
    glm::vec4 params6{0.0f}; // x: reflectionMaxDistance, y: reflectionMaxRoughness, z: reflectionSamples, w: reflectionIntensity
};

// CPU-side world space GI probe. The radius only controls the debug marker size.
struct GiProbe
{
    glm::vec3 position{0.0f};
    float radius = 0.5f;
    glm::vec4 color{1.0f, 0.3f, 0.9f, 1.0f};
};

// GPU irradiance probe. The SH L1 coefficients are stored premultiplied so the lighting pass
// evaluates the cosine-convolved irradiance over pi as `sh0 + shX * n.x + shY * n.y + shZ * n.z`.
struct GpuGiProbe
{
    glm::vec4 position{0.0f}; // xyz: world position
    glm::vec4 sh0{0.0f};      // L0 rgb
    glm::vec4 shX{0.0f};
    glm::vec4 shY{0.0f};
    glm::vec4 shZ{0.0f};
};

// Mirrors the lighting pass evaluation; irradiance over pi for the given surface normal.
[[nodiscard]] inline glm::vec3 EvaluateProbeIrradiance(
    const GpuGiProbe& probe,
    const glm::vec3& normal)
{
    return glm::vec3{probe.sh0} + glm::vec3{probe.shX} * normal.x +
           glm::vec3{probe.shY} * normal.y + glm::vec3{probe.shZ} * normal.z;
}

} // namespace mukki
