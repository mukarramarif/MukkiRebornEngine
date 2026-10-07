#pragma once

#include <cstdint>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

namespace mukki
{

// Volumetric cloud settings (Nubis/Decima style). The noise and weather textures are generated
// in-engine; the sliders below only drive the uniforms.
struct CloudSettings
{
    bool enabled = true;
    float coverage = 0.55f;
    // 0 = stratus, 0.33 = cumulus, 0.66 = cumulonimbus, 1 = cirrus.
    float cloudType = 0.33f;
    float density = 0.025f;          // extinction per metre at full density
    float detailStrength = 0.35f;    // high-frequency erosion amount
    float layerBottomKm = 1.4f;
    float layerTopKm = 4.2f;
    float altitudeVariationKm = 0.5f;
    float windSpeed = 18.0f;         // m/s
    float windDirectionRadians = 0.7f;
    float curlStrength = 1.0f;
    float coverageVariation = 0.5f;  // regional variation around the coverage slider
    std::uint32_t primarySteps = 64;
    std::uint32_t lightSteps = 6;
    float resolutionScale = 0.5f;    // cloud trace scale relative to the swapchain
    float temporalBlend = 0.9f;      // history weight (Phase 4)
    float godRayIntensity = 0.5f;
};

// Matches Renderer/Shaders/clouds.comp.slang / cloud_reproject.comp.slang /
// cloud_upsample.comp.slang.
struct CloudUniform
{
    glm::mat4 inverseViewProjection{1.0f};
    glm::mat4 previousViewProjection{1.0f};
    glm::mat4 viewProjection{1.0f};
    glm::vec4 cameraPosition{0.0f}; // xyz: camera, w: unused
    glm::vec4 screenSize{0.0f};     // x/y: full res, z/w: cloud trace res
    glm::vec4 sunDirectionIntensity{0.0f}; // xyz: direction toward the sun, w: intensity
    glm::vec4 sunColor{1.0f};              // rgb: color * intensity
    glm::vec4 cloudParams{0.0f};  // x: coverage, y: cloud type, z: density, w: detail strength
    glm::vec4 cloudParams2{0.0f}; // x: bottom km, y: top km, z: altitude variation km, w: curl
    glm::vec4 cloudParams3{0.0f}; // x: wind direction radians, y: wind speed, z: time seconds
    glm::vec4 skySh[9]{};         // environment irradiance / pi for ambient cloud lighting
    glm::vec4 skyParams{0.0f};    // x: ambient scale, y: coverage variation, z/w: unused
    glm::vec4 noiseScales{0.0f};  // x: base, y: detail, z: curl, w: weather (1 / metres)
    glm::vec4 temporalParams{0.0f}; // x: history blend, y: reset flag, z/w: unused
    glm::vec4 cloudParams4{0.0f};   // x: primary steps, y: light steps, z: frame, w: unused
};

static_assert(sizeof(CloudUniform) % 16 == 0, "CloudUniform must stay 16-byte aligned");

} // namespace mukki
