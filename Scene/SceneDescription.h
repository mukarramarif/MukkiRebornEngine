#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "Renderer/Clouds/CloudSettings.h"
#include "Renderer/Lighting/Light.h"

namespace mukki
{

// Authored physics collider for a scene object. Body "none" (the default) leaves the object
// render-only. Zero halfExtents/radius/height mean "fit to the loaded model bounds".
enum class ScenePhysicsBody
{
    None,
    Static,
    Dynamic,
};

enum class ScenePhysicsShape
{
    Box,
    Sphere,
    Capsule,
};

struct ScenePhysics
{
    ScenePhysicsBody body = ScenePhysicsBody::None;
    ScenePhysicsShape shape = ScenePhysicsShape::Box;
    glm::vec3 halfExtents{0.0f};
    float radius = 0.0f; // zero derives from the fitted bounds
    float height = 0.0f; // zero derives from the fitted bounds (capsule total height)
    glm::vec3 offset{0.0f};
    float density = 1000.0f;
    float friction = 0.5f;
    float restitution = 0.0f;
};

// Scene wide physics defaults authored in the scene root.
struct ScenePhysicsSettings
{
    glm::vec3 gravity{0.0f, -9.81f, 0.0f};
    float stepHz = 60.0f;
    std::uint32_t subSteps = 4;
};

// Optional environment override (image based lighting). An absent block leaves the renderer's
// current environment (command line or UI) untouched.
struct SceneEnvironment
{
    std::filesystem::path map{}; // empty keeps the gradient sky / current map
    float intensity = 1.0f;
};

struct SceneObject
{
    std::string name{};
    std::string modelPath{};              // absolute or relative to SceneDescription::assetRoot
    glm::vec3 position{0.0f};             // world units
    glm::vec3 rotation{0.0f};             // degrees around X, Y and Z, applied in YXZ order
    glm::vec3 scale{1.0f};
    ScenePhysics physics{};
};

struct SceneLight
{
    std::string name{};
    Light light{};
};

// World space GI probe. Probes are authored independently of the glTF objects and feed the
// irradiance probe pass; radius only controls the debug marker size.
struct SceneProbe
{
    std::string name{};
    glm::vec3 position{0.0f}; // world units
    float radius = 0.5f;
    bool enabled = true;
};

// Authoring format shared by the JSON loader and the compiled scene binary. The assets referenced
// by SceneObject::modelPath live under assetRoot.
struct SceneDescription
{
    std::string name{};
    std::filesystem::path assetRoot{};
    std::vector<SceneObject> objects{};
    std::vector<SceneLight> lights{};
    std::vector<SceneProbe> probes{};
    ScenePhysicsSettings physics{};
    bool hasEnvironment = false;
    SceneEnvironment environment{};
    bool hasClouds = false;
    CloudSettings clouds{};
    std::string playerPlane{}; // scene object that enters flight mode (empty = none)
};

[[nodiscard]] glm::mat4 BuildObjectTransform(const SceneObject& object);

// Quaternion for SceneObject::rotation using the same YXZ composition as BuildObjectTransform.
[[nodiscard]] glm::quat BuildObjectRotation(const SceneObject& object);

// Inverse of BuildObjectRotation; returns X/Y/Z Euler angles in degrees.
[[nodiscard]] glm::vec3 ExtractObjectRotationEuler(const glm::quat& rotation);

[[nodiscard]] std::filesystem::path ResolveModelPath(
    const SceneDescription& scene,
    const SceneObject& object);

} // namespace mukki
