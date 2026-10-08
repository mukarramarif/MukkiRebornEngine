#pragma once

#include <cstdint>

#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

namespace mukki
{

// Sentinel for collision bodies that are not tied to a SceneDescription object (for example the
// ground collider).
inline constexpr std::uint32_t kNoSceneObject = 0xFFFFFFFFu;

enum class BodyType
{
    Static,
    Kinematic,
    Dynamic,
};

enum class ColliderShapeType
{
    Box,
    Sphere,
    Capsule,
};

// Local space collider description. Shapes are centred on the body origin plus offset.
struct ColliderDesc
{
    ColliderShapeType shape = ColliderShapeType::Box;
    glm::vec3 halfExtents{0.5f}; // box half extents along the local axes
    float radius = 0.5f;         // sphere radius, capsule cap radius
    float height = 1.0f;         // capsule total height including the hemisphere caps
    glm::vec3 offset{0.0f};      // local centre offset
    float density = 1000.0f;     // kg/m^3, only used by dynamic/kinematic bodies
    float friction = 0.5f;
    float restitution = 0.0f;
};

struct BodyDesc
{
    BodyType type = BodyType::Static;
    std::uint32_t objectIndex = kNoSceneObject;
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    ColliderDesc collider{};
};

// Read-only view of a live collision body.
struct CollisionBody
{
    std::uint32_t handle = 0;
    std::uint32_t objectIndex = kNoSceneObject;
    BodyType type = BodyType::Static;
    ColliderDesc collider{};
};

struct PhysicsSettings
{
    bool enabled = true;
    glm::vec3 gravity{0.0f, -9.81f, 0.0f};
    float stepHz = 60.0f;
    std::uint32_t subSteps = 4;
    float maxAccumulatedTime = 0.25f; // seconds of backlog processed per frame
};

} // namespace mukki
