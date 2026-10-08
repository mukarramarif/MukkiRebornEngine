#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

#include "Physics/PhysicsTypes.h"

namespace mukki
{

class DebugLines;

// Box3D-backed rigid body simulation. Runs single threaded on the game thread. The fixed step
// accumulator advances the world at PhysicsSettings::stepHz; callers read body transforms after
// Step and push them into the renderer (post-step transform sync).
class PhysicsWorld
{
public:
    PhysicsWorld();
    ~PhysicsWorld();

    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    [[nodiscard]] bool Init(const PhysicsSettings& settings);
    void Shutdown();
    [[nodiscard]] bool IsInitialized() const;

    void SetSettings(const PhysicsSettings& settings);
    [[nodiscard]] const PhysicsSettings& GetSettings() const;

    // Creates a body with one collider. Returns an opaque handle (0 when the world is not
    // initialized). Handles are reused after DestroyBody.
    [[nodiscard]] std::uint32_t CreateBody(const BodyDesc& desc);
    void DestroyBody(std::uint32_t handle);
    void ClearBodies();
    [[nodiscard]] std::size_t GetBodyCount() const;

    // Consumes deltaTime through the fixed step accumulator.
    void Step(float deltaTime);

    [[nodiscard]] bool GetBodyTransform(
        std::uint32_t handle,
        glm::vec3& outPosition,
        glm::quat& outRotation) const;
    void TeleportBody(std::uint32_t handle, const glm::vec3& position, const glm::quat& rotation);

    [[nodiscard]] std::span<const CollisionBody> GetBodies() const;

    // Appends wireframe collider markers (green static, yellow kinematic, cyan dynamic).
    void BuildDebugLines(DebugLines& lines) const;

    // Total bytes currently allocated by Box3D.
    [[nodiscard]] std::size_t GetByteCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl{};
};

} // namespace mukki
