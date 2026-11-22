#include "Physics/PhysicsWorld.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <box3d/box3d.h>

#include "Renderer/Debug/DebugLines.h"
#include "Utilities/Assert.h"
#include "Utilities/Log.h"
#include "Utilities/Profiling.h"

namespace mukki
{
namespace
{

int PhysicsAssert(const char* condition, const char* fileName, int lineNumber)
{
    MRE_ERROR(
        "Box3D assert failed: {} ({}:{})",
        condition != nullptr ? condition : "",
        fileName != nullptr ? fileName : "",
        lineNumber);
    return 1; // non-zero breaks into the debugger
}

b3Vec3 ToB3(const glm::vec3& value)
{
    return b3Vec3{value.x, value.y, value.z};
}

glm::vec3 FromB3(const b3Vec3& value)
{
    return glm::vec3{value.x, value.y, value.z};
}

b3Quat ToB3(const glm::quat& value)
{
    return b3Quat{{value.x, value.y, value.z}, value.w};
}

glm::quat FromB3(const b3Quat& value)
{
    return glm::quat{value.s, value.v.x, value.v.y, value.v.z};
}

b3BodyType ToB3(BodyType type)
{
    switch (type)
    {
    case BodyType::Static:
        return b3_staticBody;
    case BodyType::Kinematic:
        return b3_kinematicBody;
    case BodyType::Dynamic:
        return b3_dynamicBody;
    }
    return b3_staticBody;
}

constexpr float kMinStepHz = 15.0f;
constexpr float kMaxStepHz = 240.0f;
constexpr std::uint32_t kMaxSubSteps = 8;

} // namespace

struct PhysicsWorld::Impl
{
    PhysicsSettings settings{};
    b3WorldId worldId{};
    bool initialized = false;
    float accumulator = 0.0f;
    std::vector<CollisionBody> bodies{}; // handle == 0 marks a free slot
    std::vector<b3BodyId> bodyIds{};
};

PhysicsWorld::PhysicsWorld() : m_impl(std::make_unique<Impl>())
{
}

PhysicsWorld::~PhysicsWorld()
{
    Shutdown();
}

bool PhysicsWorld::Init(const PhysicsSettings& settings)
{
    if (m_impl->initialized)
    {
        return true;
    }

    b3SetAssertFcn(&PhysicsAssert);

    b3WorldDef worldDef = b3DefaultWorldDef();
    worldDef.gravity = ToB3(settings.gravity);
    worldDef.workerCount = 1; // single threaded: the game thread owns the world
    worldDef.enableSleep = true;

    m_impl->worldId = b3CreateWorld(&worldDef);
    if (!b3World_IsValid(m_impl->worldId))
    {
        MRE_ERROR("Failed to create the Box3D world");
        return false;
    }

    m_impl->settings = settings;
    m_impl->initialized = true;
    MRE_INFO("Physics world initialized (Box3D, {} Hz, {} sub-steps)", settings.stepHz, settings.subSteps);
    return true;
}

void PhysicsWorld::Shutdown()
{
    if (!m_impl->initialized)
    {
        return;
    }

    b3DestroyWorld(m_impl->worldId);
    m_impl->worldId = b3WorldId{};
    m_impl->bodies.clear();
    m_impl->bodyIds.clear();
    m_impl->accumulator = 0.0f;
    m_impl->initialized = false;
}

bool PhysicsWorld::IsInitialized() const
{
    return m_impl->initialized;
}

void PhysicsWorld::SetSettings(const PhysicsSettings& settings)
{
    m_impl->settings = settings;
    if (m_impl->initialized)
    {
        b3World_SetGravity(m_impl->worldId, ToB3(settings.gravity));
    }
}

const PhysicsSettings& PhysicsWorld::GetSettings() const
{
    return m_impl->settings;
}

std::uint32_t PhysicsWorld::CreateBody(const BodyDesc& desc)
{
    if (!m_impl->initialized)
    {
        MRE_ERROR("PhysicsWorld::CreateBody called before PhysicsWorld::Init");
        return 0;
    }
    MRE_ASSERT(desc.collider.halfExtents.x > 0.0f && desc.collider.halfExtents.y > 0.0f &&
                   desc.collider.halfExtents.z > 0.0f,
               "Box half extents must be positive");
    MRE_ASSERT(desc.collider.radius > 0.0f, "Sphere/capsule radius must be positive");

    b3BodyDef bodyDef = b3DefaultBodyDef();
    bodyDef.type = ToB3(desc.type);
    bodyDef.position = ToB3(desc.position);
    bodyDef.rotation = ToB3(desc.rotation);

    const b3BodyId bodyId = b3CreateBody(m_impl->worldId, &bodyDef);
    if (!b3Body_IsValid(bodyId))
    {
        MRE_ERROR("Failed to create a Box3D body");
        return 0;
    }

    b3ShapeDef shapeDef = b3DefaultShapeDef();
    shapeDef.density = desc.collider.density;
    shapeDef.baseMaterial.friction = desc.collider.friction;
    shapeDef.baseMaterial.restitution = desc.collider.restitution;

    switch (desc.collider.shape)
    {
    case ColliderShapeType::Box:
    {
        const b3BoxHull hull = b3MakeOffsetBoxHull(
            desc.collider.halfExtents.x,
            desc.collider.halfExtents.y,
            desc.collider.halfExtents.z,
            ToB3(desc.collider.offset));
        b3CreateHullShape(bodyId, &shapeDef, &hull.base);
        break;
    }
    case ColliderShapeType::Sphere:
    {
        b3Sphere sphere{};
        sphere.center = ToB3(desc.collider.offset);
        sphere.radius = desc.collider.radius;
        b3CreateSphereShape(bodyId, &shapeDef, &sphere);
        break;
    }
    case ColliderShapeType::Capsule:
    {
        const float halfSegment =
            std::max(desc.collider.height * 0.5f - desc.collider.radius, 0.0f);
        b3Capsule capsule{};
        capsule.center1 = ToB3(desc.collider.offset - glm::vec3{0.0f, halfSegment, 0.0f});
        capsule.center2 = ToB3(desc.collider.offset + glm::vec3{0.0f, halfSegment, 0.0f});
        capsule.radius = desc.collider.radius;
        b3CreateCapsuleShape(bodyId, &shapeDef, &capsule);
        break;
    }
    }

    std::size_t slot = m_impl->bodies.size();
    for (std::size_t index = 0; index < m_impl->bodies.size(); ++index)
    {
        if (m_impl->bodies[index].handle == 0)
        {
            slot = index;
            break;
        }
    }

    CollisionBody body{};
    body.handle = static_cast<std::uint32_t>(slot) + 1;
    body.objectIndex = desc.objectIndex;
    body.type = desc.type;
    body.collider = desc.collider;

    if (slot == m_impl->bodies.size())
    {
        m_impl->bodies.push_back(body);
        m_impl->bodyIds.push_back(bodyId);
    }
    else
    {
        m_impl->bodies[slot] = body;
        m_impl->bodyIds[slot] = bodyId;
    }
    return body.handle;
}

void PhysicsWorld::DestroyBody(std::uint32_t handle)
{
    if (!m_impl->initialized || handle == 0)
    {
        return;
    }

    const std::size_t slot = static_cast<std::size_t>(handle) - 1;
    if (slot >= m_impl->bodies.size() || m_impl->bodies[slot].handle != handle)
    {
        return;
    }

    b3DestroyBody(m_impl->bodyIds[slot]);
    m_impl->bodies[slot] = CollisionBody{};
    m_impl->bodyIds[slot] = b3BodyId{};
}

void PhysicsWorld::ClearBodies()
{
    if (!m_impl->initialized)
    {
        m_impl->bodies.clear();
        m_impl->bodyIds.clear();
        return;
    }

    for (std::size_t slot = 0; slot < m_impl->bodies.size(); ++slot)
    {
        if (m_impl->bodies[slot].handle != 0)
        {
            b3DestroyBody(m_impl->bodyIds[slot]);
        }
    }
    m_impl->bodies.clear();
    m_impl->bodyIds.clear();
    m_impl->accumulator = 0.0f;
}

std::size_t PhysicsWorld::GetBodyCount() const
{
    return static_cast<std::size_t>(std::count_if(
        m_impl->bodies.begin(),
        m_impl->bodies.end(),
        [](const CollisionBody& body) { return body.handle != 0; }));
}

void PhysicsWorld::Step(float deltaTime)
{
    MRE_PROFILE_ZONE_N("PhysicsWorld::Step");

    if (!m_impl->initialized || !m_impl->settings.enabled)
    {
        return;
    }

    const float stepHz = std::clamp(m_impl->settings.stepHz, kMinStepHz, kMaxStepHz);
    const float step = 1.0f / stepHz;
    const int subSteps = static_cast<int>(std::clamp(m_impl->settings.subSteps, 1u, kMaxSubSteps));
    const float maxAccumulated = std::max(m_impl->settings.maxAccumulatedTime, step);

    m_impl->accumulator = std::min(m_impl->accumulator + std::max(deltaTime, 0.0f), maxAccumulated);

    const int maxStepsPerFrame = std::max(1, static_cast<int>(std::ceil(maxAccumulated / step)));
    int steps = 0;
    while (m_impl->accumulator >= step && steps < maxStepsPerFrame)
    {
        b3World_Step(m_impl->worldId, step, subSteps);
        m_impl->accumulator -= step;
        ++steps;
    }
}

bool PhysicsWorld::GetBodyTransform(
    std::uint32_t handle,
    glm::vec3& outPosition,
    glm::quat& outRotation) const
{
    if (!m_impl->initialized || handle == 0)
    {
        return false;
    }

    const std::size_t slot = static_cast<std::size_t>(handle) - 1;
    if (slot >= m_impl->bodies.size() || m_impl->bodies[slot].handle != handle)
    {
        return false;
    }

    const b3WorldTransform transform = b3Body_GetTransform(m_impl->bodyIds[slot]);
    outPosition = FromB3(transform.p);
    outRotation = FromB3(transform.q);
    return true;
}

void PhysicsWorld::TeleportBody(
    std::uint32_t handle,
    const glm::vec3& position,
    const glm::quat& rotation)
{
    if (!m_impl->initialized || handle == 0)
    {
        return;
    }

    const std::size_t slot = static_cast<std::size_t>(handle) - 1;
    if (slot >= m_impl->bodies.size() || m_impl->bodies[slot].handle != handle)
    {
        return;
    }

    const b3BodyId bodyId = m_impl->bodyIds[slot];
    b3Body_SetTransform(bodyId, ToB3(position), ToB3(rotation));
    b3Body_SetLinearVelocity(bodyId, b3Vec3{0.0f, 0.0f, 0.0f});
    b3Body_SetAngularVelocity(bodyId, b3Vec3{0.0f, 0.0f, 0.0f});
    b3Body_SetAwake(bodyId, true);
}

std::span<const CollisionBody> PhysicsWorld::GetBodies() const
{
    return m_impl->bodies;
}

void PhysicsWorld::BuildDebugLines(DebugLines& lines) const
{
    const glm::vec4 staticColor{0.2f, 0.9f, 0.4f, 1.0f};
    const glm::vec4 kinematicColor{1.0f, 0.8f, 0.2f, 1.0f};
    const glm::vec4 dynamicColor{0.2f, 0.8f, 1.0f, 1.0f};

    for (const CollisionBody& body : m_impl->bodies)
    {
        if (body.handle == 0)
        {
            continue;
        }

        glm::vec3 position{0.0f};
        glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
        if (!GetBodyTransform(body.handle, position, rotation))
        {
            continue;
        }

        glm::vec4 color = staticColor;
        if (body.type == BodyType::Kinematic)
        {
            color = kinematicColor;
        }
        else if (body.type == BodyType::Dynamic)
        {
            color = dynamicColor;
        }

        const glm::mat3 basis = glm::mat3_cast(rotation);
        const glm::vec3 center = position + basis * body.collider.offset;

        switch (body.collider.shape)
        {
        case ColliderShapeType::Box:
        {
            glm::vec3 corners[8];
            for (int corner = 0; corner < 8; ++corner)
            {
                const glm::vec3 sign{
                    (corner & 1) != 0 ? 1.0f : -1.0f,
                    (corner & 2) != 0 ? 1.0f : -1.0f,
                    (corner & 4) != 0 ? 1.0f : -1.0f};
                corners[corner] = center + basis * (body.collider.halfExtents * sign);
            }
            static constexpr int kEdges[12][2] = {
                {0, 1}, {1, 3}, {3, 2}, {2, 0},
                {4, 5}, {5, 7}, {7, 6}, {6, 4},
                {0, 4}, {1, 5}, {2, 6}, {3, 7}};
            for (const auto& edge : kEdges)
            {
                lines.AddLine(corners[edge[0]], corners[edge[1]], color);
            }
            break;
        }
        case ColliderShapeType::Sphere:
            lines.AddSphere(center, body.collider.radius, color);
            break;
        case ColliderShapeType::Capsule:
        {
            const glm::vec3 axis = basis * glm::vec3{0.0f, 1.0f, 0.0f};
            const float halfSegment =
                std::max(body.collider.height * 0.5f - body.collider.radius, 0.0f);
            const glm::vec3 top = center + axis * halfSegment;
            const glm::vec3 bottom = center - axis * halfSegment;
            const glm::vec3 axisA = basis * glm::vec3{1.0f, 0.0f, 0.0f};
            const glm::vec3 axisB = basis * glm::vec3{0.0f, 0.0f, 1.0f};
            lines.AddCircle(top, axisA, axisB, body.collider.radius, color, 24);
            lines.AddCircle(bottom, axisA, axisB, body.collider.radius, color, 24);
            lines.AddLine(bottom + axisA * body.collider.radius, top + axisA * body.collider.radius, color);
            lines.AddLine(bottom - axisA * body.collider.radius, top - axisA * body.collider.radius, color);
            lines.AddLine(bottom + axisB * body.collider.radius, top + axisB * body.collider.radius, color);
            lines.AddLine(bottom - axisB * body.collider.radius, top - axisB * body.collider.radius, color);
            break;
        }
        }
    }
}

std::size_t PhysicsWorld::GetByteCount() const
{
    return static_cast<std::size_t>(std::max(b3GetByteCount(), 0));
}

} // namespace mukki
