#include <gtest/gtest.h>

#include <cmath>

#include <glm/gtc/quaternion.hpp>
#include <glm/trigonometric.hpp>

#include "Physics/PhysicsWorld.h"
#include "Renderer/Debug/DebugLines.h"

namespace
{

mukki::PhysicsSettings TestSettings()
{
    mukki::PhysicsSettings settings{};
    settings.gravity = glm::vec3{0.0f, -9.81f, 0.0f};
    settings.stepHz = 60.0f;
    settings.subSteps = 4;
    return settings;
}

std::uint32_t CreateGround(mukki::PhysicsWorld& world)
{
    mukki::BodyDesc ground{};
    ground.type = mukki::BodyType::Static;
    ground.position = glm::vec3{0.0f, -0.5f, 0.0f};
    ground.collider.shape = mukki::ColliderShapeType::Box;
    ground.collider.halfExtents = glm::vec3{10.0f, 0.5f, 10.0f};
    ground.collider.friction = 0.8f;
    return world.CreateBody(ground);
}

} // namespace

TEST(PhysicsWorld, DynamicSphereRestsOnGround)
{
    mukki::PhysicsWorld world{};
    ASSERT_TRUE(world.Init(TestSettings()));
    CreateGround(world);

    mukki::BodyDesc sphere{};
    sphere.type = mukki::BodyType::Dynamic;
    sphere.position = glm::vec3{0.0f, 3.0f, 0.0f};
    sphere.collider.shape = mukki::ColliderShapeType::Sphere;
    sphere.collider.radius = 0.5f;
    const std::uint32_t handle = world.CreateBody(sphere);
    ASSERT_NE(handle, 0u);

    for (int step = 0; step < 300; ++step)
    {
        world.Step(1.0f / 60.0f);
    }

    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(world.GetBodyTransform(handle, position, rotation));
    EXPECT_NEAR(position.y, 0.5f, 0.1f);
    EXPECT_NEAR(position.x, 0.0f, 0.05f);
    EXPECT_NEAR(position.z, 0.0f, 0.05f);
    EXPECT_EQ(world.GetBodyCount(), 2u);
}

TEST(PhysicsWorld, StaticBodiesIgnoreGravity)
{
    mukki::PhysicsWorld world{};
    ASSERT_TRUE(world.Init(TestSettings()));
    const std::uint32_t handle = CreateGround(world);

    for (int step = 0; step < 120; ++step)
    {
        world.Step(1.0f / 60.0f);
    }

    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(world.GetBodyTransform(handle, position, rotation));
    EXPECT_FLOAT_EQ(position.y, -0.5f);
}

TEST(PhysicsWorld, TeleportMovesBodyAndZerosVelocity)
{
    mukki::PhysicsWorld world{};
    ASSERT_TRUE(world.Init(TestSettings()));

    mukki::BodyDesc box{};
    box.type = mukki::BodyType::Dynamic;
    box.position = glm::vec3{0.0f, 5.0f, 0.0f};
    box.collider.shape = mukki::ColliderShapeType::Box;
    box.collider.halfExtents = glm::vec3{0.5f};
    const std::uint32_t handle = world.CreateBody(box);
    ASSERT_NE(handle, 0u);

    for (int step = 0; step < 30; ++step)
    {
        world.Step(1.0f / 60.0f);
    }

    world.TeleportBody(handle, glm::vec3{2.0f, 10.0f, -1.0f}, glm::quat{1.0f, 0.0f, 0.0f, 0.0f});

    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(world.GetBodyTransform(handle, position, rotation));
    EXPECT_NEAR(position.x, 2.0f, 1.0e-4f);
    EXPECT_NEAR(position.y, 10.0f, 1.0e-4f);
    EXPECT_NEAR(position.z, -1.0f, 1.0e-4f);

    world.Step(1.0f / 60.0f);
    ASSERT_TRUE(world.GetBodyTransform(handle, position, rotation));
    EXPECT_LT(position.y, 10.0f);
    EXPECT_NEAR(position.x, 2.0f, 1.0e-3f);
}

TEST(PhysicsWorld, RotationRoundTrips)
{
    mukki::PhysicsWorld world{};
    ASSERT_TRUE(world.Init(TestSettings()));

    const glm::quat rotation = glm::quat{glm::vec3{0.3f, 0.7f, -0.2f}};
    mukki::BodyDesc box{};
    box.type = mukki::BodyType::Static;
    box.position = glm::vec3{1.0f, 2.0f, 3.0f};
    box.rotation = rotation;
    box.collider.halfExtents = glm::vec3{1.0f, 2.0f, 3.0f};
    const std::uint32_t handle = world.CreateBody(box);
    ASSERT_NE(handle, 0u);

    glm::vec3 position{0.0f};
    glm::quat stored{1.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(world.GetBodyTransform(handle, position, stored));
    EXPECT_NEAR(position.x, 1.0f, 1.0e-4f);
    EXPECT_NEAR(position.y, 2.0f, 1.0e-4f);
    EXPECT_NEAR(position.z, 3.0f, 1.0e-4f);
    EXPECT_NEAR(std::abs(glm::dot(rotation, stored)), 1.0f, 1.0e-4f);
}

TEST(PhysicsWorld, DisabledSettingsSkipSimulation)
{
    mukki::PhysicsWorld world{};
    mukki::PhysicsSettings settings = TestSettings();
    settings.enabled = false;
    ASSERT_TRUE(world.Init(settings));

    mukki::BodyDesc box{};
    box.type = mukki::BodyType::Dynamic;
    box.position = glm::vec3{0.0f, 5.0f, 0.0f};
    const std::uint32_t handle = world.CreateBody(box);

    for (int step = 0; step < 120; ++step)
    {
        world.Step(1.0f / 60.0f);
    }

    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    ASSERT_TRUE(world.GetBodyTransform(handle, position, rotation));
    EXPECT_FLOAT_EQ(position.y, 5.0f);
}

TEST(PhysicsWorld, DestroyAndClearBodies)
{
    mukki::PhysicsWorld world{};
    ASSERT_TRUE(world.Init(TestSettings()));
    const std::uint32_t ground = CreateGround(world);
    const std::uint32_t box = world.CreateBody(mukki::BodyDesc{});
    ASSERT_EQ(world.GetBodyCount(), 2u);

    world.DestroyBody(ground);
    EXPECT_EQ(world.GetBodyCount(), 1u);

    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_FALSE(world.GetBodyTransform(ground, position, rotation));

    world.ClearBodies();
    EXPECT_EQ(world.GetBodyCount(), 0u);
    EXPECT_FALSE(world.GetBodyTransform(box, position, rotation));
}

TEST(PhysicsWorld, BuildsColliderDebugLines)
{
    mukki::PhysicsWorld world{};
    ASSERT_TRUE(world.Init(TestSettings()));
    CreateGround(world);

    mukki::BodyDesc sphere{};
    sphere.type = mukki::BodyType::Dynamic;
    sphere.position = glm::vec3{0.0f, 3.0f, 0.0f};
    sphere.collider.shape = mukki::ColliderShapeType::Sphere;
    world.CreateBody(sphere);

    mukki::BodyDesc capsule{};
    capsule.type = mukki::BodyType::Dynamic;
    capsule.position = glm::vec3{1.0f, 3.0f, 0.0f};
    capsule.collider.shape = mukki::ColliderShapeType::Capsule;
    capsule.collider.radius = 0.3f;
    capsule.collider.height = 1.4f;
    world.CreateBody(capsule);

    mukki::DebugLines lines{};
    world.BuildDebugLines(lines);
    EXPECT_GT(lines.VertexCount(), 0u);

    world.ClearBodies();
    mukki::DebugLines cleared{};
    world.BuildDebugLines(cleared);
    EXPECT_EQ(cleared.VertexCount(), 0u);
}

TEST(PhysicsWorld, UsesNoSharedStateAcrossWorlds)
{
    mukki::PhysicsWorld first{};
    mukki::PhysicsWorld second{};
    ASSERT_TRUE(first.Init(TestSettings()));
    ASSERT_TRUE(second.Init(TestSettings()));
    EXPECT_EQ(first.GetBodyCount(), 0u);
    EXPECT_EQ(second.GetBodyCount(), 0u);

    const std::uint32_t handle = first.CreateBody(mukki::BodyDesc{});
    EXPECT_EQ(first.GetBodyCount(), 1u);
    EXPECT_EQ(second.GetBodyCount(), 0u);

    // Handles are per world; the second world must not resolve the first world's body.
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_FALSE(second.GetBodyTransform(handle, position, rotation));
    EXPECT_TRUE(first.GetBodyTransform(handle, position, rotation));
}
