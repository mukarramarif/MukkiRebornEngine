#include "Application/Camera.h"

#include <cmath>

#include <gtest/gtest.h>

namespace
{

constexpr float kEpsilon = 1.0e-4f;

TEST(Camera, DefaultLooksDownNegativeZ)
{
    const mukki::Camera camera{};
    const glm::vec3 forward = camera.GetForward();

    EXPECT_NEAR(forward.x, 0.0f, kEpsilon);
    EXPECT_NEAR(forward.y, 0.0f, kEpsilon);
    EXPECT_NEAR(forward.z, -1.0f, kEpsilon);
}

TEST(Camera, YawRotatesForward)
{
    mukki::Camera camera{};
    camera.SetRotation(1.5707963268f, 0.0f);
    const glm::vec3 forward = camera.GetForward();

    EXPECT_NEAR(forward.x, 1.0f, kEpsilon);
    EXPECT_NEAR(forward.y, 0.0f, kEpsilon);
    EXPECT_NEAR(forward.z, 0.0f, kEpsilon);
}

TEST(Camera, PitchIsClamped)
{
    mukki::Camera camera{};
    camera.SetRotation(0.0f, 10.0f);

    EXPECT_LT(std::abs(camera.GetPitch()), 1.5707963268f);
}

TEST(Camera, ViewProjectionIsProjectionTimesView)
{
    mukki::Camera camera{};
    camera.SetPerspective(1.0471975512f, 16.0f / 9.0f, 0.1f, 100.0f);
    camera.SetPosition(glm::vec3{1.0f, 2.0f, 3.0f});
    camera.SetRotation(0.5f, -0.25f);

    const glm::mat4 expected = camera.GetProjectionMatrix() * camera.GetViewMatrix();
    EXPECT_EQ(camera.GetViewProjectionMatrix(), expected);
}

TEST(FlyCameraController, MovesAlongForward)
{
    mukki::Camera camera{};
    mukki::FlyCameraController controller{};
    mukki::CameraInput input{};
    input.move.y = 1.0f;

    controller.Update(camera, input, 1.0f);

    const glm::vec3 position = camera.GetPosition();
    EXPECT_NEAR(position.x, 0.0f, kEpsilon);
    EXPECT_NEAR(position.y, 0.0f, kEpsilon);
    EXPECT_NEAR(position.z, -controller.GetSettings().moveSpeed, kEpsilon);
}

TEST(FlyCameraController, FastMultiplierApplies)
{
    mukki::Camera camera{};
    mukki::FlyCameraController controller{};
    mukki::CameraInput input{};
    input.move.y = 1.0f;
    input.fast = true;

    controller.Update(camera, input, 1.0f);

    const auto& settings = controller.GetSettings();
    EXPECT_NEAR(camera.GetPosition().z, -settings.moveSpeed * settings.fastMultiplier, kEpsilon);
}

TEST(FlyCameraController, MouseLookRotatesCamera)
{
    mukki::Camera camera{};
    mukki::FlyCameraController controller{};
    mukki::CameraInput input{};
    input.lookEnabled = true;
    input.look = glm::vec2{100.0f, 0.0f};

    controller.Update(camera, input, 1.0f);
    input.look = glm::vec2{0.0f, 0.0f};

    EXPECT_GT(camera.GetYaw(), 0.0f);

    input.move.y = 1.0f;
    controller.Update(camera, input, 1.0f);
    EXPECT_GT(camera.GetPosition().x, 0.0f);
}

} // namespace
