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

// Documents the NDC convention used by the screen-space reprojections: the projection flips y and
// the viewport has a positive height, so world-up maps to negative NDC y (top of the framebuffer,
// since Vulkan's framebuffer y grows downward). NDC -> uv must therefore be uv.y = ndc.y * 0.5 + 0.5.
TEST(Camera, WorldUpMapsToNegativeNdcY)
{
    mukki::Camera camera{};
    camera.SetPerspective(1.0471975512f, 1.0f, 0.1f, 100.0f);
    camera.SetPosition(glm::vec3{0.0f, 0.0f, 0.0f});
    camera.SetRotation(0.0f, 0.0f);

    const glm::vec4 above = camera.GetViewProjectionMatrix() * glm::vec4{0.0f, 10.0f, -10.0f, 1.0f};
    ASSERT_GT(above.w, 0.0f);
    EXPECT_LT(above.y / above.w, 0.0f);

    const glm::vec4 below = camera.GetViewProjectionMatrix() * glm::vec4{0.0f, -10.0f, -10.0f, 1.0f};
    ASSERT_GT(below.w, 0.0f);
    EXPECT_GT(below.y / below.w, 0.0f);
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

TEST(Camera, SetViewMapsFrameAndKeepsRoll)
{
    mukki::Camera camera{};
    camera.SetPerspective(1.0471975512f, 1.0f, 0.1f, 100.0f);

    const glm::vec3 position{3.0f, 4.0f, 5.0f};
    const glm::vec3 forward = glm::normalize(glm::vec3{0.0f, 0.3f, -1.0f});
    const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3{0.0f, 1.0f, 0.0f}));
    const glm::vec3 up = glm::normalize(glm::cross(right, forward));
    camera.SetView(position, forward, up);

    const glm::mat4 view = camera.GetViewMatrix();
    const glm::vec4 viewPosition = view * glm::vec4{position, 1.0f};
    EXPECT_NEAR(viewPosition.x, 0.0f, kEpsilon);
    EXPECT_NEAR(viewPosition.y, 0.0f, kEpsilon);
    EXPECT_NEAR(viewPosition.z, 0.0f, kEpsilon);

    // The forward point maps to negative Z and the up point to positive Y in view space.
    const glm::vec4 forwardPoint = view * glm::vec4{position + forward, 1.0f};
    EXPECT_NEAR(forwardPoint.x, 0.0f, kEpsilon);
    EXPECT_NEAR(forwardPoint.y, 0.0f, kEpsilon);
    EXPECT_NEAR(forwardPoint.z, -1.0f, kEpsilon);

    const glm::vec4 upPoint = view * glm::vec4{position + up, 1.0f};
    EXPECT_NEAR(upPoint.x, 0.0f, kEpsilon);
    EXPECT_NEAR(upPoint.y, 1.0f, kEpsilon);
    EXPECT_NEAR(upPoint.z, 0.0f, kEpsilon);

    // The Euler members stay in sync with the horizontal forward direction.
    const glm::vec3 derivedForward = camera.GetForward();
    EXPECT_NEAR(derivedForward.x, forward.x, kEpsilon);
    EXPECT_NEAR(derivedForward.y, forward.y, kEpsilon);
    EXPECT_NEAR(derivedForward.z, forward.z, kEpsilon);
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
