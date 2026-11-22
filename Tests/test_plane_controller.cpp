#include "Application/PlaneController.h"

#include <cmath>

#include <gtest/gtest.h>
#include <glm/trigonometric.hpp>

namespace
{

constexpr float kEpsilon = 1.0e-3f;

mukki::PlaneController MakeController()
{
    mukki::PlaneController controller{};
    controller.Reset(glm::vec3{0.0f}, glm::quat{1.0f, 0.0f, 0.0f, 0.0f}, 120.0f, 0.6f);
    return controller;
}

void Step(mukki::PlaneController& controller, const mukki::FlightInput& input, float seconds)
{
    constexpr float kStep = 1.0f / 120.0f;
    for (float elapsed = 0.0f; elapsed < seconds; elapsed += kStep)
    {
        controller.Update(input, kStep);
    }
}

} // namespace

TEST(PlaneController, FliesStraightAlongForward)
{
    mukki::PlaneController controller = MakeController();
    mukki::FlightInput input{};
    Step(controller, input, 1.0f);

    const glm::vec3 position = controller.GetPosition();
    EXPECT_NEAR(controller.GetSpeed(), 139.4f, 5.0f);
    EXPECT_NEAR(position.x, 0.0f, kEpsilon);
    EXPECT_NEAR(position.z, -130.0f, 10.0f);
    EXPECT_NEAR(controller.GetForward().z, -1.0f, kEpsilon);
    EXPECT_NEAR(controller.GetRight().y, 0.0f, kEpsilon);
}

TEST(PlaneController, PitchUpClimbs)
{
    mukki::PlaneController controller = MakeController();
    mukki::FlightInput input{};
    input.pitch = 1.0f;

    Step(controller, input, 0.5f);

    EXPECT_GT(controller.GetForward().y, 0.2f);
    EXPECT_GT(controller.GetPosition().y, 0.0f);
}

TEST(PlaneController, PitchDownDescendsAndSpeedsUp)
{
    mukki::PlaneController controller = MakeController();
    const float initialSpeed = controller.GetSpeed();
    mukki::FlightInput input{};
    input.pitch = -1.0f;
    input.throttle = 0.0f;

    Step(controller, input, 1.0f);

    EXPECT_LT(controller.GetPosition().y, 0.0f);
    EXPECT_GT(controller.GetSpeed(), initialSpeed);
}

TEST(PlaneController, RollBanksThenAutoLevels)
{
    mukki::PlaneController controller = MakeController();
    mukki::FlightInput roll{};
    roll.roll = 1.0f;
    Step(controller, roll, 0.25f);
    EXPECT_LT(controller.GetRight().y, -0.1f);

    mukki::FlightInput level{};
    Step(controller, level, 3.0f);
    EXPECT_NEAR(controller.GetRight().y, 0.0f, 0.03f);
    EXPECT_NEAR(controller.GetForward().y, 0.0f, 0.05f);
}

TEST(PlaneController, BankedTurnChangesHeading)
{
    // Start banked to the right; with no stick input the auto-level reduces the bank but the
    // coordinated turn coupling still yaws the nose toward +X before the plane is level.
    mukki::PlaneController controller{};
    controller.Reset(
        glm::vec3{0.0f},
        glm::angleAxis(1.0f, glm::vec3{0.0f, 0.0f, -1.0f}),
        160.0f,
        0.6f);
    ASSERT_LT(controller.GetRight().y, -0.1f);

    const glm::vec3 forwardBefore = controller.GetForward();
    mukki::FlightInput input{};
    Step(controller, input, 0.35f);

    EXPECT_GT(controller.GetForward().x, forwardBefore.x + 0.01f);
    EXPECT_GT(controller.GetPosition().x, 0.2f);
}

TEST(PlaneController, RudderYaws)
{
    mukki::PlaneController controller = MakeController();
    mukki::FlightInput input{};
    input.yaw = 1.0f;

    Step(controller, input, 0.5f);

    // Right rudder: the nose moves toward +X.
    EXPECT_GT(controller.GetForward().x, 0.1f);
}

TEST(PlaneController, ThrottleControlsSpeed)
{
    mukki::PlaneController slow = MakeController();
    mukki::FlightInput idle{};
    idle.throttle = -1.0f;
    Step(slow, idle, 3.0f);

    mukki::PlaneController fast = MakeController();
    mukki::FlightInput open{};
    open.throttle = 1.0f;
    Step(fast, open, 3.0f);

    EXPECT_LT(slow.GetSpeed(), fast.GetSpeed());
    EXPECT_NEAR(slow.GetThrottle(), 0.0f, kEpsilon);
    EXPECT_NEAR(fast.GetThrottle(), 1.0f, kEpsilon);
}

TEST(PlaneController, ResetRestoresState)
{
    mukki::PlaneController controller = MakeController();
    mukki::FlightInput input{};
    input.pitch = 1.0f;
    input.roll = 1.0f;
    Step(controller, input, 1.0f);

    const glm::vec3 startPosition{100.0f, 200.0f, -300.0f};
    const glm::quat startOrientation = glm::angleAxis(glm::radians(90.0f), glm::vec3{0.0f, 1.0f, 0.0f});
    controller.Reset(startPosition, startOrientation, 150.0f, 0.5f);

    EXPECT_NEAR(glm::length(controller.GetPosition() - startPosition), 0.0f, kEpsilon);
    EXPECT_NEAR(std::abs(glm::dot(controller.GetOrientation(), startOrientation)), 1.0f, kEpsilon);
    EXPECT_NEAR(controller.GetSpeed(), 150.0f, kEpsilon);
    EXPECT_NEAR(controller.GetThrottle(), 0.5f, kEpsilon);
    EXPECT_NEAR(controller.GetBankRadians(), 0.0f, kEpsilon);
}

TEST(PlaneController, BoostRaisesTargetSpeed)
{
    mukki::PlaneController controller = MakeController();
    mukki::FlightInput input{};
    input.throttle = 1.0f;
    input.boost = true;
    Step(controller, input, 12.0f);

    EXPECT_GT(controller.GetSpeed(), controller.GetSettings().maxSpeed + 1.0f);
}
