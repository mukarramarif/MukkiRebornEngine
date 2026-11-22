#include "Application/PlaneController.h"

#include <algorithm>
#include <cmath>

#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/trigonometric.hpp>

#include "Utilities/Assert.h"

namespace mukki
{
namespace
{

constexpr glm::vec3 kLocalForward{0.0f, 0.0f, -1.0f};
constexpr glm::vec3 kLocalUp{0.0f, 1.0f, 0.0f};
constexpr glm::vec3 kLocalRight{1.0f, 0.0f, 0.0f};

} // namespace

void PlaneController::Reset(
    const glm::vec3& position,
    const glm::quat& orientation,
    float speed,
    float throttle)
{
    m_position = position;
    m_orientation = glm::normalize(orientation);
    m_speed = std::clamp(speed, m_settings.minSpeed, m_settings.maxSpeed);
    m_throttle = std::clamp(throttle, 0.0f, 1.0f);
}

void PlaneController::Update(const FlightInput& input, float deltaTime)
{
    MRE_ASSERT(deltaTime >= 0.0f, "PlaneController::Update requires a non-negative delta time");
    if (deltaTime <= 0.0f)
    {
        return;
    }

    const float authority = std::clamp(
        m_speed / std::max(m_settings.referenceSpeed, 1.0f),
        m_settings.minAuthority,
        1.0f);

    // Auto-level replaces the roll stick when the pilot is not holding it: the bank angle is
    // corrected back toward level.
    float rollInput = std::clamp(input.roll, -1.0f, 1.0f);
    const float bank = GetBankRadians();
    if (std::abs(rollInput) < 0.05f)
    {
        rollInput = std::clamp(-bank * m_settings.autoLevelGain, -1.0f, 1.0f);
    }

    // Body local rotation: roll about the local forward, pitch about the right, rudder about up.
    const glm::quat roll = glm::angleAxis(rollInput * m_settings.rollRate * authority * deltaTime, kLocalForward);
    const glm::quat pitch =
        glm::angleAxis(std::clamp(input.pitch, -1.0f, 1.0f) * m_settings.pitchRate * authority * deltaTime, kLocalRight);
    const glm::quat rudder =
        glm::angleAxis(-std::clamp(input.yaw, -1.0f, 1.0f) * m_settings.yawRate * authority * deltaTime, kLocalUp);
    m_orientation = glm::normalize(m_orientation * roll * pitch * rudder);

    // Coordinated turn: a banked airplane yaws toward the bank (omega = g * tan(bank) / v).
    const float bankClamped = std::clamp(bank, -1.3f, 1.3f);
    const float turnRate = 9.81f * std::tan(bankClamped) / std::max(m_speed, 20.0f);
    m_orientation =
        glm::normalize(m_orientation * glm::angleAxis(-turnRate * m_settings.turnCoupling * deltaTime, kLocalUp));

    // Throttle lever and speed: approach the target airspeed, then trade speed for altitude.
    m_throttle = std::clamp(m_throttle + input.throttle * m_settings.throttleRate * deltaTime, 0.0f, 1.0f);
    const float targetSpeed = glm::mix(m_settings.minSpeed, m_settings.maxSpeed, m_throttle) *
                              (input.boost ? m_settings.boostMultiplier : 1.0f);
    const float approach = std::clamp(m_settings.acceleration * deltaTime / std::max(targetSpeed, 1.0f), 0.0f, 1.0f);
    m_speed += (targetSpeed - m_speed) * approach;
    m_speed -= GetForward().y * m_settings.gravityPitchCoupling * deltaTime;
    m_speed = std::clamp(m_speed, m_settings.minSpeed * 0.4f, m_settings.maxSpeed * m_settings.boostMultiplier);

    m_position += GetForward() * m_speed * deltaTime;
}

glm::vec3 PlaneController::GetForward() const
{
    return m_orientation * kLocalForward;
}

glm::vec3 PlaneController::GetUp() const
{
    return m_orientation * kLocalUp;
}

glm::vec3 PlaneController::GetRight() const
{
    return m_orientation * kLocalRight;
}

float PlaneController::GetBankRadians() const
{
    // Positive when rolled right (the right wing drops below the horizon).
    return std::asin(std::clamp(-GetRight().y, -1.0f, 1.0f));
}

} // namespace mukki
