#pragma once

#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

namespace mukki
{

// Arcade flight input. Pitch/roll/yaw are stick deflections in [-1, 1]; throttle is the per second
// change of the throttle lever (the lever itself lives in the controller).
struct FlightInput
{
    float pitch = 0.0f;    // +1 nose up
    float roll = 0.0f;     // +1 roll right
    float yaw = 0.0f;      // +1 yaw right (rudder)
    float throttle = 0.0f; // +1 open, -1 close (per second)
    bool boost = false;
};

struct FlightSettings
{
    float minSpeed = 60.0f;              // m/s at throttle 0
    float maxSpeed = 280.0f;             // m/s at throttle 1
    float boostMultiplier = 1.3f;
    float throttleRate = 0.8f;           // throttle lever units per second
    float acceleration = 60.0f;          // m/s^2 toward the target speed
    float pitchRate = 1.35f;             // rad/s at full stick and full control authority
    float rollRate = 2.6f;               // rad/s
    float yawRate = 0.7f;                // rad/s
    float autoLevelGain = 0.5f;          // roll correction per second at full bank
    float turnCoupling = 1.0f;           // bank -> coordinated turn scale
    float gravityPitchCoupling = 7.0f;   // m/s^2 climb/descent speed change at full pitch
    float minAuthority = 0.25f;          // control authority at zero speed
    float referenceSpeed = 90.0f;        // speed at which authority reaches 1
};

// Arcade airplane model: body-local pitch/roll/yaw rates scaled by airspeed, a bank -> turn
// coupling so banking turns naturally, mild auto-level with no roll input and throttle driven
// target speed. Transforms are integrated on the caller's thread; the caller syncs the visual
// object and camera after Update.
class PlaneController
{
public:
    void Reset(
        const glm::vec3& position,
        const glm::quat& orientation,
        float speed = 120.0f,
        float throttle = 0.6f);

    void Update(const FlightInput& input, float deltaTime);

    [[nodiscard]] const glm::vec3& GetPosition() const { return m_position; }
    [[nodiscard]] const glm::quat& GetOrientation() const { return m_orientation; }
    [[nodiscard]] glm::vec3 GetForward() const;
    [[nodiscard]] glm::vec3 GetUp() const;
    [[nodiscard]] glm::vec3 GetRight() const;
    [[nodiscard]] float GetSpeed() const { return m_speed; }
    [[nodiscard]] float GetThrottle() const { return m_throttle; }
    [[nodiscard]] float GetBankRadians() const;

    [[nodiscard]] const FlightSettings& GetSettings() const { return m_settings; }
    FlightSettings& GetSettings() { return m_settings; }

private:
    FlightSettings m_settings{};
    glm::vec3 m_position{0.0f};
    glm::quat m_orientation{1.0f, 0.0f, 0.0f, 0.0f};
    float m_speed = 120.0f;
    float m_throttle = 0.6f;
};

} // namespace mukki
