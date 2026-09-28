#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace mukki
{

class Camera
{
public:
    void SetPerspective(float fovYRadians, float aspectRatio, float nearPlane, float farPlane);
    void SetPosition(const glm::vec3& position);
    void SetRotation(float yawRadians, float pitchRadians);

    [[nodiscard]] const glm::mat4& GetViewMatrix() const { return m_view; }
    [[nodiscard]] const glm::mat4& GetProjectionMatrix() const { return m_projection; }
    [[nodiscard]] const glm::mat4& GetViewProjectionMatrix() const { return m_viewProjection; }

    [[nodiscard]] const glm::vec3& GetPosition() const { return m_position; }
    [[nodiscard]] glm::vec3 GetForward() const;
    [[nodiscard]] glm::vec3 GetRight() const;
    [[nodiscard]] glm::vec3 GetUp() const;
    [[nodiscard]] float GetYaw() const { return m_yaw; }
    [[nodiscard]] float GetPitch() const { return m_pitch; }
    [[nodiscard]] float GetFovY() const { return m_fovY; }
    [[nodiscard]] float GetAspectRatio() const { return m_aspectRatio; }
    [[nodiscard]] float GetNearPlane() const { return m_nearPlane; }
    [[nodiscard]] float GetFarPlane() const { return m_farPlane; }

private:
    void RecalculateView();
    void RecalculateProjection();

    glm::vec3 m_position{0.0f, 0.0f, 0.0f};
    float m_yaw = 0.0f;
    float m_pitch = 0.0f;
    float m_fovY = 1.0471975512f;
    float m_aspectRatio = 16.0f / 9.0f;
    float m_nearPlane = 0.1f;
    float m_farPlane = 1000.0f;
    glm::mat4 m_view{1.0f};
    glm::mat4 m_projection{1.0f};
    glm::mat4 m_viewProjection{1.0f};
};

struct CameraInput
{
    glm::vec2 move{0.0f};
    glm::vec2 look{0.0f};
    float vertical = 0.0f;
    bool fast = false;
    bool lookEnabled = false;
};

class FlyCameraController
{
public:
    struct Settings
    {
        float moveSpeed = 5.0f;
        float fastMultiplier = 4.0f;
        float mouseSensitivity = 0.0025f;
        float maxPitch = 1.5533431f;
    };

    FlyCameraController() = default;
    explicit FlyCameraController(const Settings& settings) : m_settings(settings) {}

    void Update(Camera& camera, const CameraInput& input, float deltaTime);

    [[nodiscard]] const Settings& GetSettings() const { return m_settings; }
    Settings& GetSettings() { return m_settings; }

private:
    Settings m_settings{};
    float m_yaw = 0.0f;
    float m_pitch = 0.0f;
};

} // namespace mukki
