#include "Application/Camera.h"

#include "Utilities/Assert.h"

#include <algorithm>
#include <cmath>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>

namespace mukki
{
namespace
{

constexpr glm::vec3 kWorldUp{0.0f, 1.0f, 0.0f};

} // namespace

void Camera::SetPerspective(float fovYRadians, float aspectRatio, float nearPlane, float farPlane)
{
    MRE_ASSERT(fovYRadians > 0.0f && fovYRadians < 3.1415926536f, "Camera fov must be in (0, pi)");
    MRE_ASSERT(aspectRatio > 0.0f, "Camera aspect ratio must be positive");
    MRE_ASSERT(nearPlane > 0.0f, "Camera near plane must be positive");
    MRE_ASSERT(farPlane > nearPlane, "Camera far plane must be greater than the near plane");

    m_fovY = fovYRadians;
    m_aspectRatio = aspectRatio;
    m_nearPlane = nearPlane;
    m_farPlane = farPlane;

    RecalculateProjection();
}

void Camera::SetPosition(const glm::vec3& position)
{
    m_position = position;
    RecalculateView();
}

void Camera::SetRotation(float yawRadians, float pitchRadians)
{
    m_yaw = yawRadians;
    m_pitch = std::clamp(pitchRadians, -1.5533431f, 1.5533431f);
    RecalculateView();
}

glm::vec3 Camera::GetForward() const
{
    const float cosPitch = std::cos(m_pitch);
    return glm::vec3{
        cosPitch * std::sin(m_yaw),
        std::sin(m_pitch),
        -cosPitch * std::cos(m_yaw),
    };
}

glm::vec3 Camera::GetRight() const
{
    return glm::normalize(glm::cross(GetForward(), kWorldUp));
}

glm::vec3 Camera::GetUp() const
{
    return glm::normalize(glm::cross(GetRight(), GetForward()));
}

void Camera::RecalculateView()
{
    m_view = glm::lookAt(m_position, m_position + GetForward(), kWorldUp);
    m_viewProjection = m_projection * m_view;
}

void Camera::RecalculateProjection()
{
    m_projection = glm::perspective(m_fovY, m_aspectRatio, m_nearPlane, m_farPlane);
    m_viewProjection = m_projection * m_view;
}

void FlyCameraController::Update(Camera& camera, const CameraInput& input, float deltaTime)
{
    MRE_ASSERT(deltaTime >= 0.0f, "FlyCameraController::Update requires a non-negative delta time");

    if (input.lookEnabled)
    {
        m_yaw += input.look.x * m_settings.mouseSensitivity;
        m_pitch -= input.look.y * m_settings.mouseSensitivity;
        m_pitch = std::clamp(m_pitch, -m_settings.maxPitch, m_settings.maxPitch);
        camera.SetRotation(m_yaw, m_pitch);
    }

    glm::vec3 direction = camera.GetForward() * input.move.y + camera.GetRight() * input.move.x;
    direction += kWorldUp * input.vertical;

    if (glm::dot(direction, direction) > 0.0f)
    {
        direction = glm::normalize(direction);
        const float speed = m_settings.moveSpeed * (input.fast ? m_settings.fastMultiplier : 1.0f);
        camera.SetPosition(camera.GetPosition() + direction * speed * deltaTime);
    }
}

} // namespace mukki
