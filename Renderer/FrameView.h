#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace mukki
{

// Camera state for one frame, shared by the lighting, clustering and shadow code.
struct FrameView
{
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::mat4 viewProjection{1.0f};
    glm::mat4 previousViewProjection{1.0f};
    glm::vec3 position{0.0f};
    float fovYRadians = 1.0471975512f;
    float aspectRatio = 16.0f / 9.0f;
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;
};

} // namespace mukki
