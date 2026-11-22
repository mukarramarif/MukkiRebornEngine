#include "Scene/SceneDescription.h"

#include <cmath>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>

namespace mukki
{

glm::mat4 BuildObjectTransform(const SceneObject& object)
{
    glm::mat4 transform = glm::translate(glm::mat4{1.0f}, object.position);
    transform = glm::rotate(transform, glm::radians(object.rotation.y), glm::vec3{0.0f, 1.0f, 0.0f});
    transform = glm::rotate(transform, glm::radians(object.rotation.x), glm::vec3{1.0f, 0.0f, 0.0f});
    transform = glm::rotate(transform, glm::radians(object.rotation.z), glm::vec3{0.0f, 0.0f, 1.0f});
    transform = glm::scale(transform, object.scale);
    return transform;
}

glm::quat BuildObjectRotation(const SceneObject& object)
{
    const glm::quat yaw =
        glm::angleAxis(glm::radians(object.rotation.y), glm::vec3{0.0f, 1.0f, 0.0f});
    const glm::quat pitch =
        glm::angleAxis(glm::radians(object.rotation.x), glm::vec3{1.0f, 0.0f, 0.0f});
    const glm::quat roll =
        glm::angleAxis(glm::radians(object.rotation.z), glm::vec3{0.0f, 0.0f, 1.0f});
    return yaw * pitch * roll;
}

glm::vec3 ExtractObjectRotationEuler(const glm::quat& rotation)
{
    const glm::mat4 matrix = glm::mat4_cast(rotation);
    const float yaw = std::atan2(matrix[2][0], matrix[2][2]);
    const float cosPitch = std::sqrt(matrix[0][1] * matrix[0][1] + matrix[1][1] * matrix[1][1]);
    const float pitch = std::atan2(-matrix[2][1], cosPitch);
    const float sinYaw = std::sin(yaw);
    const float cosYaw = std::cos(yaw);
    const float roll = std::atan2(
        sinYaw * matrix[1][2] - cosYaw * matrix[1][0],
        cosYaw * matrix[0][0] - sinYaw * matrix[0][2]);
    return glm::degrees(glm::vec3{pitch, yaw, roll});
}

std::filesystem::path ResolveModelPath(const SceneDescription& scene, const SceneObject& object)
{
    const std::filesystem::path modelPath{object.modelPath};
    if (modelPath.is_absolute() || scene.assetRoot.empty())
    {
        return modelPath;
    }
    return scene.assetRoot / modelPath;
}

} // namespace mukki
