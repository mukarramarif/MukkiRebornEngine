#include "Scene/SceneDescription.h"

#include <glm/gtc/matrix_transform.hpp>

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
