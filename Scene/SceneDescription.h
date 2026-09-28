#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "Renderer/Lighting/Light.h"

namespace mukki
{

struct SceneObject
{
    std::string name{};
    std::string modelPath{};              // absolute or relative to SceneDescription::assetRoot
    glm::vec3 position{0.0f};             // world units
    glm::vec3 rotation{0.0f};             // degrees around X, Y and Z, applied in YXZ order
    glm::vec3 scale{1.0f};
};

struct SceneLight
{
    std::string name{};
    Light light{};
};

// Authoring format shared by the JSON loader and the compiled scene binary. The assets referenced
// by SceneObject::modelPath live under assetRoot.
struct SceneDescription
{
    std::string name{};
    std::filesystem::path assetRoot{};
    std::vector<SceneObject> objects{};
    std::vector<SceneLight> lights{};
};

[[nodiscard]] glm::mat4 BuildObjectTransform(const SceneObject& object);

[[nodiscard]] std::filesystem::path ResolveModelPath(
    const SceneDescription& scene,
    const SceneObject& object);

} // namespace mukki
