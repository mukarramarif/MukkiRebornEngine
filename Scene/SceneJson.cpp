#include "Scene/SceneJson.h"

#include <fstream>
#include <sstream>
#include <utility>

#include <nlohmann/json.hpp>

#include "Utilities/PathUtils.h"

namespace mukki
{
namespace
{

using Json = nlohmann::json;

constexpr float kRadiansPerDegree = 0.017453292519943295f;
constexpr float kDegreesPerRadian = 57.29577951308232f;

bool ParseLightType(std::string_view text, LightType& type)
{
    if (text == "directional")
    {
        type = LightType::Directional;
        return true;
    }
    if (text == "point")
    {
        type = LightType::Point;
        return true;
    }
    if (text == "spot")
    {
        type = LightType::Spot;
        return true;
    }
    return false;
}

std::string_view LightTypeName(LightType type)
{
    switch (type)
    {
    case LightType::Directional:
        return "directional";
    case LightType::Point:
        return "point";
    case LightType::Spot:
        return "spot";
    }
    return "point";
}

bool ReadString(const Json& node, const char* key, std::string& out, bool required, std::string& error)
{
    const auto it = node.find(key);
    if (it == node.end())
    {
        if (required)
        {
            error = std::string{"missing required string '"} + key + "'";
            return false;
        }
        return true;
    }
    if (!it->is_string())
    {
        error = std::string{"'"} + key + "' must be a string";
        return false;
    }
    out = it->get<std::string>();
    return true;
}

bool ReadBool(const Json& node, const char* key, bool& out, std::string& error)
{
    const auto it = node.find(key);
    if (it == node.end())
    {
        return true;
    }
    if (!it->is_boolean())
    {
        error = std::string{"'"} + key + "' must be a boolean";
        return false;
    }
    out = it->get<bool>();
    return true;
}

bool ReadFloat(const Json& node, const char* key, float& out, std::string& error)
{
    const auto it = node.find(key);
    if (it == node.end())
    {
        return true;
    }
    if (!it->is_number())
    {
        error = std::string{"'"} + key + "' must be a number";
        return false;
    }
    out = it->get<float>();
    return true;
}

bool ReadVec3(const Json& node, const char* key, glm::vec3& out, std::string& error)
{
    const auto it = node.find(key);
    if (it == node.end())
    {
        return true;
    }
    if (!it->is_array() || it->size() != 3)
    {
        error = std::string{"'"} + key + "' must be an array of three numbers";
        return false;
    }
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!(*it)[axis].is_number())
        {
            error = std::string{"'"} + key + "' must contain only numbers";
            return false;
        }
        out[static_cast<int>(axis)] = (*it)[axis].get<float>();
    }
    return true;
}

std::string Indexed(std::string_view collection, std::size_t index, std::string message)
{
    return std::string{collection} + "[" + std::to_string(index) + "]: " + std::move(message);
}

Json Vec3ToJson(const glm::vec3& value)
{
    return Json::array({value.x, value.y, value.z});
}

} // namespace

bool SceneJson::Load(const std::filesystem::path& path, SceneDescription& scene, std::string& error)
{
    std::ifstream stream{path, std::ios::binary};
    if (!stream)
    {
        error = "failed to open scene file '" + PathToUtf8(path) + "'";
        return false;
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return Parse(buffer.str(), path.parent_path(), scene, error);
}

bool SceneJson::Parse(
    std::string_view text,
    const std::filesystem::path& baseDirectory,
    SceneDescription& scene,
    std::string& error)
{
    error.clear();

    const Json root = Json::parse(text.data(), text.data() + text.size(), nullptr, false);
    if (root.is_discarded())
    {
        error = "scene JSON could not be parsed";
        return false;
    }
    if (!root.is_object())
    {
        error = "scene root must be a JSON object";
        return false;
    }

    scene = SceneDescription{};
    scene.name = root.value("name", std::string{"Untitled Scene"});

    std::filesystem::path assetRoot = PathFromUtf8(root.value("assetRoot", std::string{}));
    if (assetRoot.empty())
    {
        assetRoot = baseDirectory;
    }
    else if (assetRoot.is_relative())
    {
        assetRoot = baseDirectory / assetRoot;
    }

    std::error_code pathError{};
    const std::filesystem::path canonical = std::filesystem::weakly_canonical(assetRoot, pathError);
    scene.assetRoot = pathError ? assetRoot.lexically_normal() : canonical;

    if (const auto objectsIt = root.find("objects"); objectsIt != root.end())
    {
        if (!objectsIt->is_array())
        {
            error = "'objects' must be an array";
            return false;
        }

        for (std::size_t index = 0; index < objectsIt->size(); ++index)
        {
            const Json& node = (*objectsIt)[index];
            if (!node.is_object())
            {
                error = Indexed("objects", index, "must be an object");
                return false;
            }

            SceneObject object{};
            std::string entryError;
            if (!ReadString(node, "name", object.name, false, entryError) ||
                !ReadString(node, "model", object.modelPath, true, entryError) ||
                !ReadVec3(node, "position", object.position, entryError) ||
                !ReadVec3(node, "rotation", object.rotation, entryError) ||
                !ReadVec3(node, "scale", object.scale, entryError))
            {
                error = Indexed("objects", index, std::move(entryError));
                return false;
            }

            if (object.modelPath.empty())
            {
                error = Indexed("objects", index, "'model' must not be empty");
                return false;
            }
            if (object.name.empty())
            {
                object.name = "Object " + std::to_string(index);
            }
            scene.objects.push_back(std::move(object));
        }
    }

    if (const auto probesIt = root.find("probes"); probesIt != root.end())
    {
        if (!probesIt->is_array())
        {
            error = "'probes' must be an array";
            return false;
        }

        for (std::size_t index = 0; index < probesIt->size(); ++index)
        {
            const Json& node = (*probesIt)[index];
            if (!node.is_object())
            {
                error = Indexed("probes", index, "must be an object");
                return false;
            }

            SceneProbe probe{};
            std::string entryError;
            if (!ReadString(node, "name", probe.name, false, entryError) ||
                !ReadVec3(node, "position", probe.position, entryError))
            {
                error = Indexed("probes", index, std::move(entryError));
                return false;
            }
            if (probe.name.empty())
            {
                probe.name = "Probe " + std::to_string(index);
            }
            probe.radius = node.value("radius", 0.5f);
            probe.enabled = node.value("enabled", true);
            scene.probes.push_back(std::move(probe));
        }
    }

    if (const auto lightsIt = root.find("lights"); lightsIt != root.end())
    {
        if (!lightsIt->is_array())
        {
            error = "'lights' must be an array";
            return false;
        }

        for (std::size_t index = 0; index < lightsIt->size(); ++index)
        {
            const Json& node = (*lightsIt)[index];
            if (!node.is_object())
            {
                error = Indexed("lights", index, "must be an object");
                return false;
            }

            SceneLight sceneLight{};
            std::string typeText{};
            std::string entryError;
            if (!ReadString(node, "name", sceneLight.name, false, entryError) ||
                !ReadString(node, "type", typeText, false, entryError) ||
                !ReadVec3(node, "position", sceneLight.light.position, entryError) ||
                !ReadVec3(node, "direction", sceneLight.light.direction, entryError) ||
                !ReadVec3(node, "color", sceneLight.light.color, entryError) ||
                !ReadFloat(node, "intensity", sceneLight.light.intensity, entryError) ||
                !ReadFloat(node, "range", sceneLight.light.range, entryError) ||
                !ReadFloat(node, "innerConeAngle", sceneLight.light.innerConeAngle, entryError) ||
                !ReadFloat(node, "outerConeAngle", sceneLight.light.outerConeAngle, entryError) ||
                !ReadBool(node, "castShadows", sceneLight.light.castShadows, entryError))
            {
                error = Indexed("lights", index, std::move(entryError));
                return false;
            }

            if (!typeText.empty() && !ParseLightType(typeText, sceneLight.light.type))
            {
                error = Indexed(
                    "lights",
                    index,
                    "unknown type '" + typeText + "' (expected directional, point or spot)");
                return false;
            }

            // Angles are authored in degrees; the runtime light stores radians.
            if (node.contains("innerConeAngle"))
            {
                sceneLight.light.innerConeAngle *= kRadiansPerDegree;
            }
            if (node.contains("outerConeAngle"))
            {
                sceneLight.light.outerConeAngle *= kRadiansPerDegree;
            }

            if (sceneLight.name.empty())
            {
                sceneLight.name = "Light " + std::to_string(index);
            }
            scene.lights.push_back(std::move(sceneLight));
        }
    }

    return true;
}

std::string SceneJson::Serialize(const SceneDescription& scene)
{
    Json root;
    root["name"] = scene.name;
    root["assetRoot"] = PathToUtf8(scene.assetRoot);

    Json objects = Json::array();
    for (const SceneObject& object : scene.objects)
    {
        Json node;
        node["name"] = object.name;
        node["model"] = object.modelPath;
        node["position"] = Vec3ToJson(object.position);
        node["rotation"] = Vec3ToJson(object.rotation);
        node["scale"] = Vec3ToJson(object.scale);
        objects.push_back(std::move(node));
    }
    root["objects"] = std::move(objects);

    Json probes = Json::array();
    for (const SceneProbe& probe : scene.probes)
    {
        Json node;
        node["name"] = probe.name;
        node["position"] = Vec3ToJson(probe.position);
        node["radius"] = probe.radius;
        node["enabled"] = probe.enabled;
        probes.push_back(std::move(node));
    }
    root["probes"] = std::move(probes);

    Json lights = Json::array();
    for (const SceneLight& sceneLight : scene.lights)
    {
        const Light& light = sceneLight.light;
        Json node;
        node["name"] = sceneLight.name;
        node["type"] = std::string{LightTypeName(light.type)};
        node["color"] = Vec3ToJson(light.color);
        node["intensity"] = light.intensity;

        switch (light.type)
        {
        case LightType::Directional:
            node["direction"] = Vec3ToJson(light.direction);
            node["castShadows"] = light.castShadows;
            break;
        case LightType::Point:
            node["position"] = Vec3ToJson(light.position);
            node["range"] = light.range;
            break;
        case LightType::Spot:
            node["position"] = Vec3ToJson(light.position);
            node["direction"] = Vec3ToJson(light.direction);
            node["range"] = light.range;
            node["innerConeAngle"] = light.innerConeAngle * kDegreesPerRadian;
            node["outerConeAngle"] = light.outerConeAngle * kDegreesPerRadian;
            break;
        }

        lights.push_back(std::move(node));
    }
    root["lights"] = std::move(lights);

    return root.dump(2) + "\n";
}

bool SceneJson::Save(const std::filesystem::path& path, const SceneDescription& scene, std::string& error)
{
    std::ofstream stream{path, std::ios::binary | std::ios::trunc};
    if (!stream)
    {
        error = "failed to open scene file '" + PathToUtf8(path) + "' for writing";
        return false;
    }

    stream << Serialize(scene);
    if (!stream)
    {
        error = "failed to write scene file '" + PathToUtf8(path) + "'";
        return false;
    }
    return true;
}

} // namespace mukki
