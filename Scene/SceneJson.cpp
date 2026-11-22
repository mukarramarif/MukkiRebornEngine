#include "Scene/SceneJson.h"

#include <fstream>
#include <limits>
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

std::string_view PhysicsBodyName(ScenePhysicsBody body)
{
    switch (body)
    {
    case ScenePhysicsBody::None:
        return "none";
    case ScenePhysicsBody::Static:
        return "static";
    case ScenePhysicsBody::Dynamic:
        return "dynamic";
    }
    return "none";
}

bool ParsePhysicsBody(std::string_view text, ScenePhysicsBody& body)
{
    if (text == "none")
    {
        body = ScenePhysicsBody::None;
        return true;
    }
    if (text == "static")
    {
        body = ScenePhysicsBody::Static;
        return true;
    }
    if (text == "dynamic")
    {
        body = ScenePhysicsBody::Dynamic;
        return true;
    }
    return false;
}

std::string_view PhysicsShapeName(ScenePhysicsShape shape)
{
    switch (shape)
    {
    case ScenePhysicsShape::Box:
        return "box";
    case ScenePhysicsShape::Sphere:
        return "sphere";
    case ScenePhysicsShape::Capsule:
        return "capsule";
    }
    return "box";
}

bool ParsePhysicsShape(std::string_view text, ScenePhysicsShape& shape)
{
    if (text == "box")
    {
        shape = ScenePhysicsShape::Box;
        return true;
    }
    if (text == "sphere")
    {
        shape = ScenePhysicsShape::Sphere;
        return true;
    }
    if (text == "capsule")
    {
        shape = ScenePhysicsShape::Capsule;
        return true;
    }
    return false;
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

bool ReadU32(const Json& node, const char* key, std::uint32_t& out, std::string& error)
{
    const auto it = node.find(key);
    if (it == node.end())
    {
        return true;
    }
    if (!it->is_number_integer())
    {
        error = std::string{"'"} + key + "' must be an integer";
        return false;
    }
    const std::int64_t value = it->get<std::int64_t>();
    if (value < 0 || value > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()))
    {
        error = std::string{"'"} + key + "' is out of range";
        return false;
    }
    out = static_cast<std::uint32_t>(value);
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

// Parses the optional per-object "physics" block. An absent block (or "body": "none") leaves the
// object render-only.
bool ReadScenePhysics(const Json& node, ScenePhysics& physics, std::string& error)
{
    const auto it = node.find("physics");
    if (it == node.end())
    {
        return true;
    }
    if (!it->is_object())
    {
        error = "'physics' must be an object";
        return false;
    }

    std::string bodyText{};
    std::string shapeText{};
    if (!ReadString(*it, "body", bodyText, false, error) ||
        !ReadString(*it, "shape", shapeText, false, error) ||
        !ReadVec3(*it, "halfExtents", physics.halfExtents, error) ||
        !ReadVec3(*it, "offset", physics.offset, error) ||
        !ReadFloat(*it, "radius", physics.radius, error) ||
        !ReadFloat(*it, "height", physics.height, error) ||
        !ReadFloat(*it, "density", physics.density, error) ||
        !ReadFloat(*it, "friction", physics.friction, error) ||
        !ReadFloat(*it, "restitution", physics.restitution, error))
    {
        return false;
    }

    if (!bodyText.empty() && !ParsePhysicsBody(bodyText, physics.body))
    {
        error = "physics 'body' must be none, static or dynamic, got '" + bodyText + "'";
        return false;
    }
    if (!shapeText.empty() && !ParsePhysicsShape(shapeText, physics.shape))
    {
        error = "physics 'shape' must be box, sphere or capsule, got '" + shapeText + "'";
        return false;
    }
    if (physics.density < 0.0f || physics.friction < 0.0f || physics.restitution < 0.0f)
    {
        error = "physics density, friction and restitution must not be negative";
        return false;
    }
    return true;
}

std::string Indexed(std::string_view collection, std::size_t index, std::string message)
{
    return std::string{collection} + "[" + std::to_string(index) + "]: " + std::move(message);
}

// Parses the optional scene root "clouds" block into the engine cloud settings.
bool ReadClouds(const Json& node, CloudSettings& clouds, std::string& error)
{
    const auto it = node.find("clouds");
    if (it == node.end())
    {
        return true;
    }
    if (!it->is_object())
    {
        error = "'clouds' must be an object";
        return false;
    }

    if (!ReadBool(*it, "enabled", clouds.enabled, error) ||
        !ReadFloat(*it, "coverage", clouds.coverage, error) ||
        !ReadFloat(*it, "cloudType", clouds.cloudType, error) ||
        !ReadFloat(*it, "density", clouds.density, error) ||
        !ReadFloat(*it, "detailStrength", clouds.detailStrength, error) ||
        !ReadFloat(*it, "layerBottomKm", clouds.layerBottomKm, error) ||
        !ReadFloat(*it, "layerTopKm", clouds.layerTopKm, error) ||
        !ReadFloat(*it, "altitudeVariationKm", clouds.altitudeVariationKm, error) ||
        !ReadFloat(*it, "windSpeed", clouds.windSpeed, error) ||
        !ReadFloat(*it, "windDirectionRadians", clouds.windDirectionRadians, error) ||
        !ReadFloat(*it, "curlStrength", clouds.curlStrength, error) ||
        !ReadFloat(*it, "coverageVariation", clouds.coverageVariation, error) ||
        !ReadU32(*it, "primarySteps", clouds.primarySteps, error) ||
        !ReadU32(*it, "lightSteps", clouds.lightSteps, error) ||
        !ReadFloat(*it, "resolutionScale", clouds.resolutionScale, error) ||
        !ReadFloat(*it, "temporalBlend", clouds.temporalBlend, error) ||
        !ReadFloat(*it, "godRayIntensity", clouds.godRayIntensity, error))
    {
        return false;
    }

    if (clouds.density < 0.0f || clouds.layerBottomKm <= 0.0f ||
        clouds.layerTopKm <= clouds.layerBottomKm || clouds.altitudeVariationKm < 0.0f ||
        clouds.windSpeed < 0.0f || clouds.curlStrength < 0.0f)
    {
        error = "clouds density, layer heights, altitude variation, wind speed and curl must be valid";
        return false;
    }
    if (clouds.primarySteps < 1 || clouds.primarySteps > 256 || clouds.lightSteps < 1 ||
        clouds.lightSteps > 16)
    {
        error = "clouds primarySteps must be in [1, 256] and lightSteps in [1, 16]";
        return false;
    }
    if (clouds.resolutionScale < 0.1f || clouds.resolutionScale > 1.0f)
    {
        error = "clouds resolutionScale must be in [0.1, 1.0]";
        return false;
    }
    if (clouds.temporalBlend < 0.0f || clouds.temporalBlend >= 1.0f)
    {
        error = "clouds temporalBlend must be in [0, 1)";
        return false;
    }
    return true;
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

    if (const auto physicsIt = root.find("physics"); physicsIt != root.end())
    {
        if (!physicsIt->is_object())
        {
            error = "'physics' must be an object";
            return false;
        }
        if (!ReadVec3(*physicsIt, "gravity", scene.physics.gravity, error) ||
            !ReadFloat(*physicsIt, "stepHz", scene.physics.stepHz, error) ||
            !ReadU32(*physicsIt, "subSteps", scene.physics.subSteps, error))
        {
            return false;
        }
        if (scene.physics.stepHz < 15.0f || scene.physics.stepHz > 240.0f)
        {
            error = "physics 'stepHz' must be between 15 and 240";
            return false;
        }
        if (scene.physics.subSteps < 1 || scene.physics.subSteps > 8)
        {
            error = "physics 'subSteps' must be between 1 and 8";
            return false;
        }
    }

    if (const auto environmentIt = root.find("environment"); environmentIt != root.end())
    {
        if (!environmentIt->is_object())
        {
            error = "'environment' must be an object";
            return false;
        }
        std::string mapText{};
        if (!ReadString(*environmentIt, "map", mapText, false, error) ||
            !ReadFloat(*environmentIt, "intensity", scene.environment.intensity, error))
        {
            return false;
        }
        scene.environment.map = PathFromUtf8(mapText);
        if (scene.environment.intensity < 0.0f)
        {
            error = "environment 'intensity' must not be negative";
            return false;
        }
        scene.hasEnvironment = true;
    }

    scene.hasClouds = root.contains("clouds");
    if (!ReadClouds(root, scene.clouds, error) ||
        !ReadString(root, "playerPlane", scene.playerPlane, false, error))
    {
        return false;
    }

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
                !ReadVec3(node, "scale", object.scale, entryError) ||
                !ReadScenePhysics(node, object.physics, entryError))
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

    Json physicsNode;
    physicsNode["gravity"] = Vec3ToJson(scene.physics.gravity);
    physicsNode["stepHz"] = scene.physics.stepHz;
    physicsNode["subSteps"] = scene.physics.subSteps;
    root["physics"] = std::move(physicsNode);

    if (scene.hasEnvironment)
    {
        Json environment;
        environment["map"] = PathToUtf8(scene.environment.map);
        environment["intensity"] = scene.environment.intensity;
        root["environment"] = std::move(environment);
    }

    if (scene.hasClouds)
    {
        const CloudSettings& clouds = scene.clouds;
        Json cloudsNode;
        cloudsNode["enabled"] = clouds.enabled;
        cloudsNode["coverage"] = clouds.coverage;
        cloudsNode["cloudType"] = clouds.cloudType;
        cloudsNode["density"] = clouds.density;
        cloudsNode["detailStrength"] = clouds.detailStrength;
        cloudsNode["layerBottomKm"] = clouds.layerBottomKm;
        cloudsNode["layerTopKm"] = clouds.layerTopKm;
        cloudsNode["altitudeVariationKm"] = clouds.altitudeVariationKm;
        cloudsNode["windSpeed"] = clouds.windSpeed;
        cloudsNode["windDirectionRadians"] = clouds.windDirectionRadians;
        cloudsNode["curlStrength"] = clouds.curlStrength;
        cloudsNode["coverageVariation"] = clouds.coverageVariation;
        cloudsNode["primarySteps"] = clouds.primarySteps;
        cloudsNode["lightSteps"] = clouds.lightSteps;
        cloudsNode["resolutionScale"] = clouds.resolutionScale;
        cloudsNode["temporalBlend"] = clouds.temporalBlend;
        cloudsNode["godRayIntensity"] = clouds.godRayIntensity;
        root["clouds"] = std::move(cloudsNode);
    }

    if (!scene.playerPlane.empty())
    {
        root["playerPlane"] = scene.playerPlane;
    }

    Json objects = Json::array();
    for (const SceneObject& object : scene.objects)
    {
        Json node;
        node["name"] = object.name;
        node["model"] = object.modelPath;
        node["position"] = Vec3ToJson(object.position);
        node["rotation"] = Vec3ToJson(object.rotation);
        node["scale"] = Vec3ToJson(object.scale);

        if (object.physics.body != ScenePhysicsBody::None)
        {
            Json physics;
            physics["body"] = std::string{PhysicsBodyName(object.physics.body)};
            physics["shape"] = std::string{PhysicsShapeName(object.physics.shape)};
            physics["halfExtents"] = Vec3ToJson(object.physics.halfExtents);
            physics["radius"] = object.physics.radius;
            physics["height"] = object.physics.height;
            physics["offset"] = Vec3ToJson(object.physics.offset);
            physics["density"] = object.physics.density;
            physics["friction"] = object.physics.friction;
            physics["restitution"] = object.physics.restitution;
            node["physics"] = std::move(physics);
        }

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
