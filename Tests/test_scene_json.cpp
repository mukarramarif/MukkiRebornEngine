#include "Scene/SceneJson.h"

#include <cmath>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

namespace
{

constexpr const char* kSceneText = R"({
  "name": "Test Scene",
  "assetRoot": "models",
  "physics": {
    "gravity": [0.0, -12.0, 0.0],
    "stepHz": 90.0,
    "subSteps": 2
  },
  "environment": {
    "map": "env/sunrise.hdr",
    "intensity": 0.8
  },
  "clouds": {
    "enabled": true,
    "coverage": 0.62,
    "cloudType": 0.33,
    "density": 0.03,
    "layerBottomKm": 0.9,
    "layerTopKm": 2.8,
    "windSpeed": 25.0,
    "windDirectionRadians": 0.7,
    "coverageVariation": 0.5,
    "temporalBlend": 0.85
  },
  "playerPlane": "Box",
  "objects": [
    {
      "name": "Box",
      "model": "Box/glTF/Box.gltf",
      "position": [1.0, 2.0, 3.0],
      "rotation": [10.0, 20.0, 30.0],
      "scale": [2.0, 2.0, 2.0],
      "physics": {
        "body": "dynamic",
        "shape": "box",
        "halfExtents": [0.5, 0.5, 0.5],
        "density": 400.0,
        "friction": 0.7,
        "restitution": 0.1
      }
    },
    {
      "model": "Duck.gltf",
      "physics": {
        "body": "static",
        "shape": "sphere",
        "radius": 0.25
      }
    }
  ],
  "probes": [
    {
      "name": "Center",
      "position": [0.0, 2.5, 0.0],
      "radius": 0.75
    },
    {
      "position": [-4.0, 1.5, 0.0],
      "enabled": false
    }
  ],
  "lights": [
    {
      "name": "Sun",
      "type": "directional",
      "direction": [0.0, -1.0, 0.0],
      "color": [1.0, 0.9, 0.8],
      "intensity": 3.0,
      "castShadows": false
    },
    {
      "name": "Bulb",
      "type": "point",
      "position": [1.0, 2.0, 3.0],
      "range": 8.0,
      "intensity": 5.0
    },
    {
      "name": "Cone",
      "type": "spot",
      "position": [0.0, 4.0, 0.0],
      "direction": [0.0, -1.0, 0.0],
      "innerConeAngle": 20.0,
      "outerConeAngle": 35.0
    }
  ]
})";

constexpr float kRadiansPerDegree = 0.017453292519943295f;

std::filesystem::path TestBaseDirectory()
{
    return std::filesystem::path{"C:/scenes"};
}

TEST(SceneJson, ParsesObjectsAndLights)
{
    mukki::SceneDescription scene{};
    std::string error{};
    ASSERT_TRUE(mukki::SceneJson::Parse(kSceneText, TestBaseDirectory(), scene, error)) << error;

    EXPECT_EQ(scene.name, "Test Scene");
    EXPECT_EQ(
        scene.assetRoot,
        std::filesystem::weakly_canonical(TestBaseDirectory() / "models"));

    ASSERT_EQ(scene.objects.size(), 2u);
    EXPECT_EQ(scene.objects[0].name, "Box");
    EXPECT_EQ(scene.objects[0].modelPath, "Box/glTF/Box.gltf");
    EXPECT_FLOAT_EQ(scene.objects[0].position.x, 1.0f);
    EXPECT_FLOAT_EQ(scene.objects[0].position.y, 2.0f);
    EXPECT_FLOAT_EQ(scene.objects[0].position.z, 3.0f);
    EXPECT_FLOAT_EQ(scene.objects[0].rotation.y, 20.0f);
    EXPECT_FLOAT_EQ(scene.objects[0].scale.x, 2.0f);

    EXPECT_EQ(scene.objects[1].name, "Object 1");
    EXPECT_FLOAT_EQ(scene.objects[1].scale.y, 1.0f);

    EXPECT_EQ(scene.objects[0].physics.body, mukki::ScenePhysicsBody::Dynamic);
    EXPECT_EQ(scene.objects[0].physics.shape, mukki::ScenePhysicsShape::Box);
    EXPECT_FLOAT_EQ(scene.objects[0].physics.halfExtents.x, 0.5f);
    EXPECT_FLOAT_EQ(scene.objects[0].physics.density, 400.0f);
    EXPECT_FLOAT_EQ(scene.objects[0].physics.friction, 0.7f);
    EXPECT_FLOAT_EQ(scene.objects[0].physics.restitution, 0.1f);
    EXPECT_EQ(scene.objects[1].physics.body, mukki::ScenePhysicsBody::Static);
    EXPECT_EQ(scene.objects[1].physics.shape, mukki::ScenePhysicsShape::Sphere);
    EXPECT_FLOAT_EQ(scene.objects[1].physics.radius, 0.25f);
    EXPECT_FLOAT_EQ(scene.physics.gravity.y, -12.0f);
    EXPECT_FLOAT_EQ(scene.physics.stepHz, 90.0f);
    EXPECT_EQ(scene.physics.subSteps, 2u);

    EXPECT_TRUE(scene.hasEnvironment);
    EXPECT_EQ(scene.environment.map, std::filesystem::path{"env/sunrise.hdr"});
    EXPECT_FLOAT_EQ(scene.environment.intensity, 0.8f);

    EXPECT_TRUE(scene.hasClouds);
    EXPECT_TRUE(scene.clouds.enabled);
    EXPECT_FLOAT_EQ(scene.clouds.coverage, 0.62f);
    EXPECT_FLOAT_EQ(scene.clouds.layerBottomKm, 0.9f);
    EXPECT_FLOAT_EQ(scene.clouds.layerTopKm, 2.8f);
    EXPECT_FLOAT_EQ(scene.clouds.windSpeed, 25.0f);
    EXPECT_FLOAT_EQ(scene.clouds.temporalBlend, 0.85f);
    EXPECT_EQ(scene.clouds.primarySteps, 64u);
    EXPECT_EQ(scene.clouds.lightSteps, 6u);

    EXPECT_EQ(scene.playerPlane, "Box");

    ASSERT_EQ(scene.probes.size(), 2u);
    EXPECT_EQ(scene.probes[0].name, "Center");
    EXPECT_FLOAT_EQ(scene.probes[0].position.y, 2.5f);
    EXPECT_FLOAT_EQ(scene.probes[0].radius, 0.75f);
    EXPECT_TRUE(scene.probes[0].enabled);
    EXPECT_EQ(scene.probes[1].name, "Probe 1");
    EXPECT_FALSE(scene.probes[1].enabled);

    ASSERT_EQ(scene.lights.size(), 3u);
    EXPECT_EQ(scene.lights[0].name, "Sun");
    EXPECT_EQ(scene.lights[0].light.type, mukki::LightType::Directional);
    EXPECT_FLOAT_EQ(scene.lights[0].light.direction.y, -1.0f);
    EXPECT_FLOAT_EQ(scene.lights[0].light.intensity, 3.0f);
    EXPECT_FALSE(scene.lights[0].light.castShadows);

    EXPECT_EQ(scene.lights[1].light.type, mukki::LightType::Point);
    EXPECT_FLOAT_EQ(scene.lights[1].light.range, 8.0f);
    EXPECT_TRUE(scene.lights[1].light.castShadows);

    EXPECT_EQ(scene.lights[2].light.type, mukki::LightType::Spot);
    EXPECT_NEAR(scene.lights[2].light.innerConeAngle, 20.0f * kRadiansPerDegree, 1e-6f);
    EXPECT_NEAR(scene.lights[2].light.outerConeAngle, 35.0f * kRadiansPerDegree, 1e-6f);
}

TEST(SceneJson, RoundTripsThroughSerialization)
{
    mukki::SceneDescription scene{};
    std::string error{};
    ASSERT_TRUE(mukki::SceneJson::Parse(kSceneText, TestBaseDirectory(), scene, error)) << error;

    const std::string serialized = mukki::SceneJson::Serialize(scene);

    mukki::SceneDescription reloaded{};
    ASSERT_TRUE(mukki::SceneJson::Parse(serialized, TestBaseDirectory(), reloaded, error)) << error;

    EXPECT_EQ(reloaded.name, scene.name);
    EXPECT_EQ(reloaded.assetRoot, scene.assetRoot);
    ASSERT_EQ(reloaded.objects.size(), scene.objects.size());
    ASSERT_EQ(reloaded.lights.size(), scene.lights.size());
    ASSERT_EQ(reloaded.probes.size(), scene.probes.size());

    for (std::size_t index = 0; index < scene.objects.size(); ++index)
    {
        EXPECT_EQ(reloaded.objects[index].name, scene.objects[index].name);
        EXPECT_EQ(reloaded.objects[index].modelPath, scene.objects[index].modelPath);
        EXPECT_FLOAT_EQ(reloaded.objects[index].position.x, scene.objects[index].position.x);
        EXPECT_FLOAT_EQ(reloaded.objects[index].rotation.z, scene.objects[index].rotation.z);
        EXPECT_FLOAT_EQ(reloaded.objects[index].scale.z, scene.objects[index].scale.z);
        EXPECT_EQ(reloaded.objects[index].physics.body, scene.objects[index].physics.body);
        EXPECT_EQ(reloaded.objects[index].physics.shape, scene.objects[index].physics.shape);
        EXPECT_FLOAT_EQ(
            reloaded.objects[index].physics.halfExtents.z,
            scene.objects[index].physics.halfExtents.z);
        EXPECT_FLOAT_EQ(reloaded.objects[index].physics.radius, scene.objects[index].physics.radius);
        EXPECT_FLOAT_EQ(reloaded.objects[index].physics.density, scene.objects[index].physics.density);
        EXPECT_FLOAT_EQ(reloaded.objects[index].physics.friction, scene.objects[index].physics.friction);
        EXPECT_FLOAT_EQ(
            reloaded.objects[index].physics.restitution,
            scene.objects[index].physics.restitution);
    }

    EXPECT_FLOAT_EQ(reloaded.physics.gravity.y, scene.physics.gravity.y);
    EXPECT_FLOAT_EQ(reloaded.physics.stepHz, scene.physics.stepHz);
    EXPECT_EQ(reloaded.physics.subSteps, scene.physics.subSteps);

    EXPECT_TRUE(reloaded.hasEnvironment);
    EXPECT_EQ(reloaded.environment.map, scene.environment.map);
    EXPECT_FLOAT_EQ(reloaded.environment.intensity, scene.environment.intensity);

    EXPECT_TRUE(reloaded.hasClouds);
    EXPECT_EQ(reloaded.clouds.enabled, scene.clouds.enabled);
    EXPECT_FLOAT_EQ(reloaded.clouds.coverage, scene.clouds.coverage);
    EXPECT_FLOAT_EQ(reloaded.clouds.cloudType, scene.clouds.cloudType);
    EXPECT_FLOAT_EQ(reloaded.clouds.density, scene.clouds.density);
    EXPECT_FLOAT_EQ(reloaded.clouds.detailStrength, scene.clouds.detailStrength);
    EXPECT_FLOAT_EQ(reloaded.clouds.layerBottomKm, scene.clouds.layerBottomKm);
    EXPECT_FLOAT_EQ(reloaded.clouds.layerTopKm, scene.clouds.layerTopKm);
    EXPECT_FLOAT_EQ(reloaded.clouds.altitudeVariationKm, scene.clouds.altitudeVariationKm);
    EXPECT_FLOAT_EQ(reloaded.clouds.windSpeed, scene.clouds.windSpeed);
    EXPECT_FLOAT_EQ(reloaded.clouds.windDirectionRadians, scene.clouds.windDirectionRadians);
    EXPECT_FLOAT_EQ(reloaded.clouds.curlStrength, scene.clouds.curlStrength);
    EXPECT_FLOAT_EQ(reloaded.clouds.coverageVariation, scene.clouds.coverageVariation);
    EXPECT_EQ(reloaded.clouds.primarySteps, scene.clouds.primarySteps);
    EXPECT_EQ(reloaded.clouds.lightSteps, scene.clouds.lightSteps);
    EXPECT_FLOAT_EQ(reloaded.clouds.resolutionScale, scene.clouds.resolutionScale);
    EXPECT_FLOAT_EQ(reloaded.clouds.temporalBlend, scene.clouds.temporalBlend);
    EXPECT_FLOAT_EQ(reloaded.clouds.godRayIntensity, scene.clouds.godRayIntensity);

    EXPECT_EQ(reloaded.playerPlane, scene.playerPlane);

    for (std::size_t index = 0; index < scene.probes.size(); ++index)
    {
        EXPECT_EQ(reloaded.probes[index].name, scene.probes[index].name);
        EXPECT_FLOAT_EQ(reloaded.probes[index].position.y, scene.probes[index].position.y);
        EXPECT_FLOAT_EQ(reloaded.probes[index].radius, scene.probes[index].radius);
        EXPECT_EQ(reloaded.probes[index].enabled, scene.probes[index].enabled);
    }

    for (std::size_t index = 0; index < scene.lights.size(); ++index)
    {
        EXPECT_EQ(reloaded.lights[index].name, scene.lights[index].name);
        EXPECT_EQ(reloaded.lights[index].light.type, scene.lights[index].light.type);
        EXPECT_EQ(reloaded.lights[index].light.castShadows, scene.lights[index].light.castShadows);
        EXPECT_FLOAT_EQ(reloaded.lights[index].light.intensity, scene.lights[index].light.intensity);
        EXPECT_NEAR(
            reloaded.lights[index].light.outerConeAngle,
            scene.lights[index].light.outerConeAngle,
            1e-5f);
    }
}

TEST(SceneJson, RejectsInvalidInput)
{
    const std::filesystem::path base = TestBaseDirectory();

    mukki::SceneDescription scene{};
    std::string error{};

    EXPECT_FALSE(mukki::SceneJson::Parse("this is not json", base, scene, error));
    EXPECT_FALSE(error.empty());

    EXPECT_FALSE(mukki::SceneJson::Parse(R"({"objects": [{}]})", base, scene, error));
    EXPECT_NE(error.find("model"), std::string::npos);

    EXPECT_FALSE(mukki::SceneJson::Parse(R"({"objects": "nope"})", base, scene, error));

    EXPECT_FALSE(mukki::SceneJson::Parse(
        R"({"lights": [{"type": "area"}]})",
        base,
        scene,
        error));
    EXPECT_NE(error.find("area"), std::string::npos);

    EXPECT_FALSE(mukki::SceneJson::Parse(
        R"({"lights": [{"position": [1.0, 2.0]}]})",
        base,
        scene,
        error));

    EXPECT_FALSE(mukki::SceneJson::Parse(
        R"({"objects": [{"model": "Box.gltf", "physics": {"body": "bouncy"}}]})",
        base,
        scene,
        error));
    EXPECT_NE(error.find("bouncy"), std::string::npos);

    EXPECT_FALSE(mukki::SceneJson::Parse(
        R"({"objects": [{"model": "Box.gltf", "physics": {"body": "dynamic", "shape": "torus"}}]})",
        base,
        scene,
        error));
    EXPECT_NE(error.find("torus"), std::string::npos);

    EXPECT_FALSE(mukki::SceneJson::Parse(
        R"({"physics": {"stepHz": 500.0}})",
        base,
        scene,
        error));

    EXPECT_FALSE(mukki::SceneJson::Parse(
        R"({"physics": {"subSteps": 0}})",
        base,
        scene,
        error));

    EXPECT_FALSE(mukki::SceneJson::Parse(
        R"({"clouds": {"layerBottomKm": 1.0, "layerTopKm": 0.5}})",
        base,
        scene,
        error));

    EXPECT_FALSE(mukki::SceneJson::Parse(
        R"({"clouds": {"primarySteps": 0}})",
        base,
        scene,
        error));

    EXPECT_FALSE(mukki::SceneJson::Parse(R"({"environment": []})", base, scene, error));

    EXPECT_FALSE(mukki::SceneJson::Parse(R"({"playerPlane": 5})", base, scene, error));
}

TEST(SceneJson, SceneSettingsAreOptional)
{
    mukki::SceneDescription scene{};
    std::string error{};
    ASSERT_TRUE(mukki::SceneJson::Parse(R"({"name": "Bare"})", TestBaseDirectory(), scene, error))
        << error;

    EXPECT_FALSE(scene.hasEnvironment);
    EXPECT_FALSE(scene.hasClouds);
    EXPECT_TRUE(scene.playerPlane.empty());

    const std::string serialized = mukki::SceneJson::Serialize(scene);
    EXPECT_EQ(serialized.find("\"environment\""), std::string::npos);
    EXPECT_EQ(serialized.find("\"clouds\""), std::string::npos);
    EXPECT_EQ(serialized.find("\"playerPlane\""), std::string::npos);
}

} // namespace
