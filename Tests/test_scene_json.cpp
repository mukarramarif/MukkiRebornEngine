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
  "objects": [
    {
      "name": "Box",
      "model": "Box/glTF/Box.gltf",
      "position": [1.0, 2.0, 3.0],
      "rotation": [10.0, 20.0, 30.0],
      "scale": [2.0, 2.0, 2.0]
    },
    {
      "model": "Duck.gltf"
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
    }

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
}

} // namespace
