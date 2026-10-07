#include "Scene/SceneBinary.h"

#include <cstring>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace
{

mukki::SceneDescription MakeScene()
{
    mukki::SceneDescription scene{};
    scene.name = "Binary Scene";
    scene.assetRoot = std::filesystem::path{"C:/assets/models"};

    mukki::SceneObject box{};
    box.name = "Box";
    box.modelPath = "Box/glTF/Box.gltf";
    box.position = glm::vec3{1.0f, 2.0f, 3.0f};
    box.rotation = glm::vec3{10.0f, 20.0f, 30.0f};
    box.scale = glm::vec3{2.0f, 2.0f, 2.0f};
    scene.objects.push_back(box);

    mukki::SceneObject duck{};
    duck.name = "Duck";
    duck.modelPath = "Duck.gltf";
    scene.objects.push_back(duck);

    mukki::SceneProbe probe{};
    probe.name = "Probe";
    probe.position = glm::vec3{-1.0f, 2.5f, 3.5f};
    probe.radius = 0.75f;
    probe.enabled = false;
    scene.probes.push_back(probe);

    mukki::SceneLight sun{};
    sun.name = "Sun";
    sun.light.type = mukki::LightType::Directional;
    sun.light.direction = glm::vec3{0.0f, -1.0f, 0.0f};
    sun.light.color = glm::vec3{1.0f, 0.9f, 0.8f};
    sun.light.intensity = 2.5f;
    sun.light.castShadows = false;
    scene.lights.push_back(sun);

    mukki::SceneLight spot{};
    spot.name = "Spot";
    spot.light.type = mukki::LightType::Spot;
    spot.light.position = glm::vec3{0.0f, 4.0f, 0.0f};
    spot.light.direction = glm::vec3{0.0f, -1.0f, 0.0f};
    spot.light.range = 12.0f;
    spot.light.innerConeAngle = 0.35f;
    spot.light.outerConeAngle = 0.6f;
    scene.lights.push_back(spot);

    return scene;
}

TEST(SceneBinary, RoundTrips)
{
    const mukki::SceneDescription scene = MakeScene();
    const std::vector<std::byte> data = mukki::SceneBinary::Encode(scene);

    ASSERT_GT(data.size(), 0u);
    EXPECT_EQ(std::memcmp(data.data(), "MRES", 4), 0);

    mukki::SceneDescription decoded{};
    std::string error{};
    ASSERT_TRUE(mukki::SceneBinary::Decode(data, decoded, error)) << error;

    EXPECT_EQ(decoded.name, scene.name);
    EXPECT_EQ(decoded.assetRoot, scene.assetRoot);
    ASSERT_EQ(decoded.objects.size(), scene.objects.size());
    EXPECT_EQ(decoded.objects[0].name, scene.objects[0].name);
    EXPECT_EQ(decoded.objects[0].modelPath, scene.objects[0].modelPath);
    EXPECT_FLOAT_EQ(decoded.objects[0].position.y, scene.objects[0].position.y);
    EXPECT_FLOAT_EQ(decoded.objects[0].rotation.z, scene.objects[0].rotation.z);
    EXPECT_FLOAT_EQ(decoded.objects[0].scale.x, scene.objects[0].scale.x);

    ASSERT_EQ(decoded.probes.size(), scene.probes.size());
    EXPECT_EQ(decoded.probes[0].name, scene.probes[0].name);
    EXPECT_FLOAT_EQ(decoded.probes[0].position.y, scene.probes[0].position.y);
    EXPECT_FLOAT_EQ(decoded.probes[0].radius, scene.probes[0].radius);
    EXPECT_FALSE(decoded.probes[0].enabled);

    ASSERT_EQ(decoded.lights.size(), scene.lights.size());
    EXPECT_EQ(decoded.lights[0].light.type, mukki::LightType::Directional);
    EXPECT_FLOAT_EQ(decoded.lights[0].light.intensity, scene.lights[0].light.intensity);
    EXPECT_FALSE(decoded.lights[0].light.castShadows);
    EXPECT_TRUE(decoded.lights[1].light.castShadows);
    EXPECT_EQ(decoded.lights[1].light.type, mukki::LightType::Spot);
    EXPECT_FLOAT_EQ(decoded.lights[1].light.range, scene.lights[1].light.range);
    EXPECT_FLOAT_EQ(decoded.lights[1].light.outerConeAngle, scene.lights[1].light.outerConeAngle);
}

TEST(SceneBinary, RejectsCorruptedData)
{
    const mukki::SceneDescription scene = MakeScene();
    std::vector<std::byte> data = mukki::SceneBinary::Encode(scene);

    std::string error{};
    mukki::SceneDescription decoded{};

    EXPECT_FALSE(mukki::SceneBinary::Decode({}, decoded, error));
    EXPECT_FALSE(error.empty());

    EXPECT_FALSE(mukki::SceneBinary::Decode(
        std::span<const std::byte>{data.data(), data.size() / 2},
        decoded,
        error));

    std::vector<std::byte> badMagic = data;
    badMagic[0] = std::byte{0};
    EXPECT_FALSE(mukki::SceneBinary::Decode(badMagic, decoded, error));

    std::vector<std::byte> badVersion = data;
    badVersion[4] = std::byte{0xFF};
    EXPECT_FALSE(mukki::SceneBinary::Decode(badVersion, decoded, error));
    EXPECT_NE(error.find("version"), std::string::npos);
}

TEST(SceneBinary, WriteAndReadFile)
{
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "mukki_test_scene.mres";
    const mukki::SceneDescription scene = MakeScene();

    std::string error{};
    ASSERT_TRUE(mukki::SceneBinary::Write(path, scene, error)) << error;

    mukki::SceneDescription loaded{};
    ASSERT_TRUE(mukki::SceneBinary::Read(path, loaded, error)) << error;
    EXPECT_EQ(loaded.name, scene.name);
    EXPECT_EQ(loaded.objects.size(), scene.objects.size());
    EXPECT_EQ(loaded.lights.size(), scene.lights.size());

    std::error_code removeError{};
    std::filesystem::remove(path, removeError);
}

} // namespace
