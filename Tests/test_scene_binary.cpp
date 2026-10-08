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
    box.physics.body = mukki::ScenePhysicsBody::Dynamic;
    box.physics.shape = mukki::ScenePhysicsShape::Capsule;
    box.physics.halfExtents = glm::vec3{0.4f, 0.8f, 0.4f};
    box.physics.radius = 0.4f;
    box.physics.height = 1.6f;
    box.physics.offset = glm::vec3{0.0f, 0.8f, 0.0f};
    box.physics.density = 450.0f;
    box.physics.friction = 0.7f;
    box.physics.restitution = 0.2f;
    scene.objects.push_back(box);

    mukki::SceneObject duck{};
    duck.name = "Duck";
    duck.modelPath = "Duck.gltf";
    duck.physics.body = mukki::ScenePhysicsBody::Static;
    scene.objects.push_back(duck);

    scene.physics.gravity = glm::vec3{0.0f, -3.7f, 0.0f};
    scene.physics.stepHz = 90.0f;
    scene.physics.subSteps = 2;

    scene.hasEnvironment = true;
    scene.environment.map = std::filesystem::path{"env/sunrise.hdr"};
    scene.environment.intensity = 0.8f;
    scene.hasClouds = true;
    scene.clouds.coverage = 0.62f;
    scene.clouds.cloudType = 0.4f;
    scene.clouds.layerBottomKm = 0.9f;
    scene.clouds.layerTopKm = 2.8f;
    scene.clouds.windSpeed = 25.0f;
    scene.clouds.temporalBlend = 0.85f;
    scene.playerPlane = "Box";

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
    EXPECT_EQ(decoded.objects[0].physics.body, mukki::ScenePhysicsBody::Dynamic);
    EXPECT_EQ(decoded.objects[0].physics.shape, mukki::ScenePhysicsShape::Capsule);
    EXPECT_FLOAT_EQ(decoded.objects[0].physics.radius, scene.objects[0].physics.radius);
    EXPECT_FLOAT_EQ(decoded.objects[0].physics.height, scene.objects[0].physics.height);
    EXPECT_FLOAT_EQ(decoded.objects[0].physics.offset.y, scene.objects[0].physics.offset.y);
    EXPECT_FLOAT_EQ(decoded.objects[0].physics.density, scene.objects[0].physics.density);
    EXPECT_FLOAT_EQ(decoded.objects[0].physics.friction, scene.objects[0].physics.friction);
    EXPECT_FLOAT_EQ(decoded.objects[0].physics.restitution, scene.objects[0].physics.restitution);
    EXPECT_EQ(decoded.objects[1].physics.body, mukki::ScenePhysicsBody::Static);
    EXPECT_FLOAT_EQ(decoded.physics.gravity.y, scene.physics.gravity.y);
    EXPECT_FLOAT_EQ(decoded.physics.stepHz, scene.physics.stepHz);
    EXPECT_EQ(decoded.physics.subSteps, scene.physics.subSteps);

    EXPECT_TRUE(decoded.hasEnvironment);
    EXPECT_EQ(decoded.environment.map, scene.environment.map);
    EXPECT_FLOAT_EQ(decoded.environment.intensity, scene.environment.intensity);
    EXPECT_TRUE(decoded.hasClouds);
    EXPECT_TRUE(decoded.clouds.enabled);
    EXPECT_FLOAT_EQ(decoded.clouds.coverage, scene.clouds.coverage);
    EXPECT_FLOAT_EQ(decoded.clouds.cloudType, scene.clouds.cloudType);
    EXPECT_FLOAT_EQ(decoded.clouds.layerBottomKm, scene.clouds.layerBottomKm);
    EXPECT_FLOAT_EQ(decoded.clouds.layerTopKm, scene.clouds.layerTopKm);
    EXPECT_FLOAT_EQ(decoded.clouds.windSpeed, scene.clouds.windSpeed);
    EXPECT_FLOAT_EQ(decoded.clouds.temporalBlend, scene.clouds.temporalBlend);
    EXPECT_EQ(decoded.playerPlane, "Box");

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

// Writes the little endian layout used by the legacy (pre-v6) scene binary versions.
class LegacyWriter
{
public:
    void U32(std::uint32_t value)
    {
        for (std::uint32_t byte = 0; byte < 4; ++byte)
        {
            m_data.push_back(static_cast<std::byte>((value >> (byte * 8)) & 0xFFu));
        }
    }

    void F32(float value)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        U32(bits);
    }

    void String(std::string_view value)
    {
        U32(static_cast<std::uint32_t>(value.size()));
        for (const char character : value)
        {
            m_data.push_back(static_cast<std::byte>(static_cast<unsigned char>(character)));
        }
    }

    void Vec3(const glm::vec3& value)
    {
        F32(value.x);
        F32(value.y);
        F32(value.z);
    }

    [[nodiscard]] const std::vector<std::byte>& Data() const { return m_data; }

private:
    std::vector<std::byte> m_data{};
};

TEST(SceneBinary, DecodesLegacyVersionFourWithoutPhysics)
{
    LegacyWriter writer{};
    // v4 layout: header, name, assetRoot, objects (no physics), probes, lights.
    writer.U32(mukki::SceneBinary::kMagic);
    writer.U32(mukki::SceneBinary::kLegacyVersion);
    writer.String("Legacy Scene");
    writer.String("");
    writer.U32(1);
    writer.String("LegacyBox");
    writer.String("Box.gltf");
    writer.Vec3(glm::vec3{1.0f, 2.0f, 3.0f});
    writer.Vec3(glm::vec3{0.0f});
    writer.Vec3(glm::vec3{1.0f});
    writer.U32(0);
    writer.U32(0);

    mukki::SceneDescription decoded{};
    std::string error{};
    ASSERT_TRUE(mukki::SceneBinary::Decode(writer.Data(), decoded, error)) << error;
    ASSERT_EQ(decoded.objects.size(), 1u);
    EXPECT_EQ(decoded.objects[0].name, "LegacyBox");
    EXPECT_EQ(decoded.objects[0].physics.body, mukki::ScenePhysicsBody::None);
    EXPECT_FLOAT_EQ(decoded.physics.gravity.y, -9.81f);
    EXPECT_FLOAT_EQ(decoded.physics.stepHz, 60.0f);
    EXPECT_EQ(decoded.physics.subSteps, 4u);
    EXPECT_FALSE(decoded.hasEnvironment);
    EXPECT_FALSE(decoded.hasClouds);
    EXPECT_TRUE(decoded.playerPlane.empty());
}

TEST(SceneBinary, DecodesLegacyVersionFiveWithoutSceneSettings)
{
    LegacyWriter writer{};
    // v5 layout: v4 plus per-object physics and the scene root physics block.
    writer.U32(mukki::SceneBinary::kMagic);
    writer.U32(5);
    writer.String("Physics Scene");
    writer.String("");
    writer.U32(1);
    writer.String("Crate");
    writer.String("Box.gltf");
    writer.Vec3(glm::vec3{0.0f, 4.0f, 0.0f});
    writer.Vec3(glm::vec3{0.0f});
    writer.Vec3(glm::vec3{1.0f});
    writer.U32(2); // dynamic
    writer.U32(1); // sphere
    writer.Vec3(glm::vec3{0.0f});
    writer.F32(0.5f);
    writer.F32(1.0f);
    writer.Vec3(glm::vec3{0.0f});
    writer.F32(400.0f);
    writer.F32(0.6f);
    writer.F32(0.1f);
    writer.U32(0); // probes
    writer.U32(0); // lights
    writer.Vec3(glm::vec3{0.0f, -12.0f, 0.0f});
    writer.F32(120.0f);
    writer.U32(3);

    mukki::SceneDescription decoded{};
    std::string error{};
    ASSERT_TRUE(mukki::SceneBinary::Decode(writer.Data(), decoded, error)) << error;
    ASSERT_EQ(decoded.objects.size(), 1u);
    EXPECT_EQ(decoded.objects[0].physics.body, mukki::ScenePhysicsBody::Dynamic);
    EXPECT_EQ(decoded.objects[0].physics.shape, mukki::ScenePhysicsShape::Sphere);
    EXPECT_FLOAT_EQ(decoded.objects[0].physics.radius, 0.5f);
    EXPECT_FLOAT_EQ(decoded.physics.gravity.y, -12.0f);
    EXPECT_FLOAT_EQ(decoded.physics.stepHz, 120.0f);
    EXPECT_EQ(decoded.physics.subSteps, 3u);
    EXPECT_FALSE(decoded.hasEnvironment);
    EXPECT_FALSE(decoded.hasClouds);
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
