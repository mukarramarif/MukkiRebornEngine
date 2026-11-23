#include "Assets/GltfLoader.h"
#include "Utilities/PathUtils.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace
{

std::filesystem::path SampleModelsRoot()
{
    return std::filesystem::path{MRE_SOURCE_DIR} / "ThirdParty/glTF-Sample-Models/2.0";
}

std::filesystem::path AssetLibraryRoot()
{
    return std::filesystem::path{MRE_SOURCE_DIR} / "AssetLibrary";
}

TEST(GltfLoader, ParsesDiffuseTransmissionMaterials)
{
    const std::filesystem::path modelPath =
        std::filesystem::path{MRE_SOURCE_DIR} /
        "ThirdParty/glTF-Sample-Assets/Models/DiffuseTransmissionPlant/glTF/DiffuseTransmissionPlant.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "glTF-Sample-Assets submodule is not initialized";
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));

    const auto leaves = std::find_if(
        model.materials.begin(),
        model.materials.end(),
        [](const mukki::GltfMaterial& material) {
            return material.diffuseTransmissionFactor > 0.0f;
        });
    ASSERT_NE(leaves, model.materials.end());
    EXPECT_NEAR(leaves->diffuseTransmissionFactor, 0.1f, 1.0e-4f);
    EXPECT_FLOAT_EQ(leaves->diffuseTransmissionColorFactor.r, 1.0f);
    EXPECT_FLOAT_EQ(leaves->diffuseTransmissionColorFactor.g, 1.0f);
    EXPECT_FLOAT_EQ(leaves->diffuseTransmissionColorFactor.b, 1.0f);
    EXPECT_GE(leaves->diffuseTransmissionColorTexture, 0);
    EXPECT_EQ(leaves->diffuseTransmissionTexture, -1);
    EXPECT_TRUE(leaves->alphaMask);

    // Materials without the extension stay disabled.
    const auto opaque = std::find_if(
        model.materials.begin(),
        model.materials.end(),
        [](const mukki::GltfMaterial& material) {
            return material.diffuseTransmissionFactor == 0.0f;
        });
    EXPECT_NE(opaque, model.materials.end());
}

TEST(GltfLoader, LoadsFlightDemoAirplane)
{
    const std::filesystem::path modelPath =
        AssetLibraryRoot() / "aviation_aircraft_airplane_model/scene.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "flight demo airplane asset is not present";
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));

    EXPECT_GT(model.vertexCount, 10000u);
    EXPECT_GT(model.triangleCount, 10000u);
    ASSERT_FALSE(model.materials.empty());

    // The cockpit glass is an alphaMode = BLEND material, so it goes through the transparent pass.
    const bool hasBlendGlass = std::any_of(
        model.materials.begin(),
        model.materials.end(),
        [](const mukki::GltfMaterial& material) {
            return material.alphaBlend && material.baseColorFactor.a < 1.0f;
        });
    EXPECT_TRUE(hasBlendGlass);

    // The model must be usable from the flight corridor: forward -Z, up +Y.
    const glm::vec3 size = model.maxBounds - model.minBounds;
    EXPECT_GT(size.x, size.z);
}

TEST(GltfLoader, LoadsFlightDemoSkyscraper)
{
    const std::filesystem::path modelPath =
        AssetLibraryRoot() / "skyscraper_with_interior_with_animation_video/scene.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "flight demo skyscraper asset is not present";
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));

    EXPECT_GT(model.vertexCount, 100000u);
    EXPECT_GT(model.triangleCount, 100000u);
    EXPECT_GE(model.materials.size(), 6u);
    EXPECT_FALSE(model.primitives.empty());

    // The tower is much taller than it is wide.
    const glm::vec3 size = model.maxBounds - model.minBounds;
    EXPECT_GT(size.y, size.x * 2.0f);
    EXPECT_GT(size.y, size.z * 2.0f);
}

TEST(GltfLoader, LoadsDamagedHelmet)
{
    const std::filesystem::path modelPath = SampleModelsRoot() / "DamagedHelmet/glTF/DamagedHelmet.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "glTF sample models submodule is not initialized";
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));

    EXPECT_GT(model.vertexCount, 0u);
    EXPECT_GT(model.triangleCount, 0u);
    EXPECT_FALSE(model.primitives.empty());
    EXPECT_FALSE(model.nodes.empty());
    EXPECT_FALSE(model.materials.empty());
    EXPECT_FALSE(model.images.empty());
    EXPECT_FALSE(model.textures.empty());
    EXPECT_GE(model.materials[0].baseColorTexture, 0);
    EXPECT_LT(model.materials[0].baseColorTexture, static_cast<int>(model.textures.size()));
    EXPECT_GE(model.materials[0].metallicRoughnessTexture, 0);
    EXPECT_LT(model.materials[0].metallicRoughnessTexture, static_cast<int>(model.textures.size()));

    // DamagedHelmet's textures use the default sampler, which wraps (UVs range beyond [0, 1]).
    for (const mukki::GltfTexture& texture : model.textures)
    {
        EXPECT_EQ(texture.wrapS, mukki::GltfWrapMode::Repeat);
        EXPECT_EQ(texture.wrapT, mukki::GltfWrapMode::Repeat);
    }
}

TEST(GltfLoader, ParsesTransmissionMaterials)
{
    const std::filesystem::path modelPath =
        SampleModelsRoot() / "TransmissionTest/glTF/TransmissionTest.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "glTF sample models submodule is not initialized";
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));

    const bool hasTransmission = std::any_of(
        model.materials.begin(),
        model.materials.end(),
        [](const mukki::GltfMaterial& material) { return material.transmissionFactor > 0.0f; });
    EXPECT_TRUE(hasTransmission);

    for (const mukki::GltfMaterial& material : model.materials)
    {
        EXPECT_GT(material.ior, 1.0f);
    }
}

TEST(GltfLoader, ParsesVolumeAbsorption)
{
    const std::filesystem::path modelPath =
        SampleModelsRoot() / "DragonAttenuation/glTF/DragonAttenuation.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "glTF sample models submodule is not initialized";
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));

    const bool hasVolume = std::any_of(
        model.materials.begin(),
        model.materials.end(),
        [](const mukki::GltfMaterial& material) {
            return material.attenuationDistance > 0.0f &&
                   (material.attenuationColor.r < 1.0f || material.attenuationColor.g < 1.0f ||
                    material.attenuationColor.b < 1.0f);
        });
    EXPECT_TRUE(hasVolume);
}

TEST(GltfLoader, ParsesAlphaBlendMaterials)
{
    const std::filesystem::path modelPath =
        SampleModelsRoot() / "AlphaBlendModeTest/glTF/AlphaBlendModeTest.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "glTF sample models submodule is not initialized";
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));

    const bool hasBlend = std::any_of(
        model.materials.begin(),
        model.materials.end(),
        [](const mukki::GltfMaterial& material) { return material.alphaBlend; });
    EXPECT_TRUE(hasBlend);
}

TEST(GltfLoader, ParsesEmissiveMaterials)
{
    const std::filesystem::path modelPath = SampleModelsRoot() / "Lantern/glTF/Lantern.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "glTF sample models submodule is not initialized";
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));

    const bool hasEmissive = std::any_of(
        model.materials.begin(),
        model.materials.end(),
        [](const mukki::GltfMaterial& material) { return material.emissiveTexture >= 0; });
    EXPECT_TRUE(hasEmissive);

    for (const mukki::GltfMaterial& material : model.materials)
    {
        EXPECT_GE(material.emissiveFactor.r, 0.0f);
        EXPECT_GE(material.emissiveFactor.g, 0.0f);
        EXPECT_GE(material.emissiveFactor.b, 0.0f);
    }
}

TEST(GltfLoader, AppliesEmissiveStrength)
{
    const std::filesystem::path modelPath =
        SampleModelsRoot() / "EmissiveStrengthTest/glTF/EmissiveStrengthTest.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "glTF sample models submodule is not initialized";
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));
    ASSERT_FALSE(model.materials.empty());

    // The first material scales (0.1, 0.5, 0.9) by an emissive strength of 4.
    EXPECT_NEAR(model.materials[0].emissiveFactor.r, 0.4f, 1e-4f);
    EXPECT_NEAR(model.materials[0].emissiveFactor.g, 2.0f, 1e-4f);
    EXPECT_NEAR(model.materials[0].emissiveFactor.b, 3.6f, 1e-4f);
}

TEST(GltfLoader, ParsesSheenMaterials)
{
    const std::filesystem::path modelPath =
        SampleModelsRoot() / "SheenChair/glTF/SheenChair.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "glTF sample models submodule is not initialized";
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));

    const bool hasSheen = std::any_of(
        model.materials.begin(),
        model.materials.end(),
        [](const mukki::GltfMaterial& material) {
            return material.sheenColorFactor.r > 0.0f || material.sheenColorFactor.g > 0.0f ||
                   material.sheenColorFactor.b > 0.0f;
        });
    EXPECT_TRUE(hasSheen);
}

TEST(GltfLoader, LoadsNonAsciiPath)
{
    const std::filesystem::path modelsRoot = SampleModelsRoot();
    if (!std::filesystem::exists(modelsRoot))
    {
        GTEST_SKIP() << "glTF sample models submodule is not initialized";
    }

    std::filesystem::path nonAsciiGlb{};
    for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(modelsRoot))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".glb")
        {
            continue;
        }

        const std::string name = mukki::PathToUtf8(entry.path().filename());
        const bool hasNonAscii = std::any_of(name.begin(), name.end(), [](char character) {
            return static_cast<unsigned char>(character) > 127;
        });
        if (hasNonAscii)
        {
            nonAsciiGlb = entry.path();
            break;
        }
    }

    if (nonAsciiGlb.empty())
    {
        GTEST_SKIP() << "no non-ASCII sample model found";
    }

    EXPECT_FALSE(mukki::PathToUtf8(nonAsciiGlb).empty());

    mukki::GltfModel model{};
    EXPECT_TRUE(mukki::LoadGltfModel(nonAsciiGlb, model));
    EXPECT_GT(model.vertexCount, 0u);
}

TEST(GltfLoader, ParsesSkinsAndAnimations)
{
    const std::filesystem::path modelPath =
        SampleModelsRoot() / "RiggedSimple/glTF/RiggedSimple.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "glTF sample models submodule is not initialized";
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));

    ASSERT_FALSE(model.skins.empty());
    ASSERT_FALSE(model.animations.empty());

    // RiggedSimple's clip is anonymous; the loader assigns an indexed fallback name so the
    // empty string stays reserved as the "no clip" sentinel in the animation API.
    EXPECT_EQ(model.animations.front().name, "Clip 0");

    const mukki::GltfSkin& skin = model.skins.front();
    EXPECT_EQ(skin.joints.size(), skin.inverseBindMatrices.size());
    EXPECT_GT(skin.joints.size(), 0u);
    for (const int joint : skin.joints)
    {
        EXPECT_GE(joint, 0);
        EXPECT_LT(joint, static_cast<int>(model.nodes.size()));
    }

    const mukki::GltfAnimation& animation = model.animations.front();
    EXPECT_FALSE(animation.channels.empty());
    EXPECT_FALSE(animation.samplers.empty());
    EXPECT_GT(animation.duration, 0.0f);
    for (const mukki::GltfAnimationChannel& channel : animation.channels)
    {
        EXPECT_GE(channel.node, 0);
        EXPECT_LT(channel.node, static_cast<int>(model.nodes.size()));
        EXPECT_GE(channel.sampler, 0);
        EXPECT_LT(channel.sampler, static_cast<int>(animation.samplers.size()));
    }

    bool foundSkinned = false;
    for (const mukki::GltfPrimitive& primitive : model.primitives)
    {
        if (primitive.skinIndex < 0)
        {
            continue;
        }
        foundSkinned = true;
        for (const mukki::GltfVertex& vertex : primitive.vertices)
        {
            const unsigned weightSum = vertex.weights[0] + vertex.weights[1] +
                                       vertex.weights[2] + vertex.weights[3];
            EXPECT_GT(weightSum, 0u);
        }
    }
    EXPECT_TRUE(foundSkinned);

    // The hierarchy is retained: roots reference the scene and parents/children link.
    EXPECT_FALSE(model.rootNodes.empty());
    bool linked = false;
    for (const mukki::GltfNode& node : model.nodes)
    {
        if (node.parent >= 0 || !node.children.empty())
        {
            linked = true;
        }
    }
    EXPECT_TRUE(linked);
}

TEST(GltfLoader, KeepsTrsChannelsWhenMorphSamplersArePresent)
{
    // A morph (weights) sampler legally carries keyframe * morphTargetCount values; the loader
    // must not reject it as a count mismatch and must keep the TRS channels of the same clip.
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "mukki_gltf_loader_tests";
    std::error_code directoryError{};
    std::filesystem::create_directories(directory, directoryError);
    ASSERT_FALSE(directoryError);
    const std::filesystem::path jsonPath = directory / "mixed_animation.gltf";
    const std::filesystem::path binPath = directory / "mixed_animation.bin";

    // Buffer layout (little endian): positions (3 x vec3), indices (3 x u16), two morph target
    // position sets (3 x vec3 each), translation keys (2 x vec3), keyframe times (2 x f32) and
    // morph weights (2 keys x 2 morph targets).
    std::vector<std::uint8_t> bytes(164, 0);
    const auto putFloat = [&bytes](std::size_t offset, float value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    };
    const auto putU16 = [&bytes](std::size_t offset, std::uint16_t value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    };
    putFloat(0, 0.0f); putFloat(4, 0.0f); putFloat(8, 0.0f);
    putFloat(12, 1.0f); putFloat(16, 0.0f); putFloat(20, 0.0f);
    putFloat(24, 0.0f); putFloat(28, 1.0f); putFloat(32, 0.0f);
    putU16(36, 0u); putU16(38, 1u); putU16(40, 2u);
    putFloat(44, 0.1f);
    putFloat(80, 0.2f);
    putFloat(116, 0.0f); putFloat(120, 0.0f); putFloat(124, 0.0f);
    putFloat(128, 2.0f); putFloat(132, 0.0f); putFloat(136, 0.0f);
    putFloat(140, 0.0f); putFloat(144, 1.0f);
    putFloat(148, 0.0f); putFloat(152, 0.0f);
    putFloat(156, 1.0f); putFloat(160, 1.0f);
    {
        std::ofstream bin(binPath, std::ios::binary);
        ASSERT_TRUE(bin.good());
        bin.write(
            reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    }

    const std::string json = R"({
      "asset": {"version": "2.0"},
      "buffers": [{"uri": "mixed_animation.bin", "byteLength": 164}],
      "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
        {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963},
        {"buffer": 0, "byteOffset": 44, "byteLength": 36, "target": 34962},
        {"buffer": 0, "byteOffset": 80, "byteLength": 36, "target": 34962},
        {"buffer": 0, "byteOffset": 116, "byteLength": 24},
        {"buffer": 0, "byteOffset": 140, "byteLength": 8},
        {"buffer": 0, "byteOffset": 148, "byteLength": 16}
      ],
      "accessors": [
        {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
         "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
        {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3"},
        {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC3"},
        {"bufferView": 4, "componentType": 5126, "count": 2, "type": "VEC3"},
        {"bufferView": 5, "componentType": 5126, "count": 2, "type": "SCALAR"},
        {"bufferView": 6, "componentType": 5126, "count": 4, "type": "SCALAR"}
      ],
      "meshes": [{
        "primitives": [{
          "attributes": {"POSITION": 0},
          "indices": 1,
          "targets": [{"POSITION": 2}, {"POSITION": 3}]
        }]
      }],
      "nodes": [{"mesh": 0, "name": "Morphing"}],
      "scenes": [{"nodes": [0]}],
      "scene": 0,
      "animations": [{
        "name": "Mixed",
        "samplers": [
          {"input": 5, "output": 4, "interpolation": "LINEAR"},
          {"input": 5, "output": 6, "interpolation": "LINEAR"}
        ],
        "channels": [
          {"sampler": 0, "target": {"node": 0, "path": "translation"}},
          {"sampler": 1, "target": {"node": 0, "path": "weights"}}
        ]
      }]
    })";
    {
        std::ofstream jsonFile(jsonPath);
        ASSERT_TRUE(jsonFile.good());
        jsonFile << json;
    }

    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(jsonPath, model));

    std::filesystem::remove(jsonPath);
    std::filesystem::remove(binPath);

    // The clip survives with both channels: the morph sampler is valid (2 keys x 2 targets) and
    // the translation channel is intact.
    ASSERT_EQ(model.animations.size(), 1u);
    const mukki::GltfAnimation& clip = model.animations[0];
    EXPECT_EQ(clip.name, "Mixed");
    ASSERT_EQ(clip.channels.size(), 2u);
    EXPECT_EQ(clip.channels[0].path, mukki::GltfAnimationPath::Translation);
    EXPECT_EQ(clip.channels[0].sampler, 0);
    EXPECT_EQ(clip.channels[1].path, mukki::GltfAnimationPath::Weights);
    EXPECT_EQ(clip.channels[1].sampler, 1);
    EXPECT_EQ(clip.channels[1].weightCount, 2);
    ASSERT_EQ(clip.samplers.size(), 2u);
    EXPECT_EQ(clip.samplers[0].times.size(), 2u);
    EXPECT_EQ(clip.samplers[0].values.size(), 2u);
    EXPECT_EQ(clip.samplers[1].times.size(), 2u);
    ASSERT_EQ(clip.samplers[1].weights.size(), 4u);
    EXPECT_FLOAT_EQ(clip.samplers[1].weights[2], 1.0f);
    EXPECT_FLOAT_EQ(clip.samplers[1].weights[3], 1.0f);
    EXPECT_FLOAT_EQ(clip.duration, 1.0f);
}

TEST(GltfLoader, MorphClipsParseWithWeightSamplers)
{
    const std::filesystem::path modelPath =
        SampleModelsRoot() / "AnimatedMorphCube/glTF/AnimatedMorphCube.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "glTF sample models submodule is not initialized";
    }

    // The cube morphs between two states: 127 keys driving 2 morph targets per key (254 scalar
    // outputs). The clip must parse with its weights channel and the target deltas must load.
    mukki::GltfModel model{};
    ASSERT_TRUE(mukki::LoadGltfModel(modelPath, model));

    ASSERT_EQ(model.animations.size(), 1u);
    const mukki::GltfAnimation& clip = model.animations[0];
    ASSERT_EQ(clip.channels.size(), 1u);
    EXPECT_EQ(clip.channels[0].path, mukki::GltfAnimationPath::Weights);
    EXPECT_EQ(clip.channels[0].weightCount, 2);
    ASSERT_EQ(clip.samplers.size(), 1u);
    EXPECT_EQ(clip.samplers[0].times.size(), 127u);
    EXPECT_EQ(clip.samplers[0].weights.size(), 254u);

    ASSERT_FALSE(model.primitives.empty());
    const mukki::GltfPrimitive& primitive = model.primitives[0];
    ASSERT_EQ(primitive.morphTargets.size(), 2u);
    for (const mukki::GltfMorphTarget& target : primitive.morphTargets)
    {
        EXPECT_EQ(target.positionDeltas.size(), primitive.vertices.size());
        EXPECT_EQ(target.normalDeltas.size(), primitive.vertices.size());
    }
    // The cube authors explicit zero defaults.
    ASSERT_EQ(primitive.defaultWeights.size(), 2u);
    EXPECT_FLOAT_EQ(primitive.defaultWeights[0], 0.0f);
    EXPECT_FLOAT_EQ(primitive.defaultWeights[1], 0.0f);
    EXPECT_GT(model.vertexCount, 0u);
}

} // namespace
