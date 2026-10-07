#include "Assets/GltfLoader.h"
#include "Utilities/PathUtils.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

namespace
{

std::filesystem::path SampleModelsRoot()
{
    return std::filesystem::path{MRE_SOURCE_DIR} / "ThirdParty/glTF-Sample-Models/2.0";
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

} // namespace
