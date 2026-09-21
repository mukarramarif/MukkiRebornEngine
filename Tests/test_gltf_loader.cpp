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
    EXPECT_GE(model.materials[0].baseColorImage, 0);
    EXPECT_LT(model.materials[0].baseColorImage, static_cast<int>(model.images.size()));
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
