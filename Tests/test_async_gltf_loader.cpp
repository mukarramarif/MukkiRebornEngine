#include "Assets/AsyncGltfLoader.h"
#include "Core/JobSystem.h"

#include <chrono>
#include <filesystem>
#include <thread>

#include <gtest/gtest.h>

namespace
{

std::filesystem::path SampleModelsRoot()
{
    return std::filesystem::path{MRE_SOURCE_DIR} / "ThirdParty/glTF-Sample-Models/2.0";
}

} // namespace

TEST(AsyncGltfLoader, ReportsLoadingAndReadyStates)
{
    mukki::JobSystem jobSystem{2};
    mukki::AsyncGltfLoader loader{jobSystem};

    EXPECT_FALSE(loader.IsLoading());

    const std::filesystem::path modelPath = SampleModelsRoot() / "DamagedHelmet/glTF/DamagedHelmet.gltf";
    if (!std::filesystem::exists(modelPath))
    {
        GTEST_SKIP() << "glTF sample models submodule is not initialized";
    }

    loader.Load(modelPath);
    EXPECT_TRUE(loader.IsLoading());

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{30};
    while (!loader.IsReady() && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }

    ASSERT_TRUE(loader.IsReady());

    mukki::GltfLoadResult result = loader.Consume();
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.path, modelPath);
    EXPECT_GT(result.model.vertexCount, 0u);
    EXPECT_FALSE(loader.IsLoading());
}

TEST(AsyncGltfLoader, ConsumeWithoutReadyLoadReturnsEmptyResult)
{
    mukki::JobSystem jobSystem{1};
    mukki::AsyncGltfLoader loader{jobSystem};

    const mukki::GltfLoadResult result = loader.Consume();
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.model.vertexCount, 0u);
}
