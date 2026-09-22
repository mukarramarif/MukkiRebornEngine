#pragma once

#include <filesystem>
#include <future>

#include "Assets/GltfLoader.h"

namespace mukki
{

class JobSystem;

struct GltfLoadResult
{
    bool success = false;
    std::filesystem::path path{};
    GltfModel model{};
};

class AsyncGltfLoader
{
public:
    AsyncGltfLoader() = default;
    explicit AsyncGltfLoader(JobSystem& jobSystem);
    ~AsyncGltfLoader() = default;

    AsyncGltfLoader(const AsyncGltfLoader&) = delete;
    AsyncGltfLoader& operator=(const AsyncGltfLoader&) = delete;

    void SetJobSystem(JobSystem& jobSystem) { m_jobSystem = &jobSystem; }

    void Load(const std::filesystem::path& path);
    void Reset();

    [[nodiscard]] bool IsLoading() const { return m_future.valid(); }
    [[nodiscard]] bool IsReady() const;
    [[nodiscard]] const std::filesystem::path& GetRequestedPath() const { return m_path; }

    [[nodiscard]] GltfLoadResult Consume();

private:
    JobSystem* m_jobSystem = nullptr;
    std::future<GltfLoadResult> m_future{};
    std::filesystem::path m_path{};
};

} // namespace mukki
