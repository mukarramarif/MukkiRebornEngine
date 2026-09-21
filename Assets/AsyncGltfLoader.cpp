#include "Assets/AsyncGltfLoader.h"

#include "Core/JobSystem.h"
#include "Utilities/Log.h"
#include "Utilities/PathUtils.h"

#include <chrono>
#include <future>

namespace mukki
{

AsyncGltfLoader::AsyncGltfLoader(JobSystem& jobSystem) : m_jobSystem(&jobSystem) {}

void AsyncGltfLoader::Load(const std::filesystem::path& path)
{
    if (m_future.valid())
    {
        MRE_WARN(
            "AsyncGltfLoader replacing in-flight load of '{}' with '{}'",
            PathToUtf8(m_path),
            PathToUtf8(path));
    }

    m_path = path;

    if (m_jobSystem == nullptr)
    {
        GltfLoadResult result{};
        result.path = path;
        result.success = LoadGltfModel(path, result.model);

        std::promise<GltfLoadResult> promise{};
        promise.set_value(std::move(result));
        m_future = promise.get_future();
        return;
    }

    MRE_INFO("Queued async glTF load for '{}'", PathToUtf8(path));
    m_future = m_jobSystem->Submit([path]() {
        GltfLoadResult result{};
        result.path = path;
        result.success = LoadGltfModel(path, result.model);
        return result;
    });
}

void AsyncGltfLoader::Reset()
{
    m_future = std::future<GltfLoadResult>{};
    m_path.clear();
}

bool AsyncGltfLoader::IsReady() const
{
    return m_future.valid() && m_future.wait_for(std::chrono::seconds{0}) == std::future_status::ready;
}

GltfLoadResult AsyncGltfLoader::Consume()
{
    if (!IsReady())
    {
        return GltfLoadResult{};
    }

    GltfLoadResult result = m_future.get();
    m_future = std::future<GltfLoadResult>{};
    return result;
}

} // namespace mukki
