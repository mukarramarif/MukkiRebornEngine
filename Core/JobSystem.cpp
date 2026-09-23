#include "Core/JobSystem.h"

#include "Utilities/Assert.h"
#include "Utilities/Log.h"
#include "Utilities/Profiling.h"

#include <algorithm>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <processthreadsapi.h>
#endif

namespace mukki
{
namespace
{

void SetWorkerThreadName(std::uint32_t index)
{
    const std::string name = "Worker " + std::to_string(index);
    MRE_PROFILE_SET_THREAD_NAME(name.c_str());
    (void)name;

#ifdef _WIN32
    const std::wstring wideName = L"mukki worker " + std::to_wstring(index);
    SetThreadDescription(GetCurrentThread(), wideName.c_str());
#else
    (void)index;
#endif
}

} // namespace

JobSystem::JobSystem(std::uint32_t threadCount)
{
    if (threadCount == 0)
    {
        const std::uint32_t hardware = std::max(1u, std::thread::hardware_concurrency());
        threadCount = hardware > 1 ? hardware - 1 : 1;
    }

    m_workers.reserve(threadCount);
    for (std::uint32_t index = 0; index < threadCount; ++index)
    {
        m_workers.emplace_back([this, index]() {
            SetWorkerThreadName(index);
            WorkerLoop();
        });
    }

    MRE_INFO("Job system started with {} worker threads", threadCount);
}

JobSystem::~JobSystem()
{
    {
        std::lock_guard lock{m_mutex};
        m_stopping = true;
    }

    m_condition.notify_all();

    for (std::thread& worker : m_workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }

    MRE_INFO("Job system stopped");
}

void JobSystem::WaitIdle()
{
    std::unique_lock lock{m_mutex};
    m_idleCondition.wait(lock, [this]() { return m_jobs.empty() && m_activeJobs == 0; });
}

void JobSystem::WorkerLoop()
{
    for (;;)
    {
        std::function<void()> job{};
        {
            std::unique_lock lock{m_mutex};
            m_condition.wait(lock, [this]() { return m_stopping || !m_jobs.empty(); });

            if (m_jobs.empty())
            {
                return;
            }

            job = std::move(m_jobs.front());
            m_jobs.pop();
        }

        MRE_ASSERT(job != nullptr, "JobSystem dequeued an empty job");
        {
            MRE_PROFILE_ZONE_N("Job");
            job();
        }

        {
            std::lock_guard lock{m_mutex};
            MRE_ASSERT(m_activeJobs > 0, "JobSystem active job count underflow");
            --m_activeJobs;
        }
        m_idleCondition.notify_all();
    }
}

} // namespace mukki
