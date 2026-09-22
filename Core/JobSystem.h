#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace mukki
{

class JobSystem
{
public:
    explicit JobSystem(std::uint32_t threadCount = 0);
    ~JobSystem();

    JobSystem(const JobSystem&) = delete;
    JobSystem& operator=(const JobSystem&) = delete;

    template <typename Function, typename... Args>
    [[nodiscard]] auto Submit(Function&& function, Args&&... args)
        -> std::future<std::invoke_result_t<Function, Args...>>;

    [[nodiscard]] std::uint32_t GetThreadCount() const { return static_cast<std::uint32_t>(m_workers.size()); }

    void WaitIdle();

private:
    void WorkerLoop();

    std::vector<std::thread> m_workers{};
    std::queue<std::function<void()>> m_jobs{};
    std::mutex m_mutex{};
    std::condition_variable m_condition{};
    std::condition_variable m_idleCondition{};
    std::size_t m_activeJobs = 0;
    bool m_stopping = false;
};

template <typename Function, typename... Args>
auto JobSystem::Submit(Function&& function, Args&&... args)
    -> std::future<std::invoke_result_t<Function, Args...>>
{
    using ReturnType = std::invoke_result_t<Function, Args...>;

    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        [function = std::forward<Function>(function), ... captured = std::forward<Args>(args)]() mutable -> ReturnType {
            return function(captured...);
        });
    std::future<ReturnType> future = task->get_future();

    {
        std::lock_guard lock{m_mutex};
        m_jobs.emplace([task]() { (*task)(); });
        ++m_activeJobs;
    }

    m_condition.notify_one();
    return future;
}

} // namespace mukki
