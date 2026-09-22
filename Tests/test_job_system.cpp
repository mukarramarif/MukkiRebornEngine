#include "Core/JobSystem.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <future>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

TEST(JobSystem, ReportsAtLeastOneWorker)
{
    mukki::JobSystem jobSystem{2};
    EXPECT_EQ(jobSystem.GetThreadCount(), 2u);
}

TEST(JobSystem, RunsSubmittedJobsAndReturnsValues)
{
    mukki::JobSystem jobSystem{4};

    std::vector<std::future<int>> futures{};
    futures.reserve(32);
    for (int index = 0; index < 32; ++index)
    {
        futures.push_back(jobSystem.Submit([index]() { return index * 2; }));
    }

    for (int index = 0; index < 32; ++index)
    {
        EXPECT_EQ(futures[static_cast<std::size_t>(index)].get(), index * 2);
    }

    jobSystem.WaitIdle();
}

TEST(JobSystem, ExecutesJobsOnMultipleThreads)
{
    mukki::JobSystem jobSystem{4};

    std::atomic<std::uint32_t> distinctThreads{0};
    std::vector<std::thread::id> seen{};
    std::mutex seenMutex{};

    std::vector<std::future<void>> futures{};
    futures.reserve(16);
    for (int index = 0; index < 16; ++index)
    {
        futures.push_back(jobSystem.Submit([&distinctThreads, &seen, &seenMutex]() {
            const std::thread::id id = std::this_thread::get_id();
            std::lock_guard lock{seenMutex};
            if (std::find(seen.begin(), seen.end(), id) == seen.end())
            {
                seen.push_back(id);
                distinctThreads.fetch_add(1, std::memory_order_relaxed);
            }
        }));
    }

    for (std::future<void>& future : futures)
    {
        future.get();
    }

    EXPECT_GE(distinctThreads.load(), 1u);
    jobSystem.WaitIdle();
}

TEST(JobSystem, WaitIdleBlocksUntilAllWorkCompletes)
{
    mukki::JobSystem jobSystem{3};

    std::atomic<int> counter{0};
    for (int index = 0; index < 100; ++index)
    {
        static_cast<void>(
            jobSystem.Submit([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); }));
    }

    jobSystem.WaitIdle();
    EXPECT_EQ(counter.load(), 100);
}
