#pragma once

#include <chrono>

namespace mukki
{

class Clock
{
public:
    Clock();

    void Reset();
    float Tick();
    [[nodiscard]] double GetElapsedSeconds() const;

private:
    using TimePoint = std::chrono::steady_clock::time_point;

    TimePoint m_start{};
    TimePoint m_last{};
};

} // namespace mukki
