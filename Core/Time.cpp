#include "Core/Time.h"

namespace mukki
{

Clock::Clock()
{
    Reset();
}

void Clock::Reset()
{
    m_start = std::chrono::steady_clock::now();
    m_last = m_start;
}

float Clock::Tick()
{
    const TimePoint now = std::chrono::steady_clock::now();
    const std::chrono::duration<float> delta = now - m_last;
    m_last = now;
    return delta.count();
}

double Clock::GetElapsedSeconds() const
{
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - m_start;
    return elapsed.count();
}

} // namespace mukki
