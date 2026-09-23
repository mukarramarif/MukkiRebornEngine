#include "Application/Application.h"

#include "Core/Time.h"
#include "Utilities/Assert.h"
#include "Utilities/Log.h"
#include "Utilities/Profiling.h"

#include <algorithm>

namespace mukki
{
namespace
{

constexpr float kMaxFrameTime = 0.1f;

} // namespace

bool Application::Init(const WindowDesc& desc)
{
    MRE_ASSERT(!m_initialized, "Application::Init called twice");

    if (!m_window.Init(desc))
    {
        return false;
    }

    if (!m_input.Init(m_window.GetHandle()))
    {
        m_window.Shutdown();
        return false;
    }

    m_initialized = true;
    OnStart();
    return true;
}

void Application::Run()
{
    MRE_ASSERT(m_initialized, "Application::Run called before Application::Init");

    MRE_PROFILE_SET_THREAD_NAME("MainThread");

    Clock clock{};
    m_running = true;

    while (m_running && !m_window.ShouldClose())
    {
        MRE_PROFILE_FRAME_N("Main");

        m_window.PollEvents();
        m_input.Update();
        m_keybinds.Update(m_input.GetState());

        const float deltaTime = std::min(clock.Tick(), kMaxFrameTime);
        {
            MRE_PROFILE_ZONE_N("Update");
            OnUpdate(deltaTime);
        }
        {
            MRE_PROFILE_ZONE_N("Render");
            OnRender();
        }
    }

    MRE_INFO("Application loop finished after {:.2f} seconds", clock.GetElapsedSeconds());
}

void Application::Shutdown()
{
    if (!m_initialized)
    {
        return;
    }

    OnShutdown();
    m_input.Shutdown();
    m_window.Shutdown();
    m_initialized = false;
}

} // namespace mukki
