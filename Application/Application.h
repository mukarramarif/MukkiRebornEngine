#pragma once

#include "Application/Input.h"
#include "Application/Keybinds.h"
#include "Application/Window.h"

namespace mukki
{

class Application
{
public:
    Application() = default;
    virtual ~Application() = default;

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    Application(Application&&) = delete;
    Application& operator=(Application&&) = delete;

    bool Init(const WindowDesc& desc);
    void Run();
    void Shutdown();

    void RequestExit() { m_running = false; }

    [[nodiscard]] Window& GetWindow() { return m_window; }
    [[nodiscard]] const Window& GetWindow() const { return m_window; }
    [[nodiscard]] Input& GetInput() { return m_input; }
    [[nodiscard]] const Input& GetInput() const { return m_input; }
    [[nodiscard]] Keybinds& GetKeybinds() { return m_keybinds; }
    [[nodiscard]] const Keybinds& GetKeybinds() const { return m_keybinds; }

protected:
    virtual void OnStart() {}
    virtual void OnUpdate(float deltaTime) { (void)deltaTime; }
    virtual void OnRender() {}
    virtual void OnShutdown() {}

private:
    Window m_window{};
    Input m_input{};
    Keybinds m_keybinds{};
    bool m_running = false;
    bool m_initialized = false;
};

} // namespace mukki
