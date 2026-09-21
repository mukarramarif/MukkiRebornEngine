#pragma once

#include <string>
#include <string_view>

struct GLFWwindow;

namespace mukki
{

struct WindowDesc
{
    std::string title = "MukkiRebornEngine";
    int width = 1280;
    int height = 720;
    bool resizable = true;
};

class Window
{
public:
    Window() = default;
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&&) = delete;
    Window& operator=(Window&&) = delete;

    bool Init(const WindowDesc& desc);
    void Shutdown();

    [[nodiscard]] bool IsOpen() const;
    [[nodiscard]] bool ShouldClose() const;
    void RequestClose() const;
    void PollEvents();
    void SetTitle(std::string_view title) const;

    [[nodiscard]] GLFWwindow* GetHandle() const { return m_window; }
    [[nodiscard]] int GetWidth() const { return m_width; }
    [[nodiscard]] int GetHeight() const { return m_height; }
    [[nodiscard]] float GetAspectRatio() const;
    [[nodiscard]] bool WasResized() const { return m_resized; }
    void ClearResizedFlag() { m_resized = false; }

private:
    void UpdateFramebufferSize();

    GLFWwindow* m_window = nullptr;
    int m_width = 0;
    int m_height = 0;
    bool m_resized = false;
    bool m_glfwInitialized = false;
};

} // namespace mukki
