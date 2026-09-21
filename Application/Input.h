#pragma once

#include <array>
#include <cstddef>

#include <glm/vec2.hpp>

struct GLFWwindow;

namespace mukki
{

inline constexpr std::size_t kKeyCount = 349;
inline constexpr std::size_t kMouseButtonCount = 8;

struct InputState
{
    std::array<bool, kKeyCount> keys{};
    std::array<bool, kMouseButtonCount> mouseButtons{};
    glm::vec2 mousePosition{0.0f};
    glm::vec2 mouseDelta{0.0f};
    float scrollDelta = 0.0f;

    [[nodiscard]] bool IsKeyDown(int key) const
    {
        return key >= 0 && static_cast<std::size_t>(key) < kKeyCount && keys[static_cast<std::size_t>(key)];
    }

    [[nodiscard]] bool IsMouseButtonDown(int button) const
    {
        return button >= 0 && static_cast<std::size_t>(button) < kMouseButtonCount &&
               mouseButtons[static_cast<std::size_t>(button)];
    }
};

class Input
{
public:
    Input() = default;
    ~Input();

    Input(const Input&) = delete;
    Input& operator=(const Input&) = delete;
    Input(Input&&) = delete;
    Input& operator=(Input&&) = delete;

    bool Init(GLFWwindow* window);
    void Shutdown();
    void Update();

    [[nodiscard]] const InputState& GetState() const { return m_state; }

    void SetCursorCaptured(bool captured);
    [[nodiscard]] bool IsCursorCaptured() const { return m_cursorCaptured; }

private:
    static void ScrollCallback(GLFWwindow* window, double xOffset, double yOffset);

    GLFWwindow* m_window = nullptr;
    InputState m_state{};
    float m_scrollAccumulator = 0.0f;
    bool m_cursorCaptured = false;
};

} // namespace mukki
