#include "Application/Input.h"

#include "Utilities/Assert.h"

#include <GLFW/glfw3.h>

static_assert(mukki::kKeyCount == static_cast<std::size_t>(GLFW_KEY_LAST) + 1);
static_assert(mukki::kMouseButtonCount == static_cast<std::size_t>(GLFW_MOUSE_BUTTON_LAST) + 1);

namespace mukki
{

Input::~Input()
{
    Shutdown();
}

bool Input::Init(GLFWwindow* window)
{
    MRE_ASSERT(window != nullptr, "Input::Init requires a valid GLFW window");

    m_window = window;
    m_state = InputState{};
    m_scrollAccumulator = 0.0f;
    m_cursorCaptured = false;

    glfwSetWindowUserPointer(m_window, this);
    glfwSetScrollCallback(m_window, &Input::ScrollCallback);

    return true;
}

void Input::Shutdown()
{
    if (m_window != nullptr)
    {
        glfwSetScrollCallback(m_window, nullptr);
        glfwSetWindowUserPointer(m_window, nullptr);
        m_window = nullptr;
    }

    m_state = InputState{};
    m_scrollAccumulator = 0.0f;
    m_cursorCaptured = false;
}

void Input::Update()
{
    MRE_ASSERT(m_window != nullptr, "Input::Update called before Input::Init");

    InputState next{};

    for (std::size_t key = GLFW_KEY_SPACE; key <= static_cast<std::size_t>(GLFW_KEY_LAST); ++key)
    {
        next.keys[key] = glfwGetKey(m_window, static_cast<int>(key)) == GLFW_PRESS;
    }

    for (std::size_t button = 0; button < kMouseButtonCount; ++button)
    {
        next.mouseButtons[button] = glfwGetMouseButton(m_window, static_cast<int>(button)) == GLFW_PRESS;
    }

    double mouseX = 0.0;
    double mouseY = 0.0;
    glfwGetCursorPos(m_window, &mouseX, &mouseY);

    next.mousePosition = glm::vec2{static_cast<float>(mouseX), static_cast<float>(mouseY)};
    next.mouseDelta = next.mousePosition - m_state.mousePosition;
    next.scrollDelta = m_scrollAccumulator;

    m_scrollAccumulator = 0.0f;
    m_state = next;
}

void Input::SetCursorCaptured(bool captured)
{
    MRE_ASSERT(m_window != nullptr, "Input::SetCursorCaptured called before Input::Init");

    m_cursorCaptured = captured;
    glfwSetInputMode(m_window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
}

void Input::ScrollCallback(GLFWwindow* window, double xOffset, double yOffset)
{
    (void)xOffset;

    auto* input = static_cast<Input*>(glfwGetWindowUserPointer(window));
    if (input != nullptr)
    {
        input->m_scrollAccumulator += static_cast<float>(yOffset);
    }
}

} // namespace mukki
