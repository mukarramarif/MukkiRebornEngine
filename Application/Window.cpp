#include "Application/Window.h"

#include "Utilities/Assert.h"
#include "Utilities/Log.h"

#include <GLFW/glfw3.h>

namespace mukki
{
namespace
{

void OnGlfwError(int errorCode, const char* description)
{
    MRE_ERROR("GLFW error {}: {}", errorCode, description);
}

} // namespace

Window::~Window()
{
    Shutdown();
}

bool Window::Init(const WindowDesc& desc)
{
    MRE_ASSERT(!m_glfwInitialized, "Window::Init called twice");

    glfwSetErrorCallback(OnGlfwError);

    if (glfwInit() == GLFW_FALSE)
    {
        MRE_ERROR("Failed to initialize GLFW");
        return false;
    }

    m_glfwInitialized = true;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, desc.resizable ? GLFW_TRUE : GLFW_FALSE);

    m_window = glfwCreateWindow(desc.width, desc.height, desc.title.c_str(), nullptr, nullptr);
    if (m_window == nullptr)
    {
        MRE_ERROR("Failed to create a {}x{} window", desc.width, desc.height);
        glfwTerminate();
        m_glfwInitialized = false;
        return false;
    }

    UpdateFramebufferSize();

    MRE_INFO("Created window '{}' ({}x{})", desc.title, m_width, m_height);
    return true;
}

void Window::Shutdown()
{
    if (m_window != nullptr)
    {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }

    if (m_glfwInitialized)
    {
        glfwTerminate();
        m_glfwInitialized = false;
    }

    m_width = 0;
    m_height = 0;
    m_resized = false;
}

bool Window::IsOpen() const
{
    return m_window != nullptr;
}

bool Window::ShouldClose() const
{
    return m_window == nullptr || glfwWindowShouldClose(m_window) == GLFW_TRUE;
}

void Window::RequestClose() const
{
    if (m_window != nullptr)
    {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }
}

void Window::PollEvents()
{
    glfwPollEvents();
    UpdateFramebufferSize();
}

void Window::SetTitle(std::string_view title) const
{
    if (m_window != nullptr)
    {
        glfwSetWindowTitle(m_window, std::string{title}.c_str());
    }
}

float Window::GetAspectRatio() const
{
    if (m_height <= 0)
    {
        return 1.0f;
    }
    return static_cast<float>(m_width) / static_cast<float>(m_height);
}

void Window::UpdateFramebufferSize()
{
    if (m_window == nullptr)
    {
        return;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(m_window, &width, &height);

    if (width != m_width || height != m_height)
    {
        m_width = width;
        m_height = height;
        m_resized = true;
    }
}

} // namespace mukki
