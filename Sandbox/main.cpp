#include "Application/Application.h"
#include "Application/Camera.h"
#include "Application/Input.h"
#include "Application/Keybinds.h"
#include "Application/Window.h"
#include "Renderer/RHI/Device.h"
#include "Renderer/Vulkan/Core/VkDevice.h"
#include "Utilities/Log.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>

#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>

namespace
{

class SandboxApp final : public mukki::Application
{
public:
    explicit SandboxApp(int maxFrames) : m_maxFrames(maxFrames) {}

protected:
    void OnStart() override
    {
        mukki::Keybinds& keybinds = GetKeybinds();
        keybinds.Bind("camera.forward", mukki::Binding::Key(GLFW_KEY_W));
        keybinds.Bind("camera.backward", mukki::Binding::Key(GLFW_KEY_S));
        keybinds.Bind("camera.left", mukki::Binding::Key(GLFW_KEY_A));
        keybinds.Bind("camera.right", mukki::Binding::Key(GLFW_KEY_D));
        keybinds.Bind("camera.up", mukki::Binding::Key(GLFW_KEY_E));
        keybinds.Bind("camera.down", mukki::Binding::Key(GLFW_KEY_Q));
        keybinds.Bind("camera.fast", mukki::Binding::Key(GLFW_KEY_LEFT_SHIFT));
        keybinds.Bind("app.quit", mukki::Binding::Key(GLFW_KEY_ESCAPE));

        GetInput().SetCursorCaptured(true);

        m_camera.SetPerspective(1.0471975512f, GetWindow().GetAspectRatio(), 0.1f, 1000.0f);
        m_camera.SetPosition(glm::vec3{0.0f, 0.0f, 3.0f});

        mukki::rhi::DeviceDesc deviceDesc{};
        deviceDesc.nativeWindowHandle = GetWindow().GetHandle();
        deviceDesc.applicationName = "Mukki Sandbox";
        deviceDesc.initialExtent = mukki::rhi::Extent2D{
            static_cast<std::uint32_t>(GetWindow().GetWidth()),
            static_cast<std::uint32_t>(GetWindow().GetHeight())};
        deviceDesc.enableValidation = true;

        m_device = mukki::vulkan::CreateVulkanDevice(deviceDesc);
        if (m_device == nullptr)
        {
            MRE_CRITICAL("Failed to create the Vulkan device");
            RequestExit();
        }
    }

    void OnUpdate(float deltaTime) override
    {
        if (m_device == nullptr)
        {
            RequestExit();
            return;
        }

        const mukki::Input& input = GetInput();
        const mukki::Keybinds& keybinds = GetKeybinds();

        if (keybinds.IsActionPressed("app.quit"))
        {
            RequestExit();
            return;
        }

        mukki::CameraInput cameraInput{};
        cameraInput.move.x = keybinds.GetAxis("camera.left", "camera.right");
        cameraInput.move.y = keybinds.GetAxis("camera.backward", "camera.forward");
        cameraInput.vertical = keybinds.GetAxis("camera.down", "camera.up");
        cameraInput.look = input.GetState().mouseDelta;
        cameraInput.fast = keybinds.IsActionDown("camera.fast");
        cameraInput.lookEnabled = true;

        m_controller.Update(m_camera, cameraInput, deltaTime);

        if (GetWindow().WasResized())
        {
            m_camera.SetPerspective(1.0471975512f, GetWindow().GetAspectRatio(), 0.1f, 1000.0f);
            m_device->Resize(mukki::rhi::Extent2D{
                static_cast<std::uint32_t>(GetWindow().GetWidth()),
                static_cast<std::uint32_t>(GetWindow().GetHeight())});
            GetWindow().ClearResizedFlag();
        }

        ++m_frameCount;
        if (m_maxFrames > 0 && m_frameCount >= m_maxFrames)
        {
            RequestExit();
        }
    }

    void OnRender() override
    {
        if (m_device == nullptr)
        {
            return;
        }

        if (!m_device->BeginFrame())
        {
            return;
        }

        mukki::rhi::CommandList& commandList = m_device->GetCommandList();
        commandList.Begin();

        const float phase = static_cast<float>(m_frameCount) * 0.02f;
        const mukki::rhi::ClearColor clearColor{
            0.05f + 0.03f * (1.0f + std::sin(phase)),
            0.07f,
            0.12f + 0.05f * (1.0f + std::cos(phase)),
            1.0f};

        commandList.BeginSwapchainRendering(clearColor);
        commandList.EndSwapchainRendering();
        commandList.End();

        m_device->EndFrame();
    }

    void OnShutdown() override
    {
        m_device.reset();
    }

private:
    mukki::Camera m_camera{};
    mukki::FlyCameraController m_controller{};
    std::unique_ptr<mukki::rhi::Device> m_device{};
    int m_maxFrames = 0;
    int m_frameCount = 0;
};

} // namespace

int main(int argc, char** argv)
{
    int maxFrames = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
        {
            maxFrames = std::atoi(argv[i + 1]);
        }
    }

    mukki::Log::Init(mukki::LogConfig{});

    int exitCode = EXIT_SUCCESS;
    try
    {
        SandboxApp app{maxFrames};
        if (!app.Init(mukki::WindowDesc{}))
        {
            MRE_CRITICAL("Failed to initialize the application");
            mukki::Log::Shutdown();
            return EXIT_FAILURE;
        }

        app.Run();
        app.Shutdown();
    }
    catch (const std::exception& exception)
    {
        MRE_CRITICAL("Unhandled exception: {}", exception.what());
        exitCode = EXIT_FAILURE;
    }

    mukki::Log::Shutdown();
    return exitCode;
}
