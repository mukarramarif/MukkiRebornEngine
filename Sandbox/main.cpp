#include "Application/Application.h"
#include "Application/Camera.h"
#include "Application/Input.h"
#include "Application/Keybinds.h"
#include "Application/Window.h"
#include "Renderer/Deferred/DeferredRenderer.h"
#include "Renderer/RHI/Device.h"
#include "Renderer/RHI/Resources.h"
#include "Renderer/Vulkan/Core/VkDevice.h"
#include "Renderer/Vulkan/UiLayer.h"
#include "Utilities/Log.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <utility>

#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>
#include <imgui.h>

namespace
{

void DrawTexturePreview(const char* label, ImTextureID textureId, float width, float aspect)
{
    if (textureId == 0)
    {
        return;
    }

    ImGui::SeparatorText(label);
    ImGui::Image(ImTextureRef{textureId}, ImVec2(width, width / aspect));
}

class SandboxApp final : public mukki::Application
{
public:
    SandboxApp(int maxFrames, std::filesystem::path shaderDirectory)
        : m_maxFrames(maxFrames)
        , m_shaderDirectory(std::move(shaderDirectory))
    {
    }

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
        keybinds.Bind("camera.look", mukki::Binding::Mouse(GLFW_MOUSE_BUTTON_RIGHT));
        keybinds.Bind("app.quit", mukki::Binding::Key(GLFW_KEY_ESCAPE));

        m_camera.SetPerspective(1.0471975512f, GetWindow().GetAspectRatio(), 0.1f, 1000.0f);
        m_camera.SetPosition(glm::vec3{6.0f, 3.0f, 7.0f});
        m_camera.SetRotation(-0.7f, -0.25f);

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
            return;
        }

        m_renderer = std::make_unique<mukki::DeferredRenderer>();
        mukki::DeferredRenderer::Desc rendererDesc{};
        rendererDesc.shaderDirectory = m_shaderDirectory;
        if (!m_renderer->Init(*m_device, rendererDesc))
        {
            MRE_CRITICAL("Failed to initialize the deferred renderer");
            RequestExit();
            return;
        }

        if (!m_ui.Init(*m_device, GetWindow().GetHandle()))
        {
            MRE_CRITICAL("Failed to initialize the ImGui layer");
            RequestExit();
        }
    }

    void OnUpdate(float deltaTime) override
    {
        if (m_device == nullptr || m_renderer == nullptr)
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

        const bool lookHeld = keybinds.IsActionDown("camera.look");
        if (input.IsCursorCaptured() != lookHeld)
        {
            GetInput().SetCursorCaptured(lookHeld);
        }

        mukki::CameraInput cameraInput{};
        cameraInput.move.x = keybinds.GetAxis("camera.left", "camera.right");
        cameraInput.move.y = keybinds.GetAxis("camera.backward", "camera.forward");
        cameraInput.vertical = keybinds.GetAxis("camera.down", "camera.up");
        cameraInput.look = input.GetState().mouseDelta;
        cameraInput.fast = keybinds.IsActionDown("camera.fast");
        cameraInput.lookEnabled = lookHeld;

        m_controller.Update(m_camera, cameraInput, deltaTime);
        m_frameTimeMs = deltaTime * 1000.0f;

        if (GetWindow().WasResized())
        {
            const std::uint32_t width = static_cast<std::uint32_t>(GetWindow().GetWidth());
            const std::uint32_t height = static_cast<std::uint32_t>(GetWindow().GetHeight());

            m_camera.SetPerspective(1.0471975512f, GetWindow().GetAspectRatio(), 0.1f, 1000.0f);
            m_device->Resize(mukki::rhi::Extent2D{width, height});
            m_renderer->Resize(width, height);
            ReleaseDebugTextures();
            m_renderedFrame = false;
            GetWindow().ClearResizedFlag();
        }

        if (!m_debugTexturesRegistered && m_renderedFrame)
        {
            m_albedoTextureId = m_ui.AddTexture(m_renderer->GetAlbedoTexture());
            m_normalTextureId = m_ui.AddTexture(m_renderer->GetNormalTexture());
            m_positionTextureId = m_ui.AddTexture(m_renderer->GetPositionTexture());
            m_debugTexturesRegistered = true;
        }

        ++m_frameCount;
        if (m_maxFrames > 0 && m_frameCount >= m_maxFrames)
        {
            RequestExit();
        }
    }

    void OnRender() override
    {
        if (m_device == nullptr || m_renderer == nullptr)
        {
            return;
        }

        if (!m_device->BeginFrame())
        {
            return;
        }

        mukki::rhi::CommandList& commandList = m_device->GetCommandList();
        commandList.Begin();

        m_ui.BeginFrame();
        ImGui::Begin("Mukki Sandbox");
        ImGui::Text("Frame time: %.2f ms", static_cast<double>(m_frameTimeMs));
        ImGui::Text(
            "Camera: %.2f, %.2f, %.2f",
            static_cast<double>(m_camera.GetPosition().x),
            static_cast<double>(m_camera.GetPosition().y),
            static_cast<double>(m_camera.GetPosition().z));
        ImGui::Checkbox("Show G-buffer", &m_showGBuffer);
        ImGui::Checkbox("Show demo window", &m_showDemoWindow);
        ImGui::End();

        if (m_showGBuffer)
        {
            ImGui::Begin("G-buffer", &m_showGBuffer);

            const float previewWidth = ImGui::GetContentRegionAvail().x;
            const float width = static_cast<float>(m_renderer->GetAlbedoTexture().GetWidth());
            const float height = static_cast<float>(m_renderer->GetAlbedoTexture().GetHeight());
            const float aspect = height > 0.0f ? width / height : 1.0f;

            DrawTexturePreview("Albedo (metalness in A)", m_albedoTextureId, previewWidth, aspect);
            DrawTexturePreview("Normal (roughness in A)", m_normalTextureId, previewWidth, aspect);
            DrawTexturePreview("Position", m_positionTextureId, previewWidth, aspect);
            ImGui::End();
        }

        if (m_showDemoWindow)
        {
            ImGui::ShowDemoWindow(&m_showDemoWindow);
        }

        m_renderer->Render(commandList, m_camera.GetViewProjectionMatrix(), m_camera.GetPosition());

        commandList.BeginSwapchainRendering(mukki::rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f}, false);
        m_ui.EndFrame(commandList);
        commandList.EndSwapchainRendering();

        commandList.End();
        m_device->EndFrame();
        m_renderedFrame = true;
    }

    void OnShutdown() override
    {
        ReleaseDebugTextures();
        m_ui.Shutdown();
        m_renderer.reset();
        m_device.reset();
    }

private:
    void ReleaseDebugTextures()
    {
        if (!m_debugTexturesRegistered)
        {
            return;
        }

        m_ui.RemoveTexture(m_albedoTextureId);
        m_ui.RemoveTexture(m_normalTextureId);
        m_ui.RemoveTexture(m_positionTextureId);
        m_albedoTextureId = 0;
        m_normalTextureId = 0;
        m_positionTextureId = 0;
        m_debugTexturesRegistered = false;
    }

    mukki::Camera m_camera{};
    mukki::FlyCameraController m_controller{};
    std::unique_ptr<mukki::rhi::Device> m_device{};
    std::unique_ptr<mukki::DeferredRenderer> m_renderer{};
    mukki::vulkan::UiLayer m_ui{};
    ImTextureID m_albedoTextureId = 0;
    ImTextureID m_normalTextureId = 0;
    ImTextureID m_positionTextureId = 0;
    std::filesystem::path m_shaderDirectory{};
    float m_frameTimeMs = 0.0f;
    int m_maxFrames = 0;
    int m_frameCount = 0;
    bool m_debugTexturesRegistered = false;
    bool m_renderedFrame = false;
    bool m_showGBuffer = true;
    bool m_showDemoWindow = false;
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
        const std::filesystem::path shaderDirectory =
            std::filesystem::absolute(argv[0]).parent_path() / "shaders";

        SandboxApp app{maxFrames, shaderDirectory};
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
