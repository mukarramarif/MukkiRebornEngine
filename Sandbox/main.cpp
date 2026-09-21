#include "Application/Application.h"
#include "Application/Camera.h"
#include "Application/Input.h"
#include "Application/Keybinds.h"
#include "Application/Window.h"
#include "Renderer/BufferManager.h"
#include "Renderer/RHI/Device.h"
#include "Renderer/RHI/Resources.h"
#include "Renderer/Vulkan/Core/VkDevice.h"
#include "Renderer/Vulkan/UiLayer.h"
#include "Utilities/Log.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <utility>

#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <imgui.h>

namespace
{

constexpr float kPi = 3.14159265358979f;

struct PushConstants
{
    glm::vec4 tint{1.0f};
    glm::vec4 params{0.0f};
};

struct Vertex
{
    glm::vec2 position{0.0f};
    glm::vec3 color{0.0f};
};

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
            return;
        }

        m_bufferManager = std::make_unique<mukki::BufferManager>(*m_device);

        m_vertexShader = m_device->CreateShader(
            mukki::rhi::ShaderStage::Vertex,
            (m_shaderDirectory / "triangle.vert.spv").string());
        m_fragmentShader = m_device->CreateShader(
            mukki::rhi::ShaderStage::Fragment,
            (m_shaderDirectory / "triangle.frag.spv").string());
        if (m_vertexShader == nullptr || m_fragmentShader == nullptr)
        {
            MRE_CRITICAL("Failed to load the triangle shaders from '{}'", m_shaderDirectory.string());
            RequestExit();
            return;
        }

        mukki::rhi::VertexLayout vertexLayout{};
        vertexLayout.AddAttribute(0, mukki::rhi::Format::R32G32Sfloat);
        vertexLayout.AddAttribute(1, mukki::rhi::Format::R32G32B32Sfloat);

        mukki::rhi::PipelineDesc pipelineDesc{};
        pipelineDesc.vertexShader = m_vertexShader.get();
        pipelineDesc.fragmentShader = m_fragmentShader.get();
        pipelineDesc.vertexLayout = vertexLayout;
        pipelineDesc.cullMode = mukki::rhi::CullMode::None;
        pipelineDesc.pushConstantSize = sizeof(PushConstants);
        pipelineDesc.colorFormats = {m_device->GetSwapchainInfo().format};
        m_pipeline = m_device->CreatePipeline(pipelineDesc);

        const Vertex vertices[] = {
            {{0.0f, 0.6f}, {1.0f, 0.2f, 0.2f}},
            {{0.5f, -0.5f}, {0.2f, 1.0f, 0.3f}},
            {{-0.5f, -0.5f}, {0.3f, 0.4f, 1.0f}},
        };
        m_vertexBuffer = &m_bufferManager->CreateVertexBuffer(vertices, sizeof(vertices));

        if (!m_ui.Init(*m_device, GetWindow().GetHandle()))
        {
            MRE_CRITICAL("Failed to initialize the ImGui layer");
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

        m_pushConstants.params.x = std::fmod(m_pushConstants.params.x + deltaTime * 0.8f, kPi * 2.0f);
        m_frameTimeMs = deltaTime * 1000.0f;

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

        m_ui.BeginFrame();
        ImGui::Begin("Mukki Sandbox");
        ImGui::Text("Frame time: %.2f ms", static_cast<double>(m_frameTimeMs));
        ImGui::Text(
            "Camera: %.2f, %.2f, %.2f",
            static_cast<double>(m_camera.GetPosition().x),
            static_cast<double>(m_camera.GetPosition().y),
            static_cast<double>(m_camera.GetPosition().z));
        ImGui::ColorEdit3("Tint", &m_pushConstants.tint.x);
        ImGui::Checkbox("Show demo window", &m_showDemoWindow);
        ImGui::End();
        if (m_showDemoWindow)
        {
            ImGui::ShowDemoWindow(&m_showDemoWindow);
        }

        const mukki::rhi::ClearColor clearColor{0.05f, 0.07f, 0.12f, 1.0f};

        commandList.BeginSwapchainRendering(clearColor, true);
        commandList.BindPipeline(*m_pipeline);
        commandList.BindVertexBuffer(*m_vertexBuffer);
        commandList.SetPushConstants(&m_pushConstants, sizeof(PushConstants), mukki::rhi::ShaderStageMask::Vertex);
        commandList.Draw(3);
        commandList.EndSwapchainRendering();

        commandList.BeginSwapchainRendering(clearColor, false);
        m_ui.EndFrame(commandList);
        commandList.EndSwapchainRendering();

        commandList.End();
        m_device->EndFrame();
    }

    void OnShutdown() override
    {
        m_ui.Shutdown();

        m_vertexBuffer = nullptr;
        m_pipeline.reset();
        m_vertexShader.reset();
        m_fragmentShader.reset();
        m_bufferManager.reset();
        m_device.reset();
    }

private:
    mukki::Camera m_camera{};
    mukki::FlyCameraController m_controller{};
    std::unique_ptr<mukki::rhi::Device> m_device{};
    std::unique_ptr<mukki::BufferManager> m_bufferManager{};
    std::unique_ptr<mukki::rhi::Shader> m_vertexShader{};
    std::unique_ptr<mukki::rhi::Shader> m_fragmentShader{};
    std::unique_ptr<mukki::rhi::Pipeline> m_pipeline{};
    mukki::rhi::Buffer* m_vertexBuffer = nullptr;
    mukki::vulkan::UiLayer m_ui{};
    PushConstants m_pushConstants{};
    std::filesystem::path m_shaderDirectory{};
    float m_frameTimeMs = 0.0f;
    int m_maxFrames = 0;
    int m_frameCount = 0;
    bool m_showDemoWindow = true;
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
