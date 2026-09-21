#include "Application/Application.h"
#include "Application/Camera.h"
#include "Application/Input.h"
#include "Application/Keybinds.h"
#include "Application/Window.h"
#include "Core/JobSystem.h"
#include "Renderer/Deferred/DeferredRenderer.h"
#include "Renderer/RHI/Device.h"
#include "Renderer/RHI/Resources.h"
#include "Renderer/Vulkan/Core/VkDevice.h"
#include "Renderer/Vulkan/UiLayer.h"
#include "Utilities/Log.h"
#include "Utilities/PathUtils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
    SandboxApp(int maxFrames, std::filesystem::path shaderDirectory, std::filesystem::path requestedModel)
        : m_maxFrames(maxFrames)
        , m_shaderDirectory(std::move(shaderDirectory))
        , m_requestedModel(std::move(requestedModel))
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
        keybinds.Bind("camera.freeCursor", mukki::Binding::Key(GLFW_KEY_LEFT_ALT));
        keybinds.Bind("camera.toggleCursor", mukki::Binding::Key(GLFW_KEY_TAB));
        keybinds.Bind("app.quit", mukki::Binding::Key(GLFW_KEY_ESCAPE));

        m_camera.SetPerspective(1.0471975512f, GetWindow().GetAspectRatio(), 0.1f, 1000.0f);
        m_camera.SetPosition(glm::vec3{4.0f, 2.5f, 5.0f});
        m_camera.SetRotation(-0.65f, -0.3f);

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
        rendererDesc.jobSystem = &m_jobSystem;
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
            return;
        }

        ScanSampleModels();
        if (!m_requestedModel.empty())
        {
            m_pendingModelPath = m_requestedModel;
        }
        else
        {
            LoadModelByIndex(m_selectedModel);
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

        if (keybinds.IsActionPressed("camera.toggleCursor"))
        {
            m_cursorReleased = !m_cursorReleased;
        }

        const bool freeCursor = m_cursorReleased || keybinds.IsActionDown("camera.freeCursor");
        const bool lookEnabled = !freeCursor;
        if (input.IsCursorCaptured() != lookEnabled)
        {
            GetInput().SetCursorCaptured(lookEnabled);
        }

        mukki::CameraInput cameraInput{};
        cameraInput.move.x = keybinds.GetAxis("camera.left", "camera.right");
        cameraInput.move.y = keybinds.GetAxis("camera.backward", "camera.forward");
        cameraInput.vertical = keybinds.GetAxis("camera.down", "camera.up");
        cameraInput.look = input.GetState().mouseDelta;
        cameraInput.fast = keybinds.IsActionDown("camera.fast");
        cameraInput.lookEnabled = lookEnabled;

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
        DrawStatsWindow();
        DrawModelBrowser();
        DrawGBufferWindow();
        if (m_showDemoWindow)
        {
            ImGui::ShowDemoWindow(&m_showDemoWindow);
        }

        if (!m_pendingModelPath.empty())
        {
            const std::filesystem::path path = std::move(m_pendingModelPath);
            m_pendingModelPath.clear();
            m_renderer->RequestModelLoad(path);
        }

        m_renderer->PollModelLoad();

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
    void ScanSampleModels()
    {
#ifdef MRE_SAMPLE_MODELS_DIR
        m_modelsRoot = std::filesystem::path{MRE_SAMPLE_MODELS_DIR};
        if (!std::filesystem::exists(m_modelsRoot))
        {
            MRE_WARN("Sample models directory '{}' does not exist", mukki::PathToUtf8(m_modelsRoot));
            return;
        }

        for (const std::filesystem::directory_entry& entry :
             std::filesystem::recursive_directory_iterator(m_modelsRoot))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }

            const std::filesystem::path& path = entry.path();
            const std::string extension = mukki::PathToUtf8(path.extension());
            if (extension != ".gltf" && extension != ".glb")
            {
                continue;
            }

            const std::string variant = mukki::PathToUtf8(path.parent_path().filename());
            if (variant == "glTF-Embedded")
            {
                continue;
            }

            m_sampleModels.push_back(path);
        }

        std::sort(m_sampleModels.begin(), m_sampleModels.end());

        for (std::size_t index = 0; index < m_sampleModels.size(); ++index)
        {
            if (mukki::PathToUtf8(m_sampleModels[index].filename()).find("DamagedHelmet") != std::string::npos)
            {
                m_selectedModel = static_cast<int>(index);
                break;
            }
        }

        MRE_INFO("Found {} sample models in '{}'", m_sampleModels.size(), mukki::PathToUtf8(m_modelsRoot));
#else
        MRE_WARN("Sample models directory was not configured at build time");
#endif
    }

    void LoadModelByIndex(int index)
    {
        if (index < 0 || index >= static_cast<int>(m_sampleModels.size()))
        {
            return;
        }

        m_pendingModelPath = m_sampleModels[static_cast<std::size_t>(index)];
    }

    void DrawStatsWindow()
    {
        ImGui::Begin("Mukki Sandbox");
        ImGui::Text("Frame time: %.2f ms", static_cast<double>(m_frameTimeMs));
        ImGui::Text(
            "Camera: %.2f, %.2f, %.2f",
            static_cast<double>(m_camera.GetPosition().x),
            static_cast<double>(m_camera.GetPosition().y),
            static_cast<double>(m_camera.GetPosition().z));
        ImGui::Checkbox("Show G-buffer", &m_showGBuffer);
        ImGui::Checkbox("Show demo window", &m_showDemoWindow);
        ImGui::SeparatorText("Controls");
        ImGui::TextUnformatted("WASD/QE move, Shift fast");
        ImGui::TextUnformatted("Mouse looks around");
        ImGui::TextUnformatted("Tab toggles the cursor, Left Alt holds it free");
        ImGui::TextUnformatted("Esc quits");
        ImGui::End();
    }

    void DrawModelBrowser()
    {
        ImGui::Begin("Model browser");

        if (m_sampleModels.empty())
        {
            ImGui::TextUnformatted("No sample models available.");
            ImGui::TextUnformatted("Initialize the ThirdParty/glTF-Sample-Models submodule and reconfigure.");
            ImGui::End();
            return;
        }

        const std::string current = m_renderer->HasModel() ? m_renderer->GetModelName() : "none";
        ImGui::Text("Loaded: %s", current.c_str());
        if (m_renderer->IsModelLoading())
        {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4{1.0f, 0.8f, 0.2f, 1.0f}, "(loading...)");
        }

        const std::string selectedName = mukki::PathToUtf8(
            m_sampleModels[static_cast<std::size_t>(m_selectedModel)].filename());
        if (ImGui::BeginCombo("Model", selectedName.c_str()))
        {
            for (std::size_t index = 0; index < m_sampleModels.size(); ++index)
            {
                std::error_code relativeError{};
                const std::filesystem::path relativePath =
                    std::filesystem::relative(m_sampleModels[index], m_modelsRoot, relativeError);
                const std::string relative =
                    relativeError ? mukki::PathToUtf8(m_sampleModels[index]) : mukki::PathToUtf8(relativePath);
                const bool selected = static_cast<int>(index) == m_selectedModel;
                if (ImGui::Selectable(relative.c_str(), selected))
                {
                    m_selectedModel = static_cast<int>(index);
                    LoadModelByIndex(m_selectedModel);
                }
                if (selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        if (ImGui::Button("Reload"))
        {
            LoadModelByIndex(m_selectedModel);
        }
        ImGui::SameLine();
        if (ImGui::Button("Unload"))
        {
            m_renderer->UnloadModel();
        }

        ImGui::Separator();
        ImGui::Text("Vertices: %llu", static_cast<unsigned long long>(m_renderer->GetModelVertexCount()));
        ImGui::Text("Triangles: %llu", static_cast<unsigned long long>(m_renderer->GetModelTriangleCount()));
        ImGui::Text("Models found: %zu", m_sampleModels.size());

        ImGui::End();
    }

    void DrawGBufferWindow()
    {
        if (!m_showGBuffer)
        {
            return;
        }

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
    mukki::JobSystem m_jobSystem{};
    std::unique_ptr<mukki::rhi::Device> m_device{};
    std::unique_ptr<mukki::DeferredRenderer> m_renderer{};
    mukki::vulkan::UiLayer m_ui{};
    std::vector<std::filesystem::path> m_sampleModels{};
    std::filesystem::path m_modelsRoot{};
    std::filesystem::path m_pendingModelPath{};
    ImTextureID m_albedoTextureId = 0;
    ImTextureID m_normalTextureId = 0;
    ImTextureID m_positionTextureId = 0;
    std::filesystem::path m_shaderDirectory{};
    std::filesystem::path m_requestedModel{};
    float m_frameTimeMs = 0.0f;
    int m_selectedModel = 0;
    int m_maxFrames = 0;
    int m_frameCount = 0;
    bool m_debugTexturesRegistered = false;
    bool m_renderedFrame = false;
    bool m_cursorReleased = false;
    bool m_showGBuffer = false;
    bool m_showDemoWindow = false;
};

} // namespace

int main(int argc, char** argv)
{
    int maxFrames = 0;
    std::filesystem::path requestedModel{};
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
        {
            maxFrames = std::atoi(argv[i + 1]);
        }
        else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
        {
            requestedModel = argv[i + 1];
        }
    }

    mukki::Log::Init(mukki::LogConfig{});

    int exitCode = EXIT_SUCCESS;
    try
    {
        const std::filesystem::path shaderDirectory =
            std::filesystem::absolute(argv[0]).parent_path() / "shaders";

        SandboxApp app{maxFrames, shaderDirectory, requestedModel};
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
