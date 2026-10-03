#include "Application/Application.h"
#include "Application/Camera.h"
#include "Application/Input.h"
#include "Application/Keybinds.h"
#include "Application/Window.h"
#include "Core/JobSystem.h"
#include "Renderer/Deferred/DeferredRenderer.h"
#include "Renderer/Lighting/Light.h"
#include "Renderer/RHI/Device.h"
#include "Renderer/RHI/Resources.h"
#include "Renderer/Vulkan/Core/VkDevice.h"
#include "Renderer/Vulkan/UiLayer.h"
#include "Scene/SceneBinary.h"
#include "Scene/SceneDescription.h"
#include "Scene/SceneJson.h"
#include "Utilities/Log.h"
#include "Utilities/PathUtils.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <GLFW/glfw3.h>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec3.hpp>
#include <imgui.h>

namespace
{

const char* LightTypeLabel(mukki::LightType type)
{
    switch (type)
    {
    case mukki::LightType::Directional:
        return "directional";
    case mukki::LightType::Point:
        return "point";
    case mukki::LightType::Spot:
        return "spot";
    }
    return "point";
}

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
    SandboxApp(
        int maxFrames,
        std::filesystem::path shaderDirectory,
        std::filesystem::path requestedModel,
        std::filesystem::path requestedScene)
        : m_maxFrames(maxFrames)
        , m_shaderDirectory(std::move(shaderDirectory))
        , m_requestedModel(std::move(requestedModel))
        , m_requestedScene(std::move(requestedScene))
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

        ScanSceneFiles();

        if (!m_requestedModel.empty())
        {
            m_pendingModelPath = m_requestedModel;

            // Standalone models get a small default light rig and one probe so they are visible.
            m_scene.lights.clear();
            m_scene.probes.clear();

            mukki::SceneLight key{};
            key.name = "Key";
            key.light.type = mukki::LightType::Directional;
            key.light.direction = glm::vec3{-0.4f, -1.0f, -0.3f};
            key.light.color = glm::vec3{1.0f, 0.98f, 0.95f};
            key.light.intensity = 3.0f;
            m_scene.lights.push_back(key);

            mukki::SceneLight fill{};
            fill.name = "Fill";
            fill.light.type = mukki::LightType::Point;
            fill.light.position = glm::vec3{1.5f, 1.5f, 1.5f};
            fill.light.color = glm::vec3{0.6f, 0.8f, 1.0f};
            fill.light.intensity = 3.0f;
            fill.light.range = 8.0f;
            m_scene.lights.push_back(fill);

            mukki::SceneLight rim{};
            rim.name = "Rim";
            rim.light.type = mukki::LightType::Point;
            rim.light.position = glm::vec3{-1.5f, 1.0f, -1.5f};
            rim.light.color = glm::vec3{1.0f, 0.7f, 0.4f};
            rim.light.intensity = 2.0f;
            rim.light.range = 8.0f;
            m_scene.lights.push_back(rim);

            mukki::SceneProbe probe{};
            probe.name = "Probe";
            probe.position = glm::vec3{0.0f, 1.0f, 0.0f};
            probe.radius = 0.4f;
            m_scene.probes.push_back(probe);

            ApplySceneLights();
        }
        else
        {
            const std::filesystem::path scenePath =
                m_requestedScene.empty() ? DefaultSceneJsonPath() : m_requestedScene;
            if (!scenePath.empty())
            {
                LoadSceneFromJson(scenePath);
            }
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
            m_motionTextureId = m_ui.AddTexture(m_renderer->GetMotionTexture());
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
        DrawSceneBrowser();
        DrawSceneWindow();
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
        ApplySceneProbes();

        mukki::FrameView view{};
        view.view = m_camera.GetViewMatrix();
        view.projection = m_camera.GetProjectionMatrix();
        view.viewProjection = m_camera.GetViewProjectionMatrix();
        view.previousViewProjection = m_previousViewProjection;
        view.position = m_camera.GetPosition();
        view.fovYRadians = m_camera.GetFovY();
        view.aspectRatio = m_camera.GetAspectRatio();
        view.nearPlane = m_camera.GetNearPlane();
        view.farPlane = m_camera.GetFarPlane();

        m_renderer->Render(commandList, view);
        m_previousViewProjection = view.viewProjection;

        commandList.BeginGpuZone("ImGui");
        commandList.BeginSwapchainRendering(mukki::rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f}, false);
        m_ui.EndFrame(commandList);
        commandList.EndSwapchainRendering();
        commandList.EndGpuZone();

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
    void ScanSceneFiles()
    {
#ifdef MRE_SCENES_DIR
        m_scenesRoot = std::filesystem::path{MRE_SCENES_DIR};
        if (!std::filesystem::exists(m_scenesRoot))
        {
            MRE_WARN("Scenes directory '{}' does not exist", mukki::PathToUtf8(m_scenesRoot));
            return;
        }

        for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(m_scenesRoot))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            if (mukki::PathToUtf8(entry.path().extension()) != ".json")
            {
                continue;
            }

            m_sceneFiles.push_back(entry.path());
        }

        std::sort(m_sceneFiles.begin(), m_sceneFiles.end());

        for (std::size_t index = 0; index < m_sceneFiles.size(); ++index)
        {
            if (mukki::PathToUtf8(m_sceneFiles[index].filename()).find("demo") != std::string::npos)
            {
                m_sceneFileIndex = static_cast<int>(index);
                break;
            }
        }

        MRE_INFO("Found {} scene files in '{}'", m_sceneFiles.size(), mukki::PathToUtf8(m_scenesRoot));
#else
        MRE_WARN("Scenes directory was not configured at build time");
#endif
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

    void DrawSceneBrowser()
    {
        ImGui::SetNextWindowPos(ImVec2(560.0f, 20.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340.0f, 0.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Scenes");

        ImGui::Text("Loaded: %s", m_scene.name.empty() ? "none" : m_scene.name.c_str());
        if (m_renderer->IsModelLoading())
        {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4{1.0f, 0.8f, 0.2f, 1.0f}, "(loading model...)");
        }

        if (m_sceneFiles.empty())
        {
            ImGui::TextUnformatted("No scene files found.");
            ImGui::TextUnformatted("Add .json scenes to Sandbox/Scenes and reconfigure.");
        }
        else
        {
            const std::size_t selectedIndex = static_cast<std::size_t>(
                std::clamp(m_sceneFileIndex, 0, static_cast<int>(m_sceneFiles.size()) - 1));
            const std::string selectedName = mukki::PathToUtf8(m_sceneFiles[selectedIndex].filename());
            if (ImGui::BeginCombo("Scene", selectedName.c_str()))
            {
                for (std::size_t index = 0; index < m_sceneFiles.size(); ++index)
                {
                    const bool selected = index == selectedIndex;
                    const std::string name = mukki::PathToUtf8(m_sceneFiles[index].filename());
                    if (ImGui::Selectable(name.c_str(), selected))
                    {
                        m_sceneFileIndex = static_cast<int>(index);
                        LoadSceneFromJson(m_sceneFiles[index]);
                    }
                    if (selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            if (ImGui::Button("Load"))
            {
                LoadSceneFromJson(m_sceneFiles[selectedIndex]);
            }
            ImGui::SameLine();
            if (ImGui::Button("Reload"))
            {
                LoadSceneFromJson(
                    m_sceneSourcePath.empty() ? m_sceneFiles[selectedIndex] : m_sceneSourcePath);
            }
            ImGui::SameLine();
            if (ImGui::Button("Unload scene"))
            {
                m_renderer->ClearScene();
                m_scene = mukki::SceneDescription{};
                m_sceneSourcePath.clear();
                m_sceneStatus = "scene unloaded";
            }
        }

        ImGui::SeparatorText("Transform");
        float modelScale = m_renderer->GetModelScale();
        if (ImGui::SliderFloat("Uniform scale", &modelScale, 0.1f, 10.0f, "%.2f"))
        {
            m_renderer->SetModelScale(modelScale);
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset"))
        {
            m_renderer->SetModelScale(1.0f);
        }

        ImGui::Separator();
        ImGui::Text("Objects: %zu   Lights: %zu", m_scene.objects.size(), m_scene.lights.size());
        ImGui::Text("Vertices: %llu", static_cast<unsigned long long>(m_renderer->GetModelVertexCount()));
        ImGui::Text("Triangles: %llu", static_cast<unsigned long long>(m_renderer->GetModelTriangleCount()));

        ImGui::End();
    }

    std::filesystem::path DefaultSceneJsonPath() const
    {
#ifdef MRE_SCENES_DIR
        return std::filesystem::path{MRE_SCENES_DIR} / "demo.scene.json";
#else
        return {};
#endif
    }

    // Compiles the JSON source to bytecode next to the executable, then loads the bytecode.
    bool LoadSceneFromJson(const std::filesystem::path& jsonPath)
    {
        if (!std::filesystem::exists(jsonPath))
        {
            m_sceneStatus = "scene JSON not found: " + mukki::PathToUtf8(jsonPath);
            MRE_WARN("{}", m_sceneStatus);
            return false;
        }

        m_sceneSourcePath = jsonPath;
        m_sceneBinaryDirectory = m_shaderDirectory.parent_path() / "scenes";
        std::error_code directoryError{};
        std::filesystem::create_directories(m_sceneBinaryDirectory, directoryError);
        const std::filesystem::path binaryPath =
            m_sceneBinaryDirectory / (mukki::PathToUtf8(jsonPath.stem()) + ".mres");

        std::string error{};
        mukki::SceneDescription authored{};
        if (!mukki::SceneJson::Load(jsonPath, authored, error))
        {
            m_sceneStatus = "JSON: " + error;
            MRE_ERROR("{}", m_sceneStatus);
            return false;
        }

        // The JSON is the authoring format; the engine only ever loads the compiled bytecode.
        if (!mukki::SceneBinary::Write(binaryPath, authored, error))
        {
            m_sceneStatus = "compile: " + error;
            MRE_ERROR("{}", m_sceneStatus);
            return false;
        }

        mukki::SceneDescription loaded{};
        if (!mukki::SceneBinary::Read(binaryPath, loaded, error))
        {
            m_sceneStatus = "bytecode: " + error;
            MRE_ERROR("{}", m_sceneStatus);
            return false;
        }

        m_scene = std::move(loaded);
        m_selectedLight = 0;
        m_selectedObject = 0;
        if (!m_renderer->LoadScene(m_scene))
        {
            m_sceneStatus = "the renderer failed to load the scene";
            return false;
        }

        m_sceneLightsDirty = false;
        m_sceneStatus = "loaded '" + m_scene.name + "' (" + std::to_string(m_scene.objects.size()) +
                        " objects, " + std::to_string(m_scene.lights.size()) + " lights)";
        MRE_INFO("{}", m_sceneStatus);
        return true;
    }

    std::string SceneOutputStem() const
    {
        return m_sceneSourcePath.empty() ? std::string{"demo.scene"}
                                         : mukki::PathToUtf8(m_sceneSourcePath.stem());
    }

    void SaveSceneJson()
    {
        if (m_sceneBinaryDirectory.empty())
        {
            m_sceneBinaryDirectory = m_shaderDirectory.parent_path() / "scenes";
        }
        std::error_code directoryError{};
        std::filesystem::create_directories(m_sceneBinaryDirectory, directoryError);
        const std::filesystem::path path = m_sceneBinaryDirectory / (SceneOutputStem() + ".json");

        std::string error{};
        if (mukki::SceneJson::Save(path, m_scene, error))
        {
            m_sceneStatus = "saved " + mukki::PathToUtf8(path);
        }
        else
        {
            m_sceneStatus = error;
            MRE_ERROR("{}", error);
        }
    }

    void CompileSceneBinary()
    {
        if (m_sceneBinaryDirectory.empty())
        {
            m_sceneBinaryDirectory = m_shaderDirectory.parent_path() / "scenes";
        }
        std::error_code directoryError{};
        std::filesystem::create_directories(m_sceneBinaryDirectory, directoryError);
        const std::filesystem::path path = m_sceneBinaryDirectory / (SceneOutputStem() + ".mres");

        std::string error{};
        if (mukki::SceneBinary::Write(path, m_scene, error))
        {
            m_sceneStatus = "compiled " + mukki::PathToUtf8(path);
        }
        else
        {
            m_sceneStatus = error;
            MRE_ERROR("{}", error);
        }
    }

    void ApplySceneLights()
    {
        std::vector<mukki::Light> lights{};
        lights.reserve(m_scene.lights.size());
        for (const mukki::SceneLight& sceneLight : m_scene.lights)
        {
            lights.push_back(sceneLight.light);
        }
        m_renderer->SetLights(lights);
    }

    void ApplySceneProbes()
    {
        std::vector<mukki::GiProbe> probes{};
        probes.reserve(m_scene.probes.size());
        for (const mukki::SceneProbe& sceneProbe : m_scene.probes)
        {
            if (!sceneProbe.enabled)
            {
                continue;
            }

            mukki::GiProbe probe{};
            probe.position = sceneProbe.position;
            probe.radius = std::max(0.05f, sceneProbe.radius);
            probes.push_back(probe);
        }
        m_renderer->SetGiProbes(probes);
    }

    void DrawSceneWindow()
    {
        ImGui::Begin("Scene");

        ImGui::Text("Scene: %s", m_scene.name.empty() ? "(none)" : m_scene.name.c_str());
        ImGui::Text(
            "Objects: %zu   Lights: %zu   In renderer: %u",
            m_scene.objects.size(),
            m_scene.lights.size(),
            m_renderer->GetLightCount());
        if (!m_sceneStatus.empty())
        {
            ImGui::TextWrapped("%s", m_sceneStatus.c_str());
        }

        if (ImGui::Button("Reload JSON -> bytecode"))
        {
            LoadSceneFromJson(
                m_sceneSourcePath.empty() ? DefaultSceneJsonPath() : m_sceneSourcePath);
        }
        ImGui::SameLine();
        if (ImGui::Button("Save JSON"))
        {
            SaveSceneJson();
        }
        ImGui::SameLine();
        if (ImGui::Button("Compile bytecode"))
        {
            CompileSceneBinary();
        }

        bool debugClusters = m_renderer->IsClusterDebugViewEnabled();
        if (ImGui::Checkbox("Visualize cluster light counts", &debugClusters))
        {
            m_renderer->SetClusterDebugView(debugClusters);
        }

        bool showLightGizmos = m_renderer->AreLightGizmosEnabled();
        bool showProbeGizmos = m_renderer->AreProbeGizmosEnabled();
        if (ImGui::Checkbox("Show light gizmos", &showLightGizmos))
        {
            m_renderer->SetGizmoView(showLightGizmos, showProbeGizmos);
        }
        ImGui::SameLine();
        if (ImGui::Checkbox("Show probe gizmos", &showProbeGizmos))
        {
            m_renderer->SetGizmoView(showLightGizmos, showProbeGizmos);
        }

        ImGui::SeparatorText("Objects");
        if (ImGui::BeginChild("##object_list", ImVec2(0.0f, 90.0f), ImGuiChildFlags_Borders))
        {
            for (std::size_t index = 0; index < m_scene.objects.size(); ++index)
            {
                const mukki::SceneObject& object = m_scene.objects[index];
                char label[256];
                std::snprintf(
                    label,
                    sizeof(label),
                    "%s (%s)",
                    object.name.c_str(),
                    object.modelPath.c_str());
                if (ImGui::Selectable(label, m_selectedObject == static_cast<int>(index)))
                {
                    m_selectedObject = static_cast<int>(index);
                }
            }
        }
        ImGui::EndChild();

        if (m_selectedObject >= 0 && m_selectedObject < static_cast<int>(m_scene.objects.size()))
        {
            DrawObjectEditor(m_scene.objects[static_cast<std::size_t>(m_selectedObject)]);
        }

        DrawProbeSection();

        ImGui::SeparatorText("Lights");
        if (ImGui::BeginChild("##light_list", ImVec2(0.0f, 110.0f), ImGuiChildFlags_Borders))
        {
            for (std::size_t index = 0; index < m_scene.lights.size(); ++index)
            {
                const mukki::SceneLight& sceneLight = m_scene.lights[index];
                char label[192];
                std::snprintf(
                    label,
                    sizeof(label),
                    "%s (%s)",
                    sceneLight.name.c_str(),
                    LightTypeLabel(sceneLight.light.type));
                if (ImGui::Selectable(label, m_selectedLight == static_cast<int>(index)))
                {
                    m_selectedLight = static_cast<int>(index);
                }
            }
        }
        ImGui::EndChild();

        if (ImGui::Button("Add light"))
        {
            mukki::SceneLight sceneLight{};
            sceneLight.name = "Light " + std::to_string(m_scene.lights.size());
            sceneLight.light.type = mukki::LightType::Point;
            sceneLight.light.position = glm::vec3{0.0f, 2.0f, 0.0f};
            sceneLight.light.range = 5.0f;
            sceneLight.light.intensity = 1.0f;
            m_scene.lights.push_back(sceneLight);
            m_selectedLight = static_cast<int>(m_scene.lights.size()) - 1;
            m_sceneLightsDirty = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Remove light") && !m_scene.lights.empty())
        {
            const std::size_t index = static_cast<std::size_t>(
                std::clamp(m_selectedLight, 0, static_cast<int>(m_scene.lights.size()) - 1));
            m_scene.lights.erase(m_scene.lights.begin() + static_cast<std::ptrdiff_t>(index));
            m_selectedLight = std::min(m_selectedLight, static_cast<int>(m_scene.lights.size()) - 1);
            m_sceneLightsDirty = true;
        }

        if (m_selectedLight >= 0 && m_selectedLight < static_cast<int>(m_scene.lights.size()))
        {
            DrawLightEditor(m_scene.lights[static_cast<std::size_t>(m_selectedLight)]);
        }

        if (m_sceneLightsDirty)
        {
            ApplySceneLights();
            m_sceneLightsDirty = false;
        }

        DrawShadowSettings();
        DrawGiSettings();
        DrawReflectionSettings();

        ImGui::End();
    }

    void DrawReflectionSettings()
    {
        ImGui::SeparatorText("Ray traced reflections");

        mukki::ReflectionSettings settings = m_renderer->GetReflectionSettings();
        bool changed = ImGui::Checkbox("Reflections", &settings.enabled);
        if (!m_device->IsRayTracingSupported())
        {
            ImGui::SameLine();
            ImGui::TextDisabled("(ray tracing unavailable)");
        }

        changed |= ImGui::SliderFloat(
            "Reflection intensity",
            &settings.intensity,
            0.0f,
            4.0f,
            "%.2f");
        changed |= ImGui::SliderFloat(
            "Reflection max roughness",
            &settings.maxRoughness,
            0.0f,
            1.0f,
            "%.2f");
        changed |= ImGui::SliderFloat(
            "Reflection distance",
            &settings.maxDistance,
            1.0f,
            300.0f,
            "%.0f");
        int samples = static_cast<int>(settings.samples);
        if (ImGui::SliderInt("Reflection samples", &samples, 1, 8))
        {
            settings.samples = static_cast<std::uint32_t>(samples);
            changed = true;
        }
        if (ImGui::SliderFloat(
                "Reflection trace scale",
                &settings.resolutionScale,
                0.25f,
                1.0f,
                "%.2f"))
        {
            changed = true;
        }
        ImGui::SetItemTooltip(
            "Fraction of the swapchain resolution used for the reflection trace.\n"
            "1.0 keeps mirror reflections sharp, lower values trade sharpness for performance.");

        ImGui::SeparatorText("Glass");
        changed |= ImGui::Checkbox("Dispersion", &settings.dispersion);
        int bounces = static_cast<int>(settings.refractionBounces);
        if (ImGui::SliderInt("Refraction bounces", &bounces, 1, 8))
        {
            settings.refractionBounces = static_cast<std::uint32_t>(bounces);
            changed = true;
        }
        changed |= ImGui::SliderFloat(
            "Volume absorption",
            &settings.volumeAbsorptionScale,
            0.0f,
            3.0f,
            "%.2f");

        if (changed)
        {
            m_renderer->SetReflectionSettings(settings);
        }
    }

    static float RadicalInverse(int index, int base)
    {
        float result = 0.0f;
        float fraction = 1.0f;
        while (index > 0)
        {
            fraction /= static_cast<float>(base);
            result += fraction * static_cast<float>(index % base);
            index /= base;
        }
        return result;
    }

    void FillSceneProbes(int count)
    {
        glm::vec3 boundsMin{0.0f};
        glm::vec3 boundsMax{0.0f};
        if (!m_renderer->GetSceneBounds(boundsMin, boundsMax))
        {
            m_sceneStatus = "probe fill needs a loaded scene";
            return;
        }

        const int probeCount = std::clamp(count, 1, static_cast<int>(mukki::kMaxGiProbes));
        const glm::vec3 inset{0.05f};
        m_scene.probes.clear();
        for (int index = 1; index <= probeCount; ++index)
        {
            const glm::vec3 t{
                RadicalInverse(index, 2),
                RadicalInverse(index, 3),
                RadicalInverse(index, 5)};
            mukki::SceneProbe probe{};
            probe.name = "Auto " + std::to_string(index);
            probe.position =
                glm::mix(boundsMin, boundsMax, inset + t * (glm::vec3{1.0f} - inset * 2.0f));
            probe.radius = 0.35f;
            m_scene.probes.push_back(probe);
        }

        m_selectedProbe = m_scene.probes.empty() ? -1 : 0;
        ApplySceneProbes();
        m_sceneStatus = "scene probes: " + std::to_string(m_scene.probes.size());
    }

    void DrawProbeSection()
    {
        ImGui::SeparatorText("Probes");

        int enabledCount = 0;
        for (const mukki::SceneProbe& probe : m_scene.probes)
        {
            if (probe.enabled)
            {
                ++enabledCount;
            }
        }
        ImGui::Text(
            "%zu placed, %d enabled (max %u on the GPU)",
            m_scene.probes.size(),
            enabledCount,
            mukki::kMaxGiProbes);

        if (ImGui::BeginChild("##probe_list", ImVec2(0.0f, 100.0f), ImGuiChildFlags_Borders))
        {
            for (std::size_t index = 0; index < m_scene.probes.size(); ++index)
            {
                const mukki::SceneProbe& probe = m_scene.probes[index];
                char label[192];
                std::snprintf(
                    label,
                    sizeof(label),
                    "%s%s (%.1f, %.1f, %.1f)",
                    probe.enabled ? "" : "[off] ",
                    probe.name.c_str(),
                    probe.position.x,
                    probe.position.y,
                    probe.position.z);
                if (ImGui::Selectable(label, m_selectedProbe == static_cast<int>(index)))
                {
                    m_selectedProbe = static_cast<int>(index);
                }
            }
        }
        ImGui::EndChild();

        if (ImGui::Button("Add probe"))
        {
            mukki::SceneProbe probe{};
            probe.name = "Probe " + std::to_string(m_scene.probes.size());
            const glm::mat4 view = m_camera.GetViewMatrix();
            const glm::vec3 forward = -glm::vec3{view[2]};
            probe.position = m_camera.GetPosition() + forward * 4.0f;
            probe.radius = 0.4f;
            m_scene.probes.push_back(probe);
            m_selectedProbe = static_cast<int>(m_scene.probes.size()) - 1;
            ApplySceneProbes();
        }
        ImGui::SameLine();
        if (ImGui::Button("Add grid"))
        {
            constexpr int kColumns = 3;
            constexpr int kRows = 2;
            constexpr int kLayers = 3;
            constexpr float kSpacing = 4.0f;
            for (int layer = 0; layer < kLayers; ++layer)
            {
                for (int row = 0; row < kRows; ++row)
                {
                    for (int column = 0; column < kColumns; ++column)
                    {
                        mukki::SceneProbe probe{};
                        probe.name = "Grid " + std::to_string(layer) + "," + std::to_string(row) +
                                     "," + std::to_string(column);
                        probe.position = glm::vec3{
                            (static_cast<float>(column) - 1.0f) * kSpacing,
                            1.5f + static_cast<float>(row) * 2.5f,
                            (static_cast<float>(layer) - 1.0f) * kSpacing};
                        probe.radius = 0.35f;
                        m_scene.probes.push_back(probe);
                    }
                }
            }
            ApplySceneProbes();
            m_sceneStatus = "probe grid added";
        }
        ImGui::SameLine();
        if (ImGui::Button("Remove probe") && !m_scene.probes.empty())
        {
            const std::size_t index = static_cast<std::size_t>(
                std::clamp(m_selectedProbe, 0, static_cast<int>(m_scene.probes.size()) - 1));
            m_scene.probes.erase(m_scene.probes.begin() + static_cast<std::ptrdiff_t>(index));
            m_selectedProbe = std::min(m_selectedProbe, static_cast<int>(m_scene.probes.size()) - 1);
            ApplySceneProbes();
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear probes") && !m_scene.probes.empty())
        {
            m_scene.probes.clear();
            m_selectedProbe = -1;
            ApplySceneProbes();
        }

        // Scene fill: distributes probes over the loaded scene bounds (Halton sequence) so the
        // whole scene can be covered without placing probes by hand. "Maximize" places the GPU
        // limit in one click.
        glm::vec3 boundsMin{0.0f};
        glm::vec3 boundsMax{0.0f};
        if (m_renderer->GetSceneBounds(boundsMin, boundsMax))
        {
            const glm::vec3 extent = boundsMax - boundsMin;
            ImGui::Text("Scene extent %.1f x %.1f x %.1f", extent.x, extent.y, extent.z);
        }
        else
        {
            ImGui::TextDisabled("Scene extent: no loaded objects");
        }
        ImGui::SetNextItemWidth(130.0f);
        ImGui::SliderInt(
            "##probe_fill_count",
            &m_probeFillCount,
            1,
            static_cast<int>(mukki::kMaxGiProbes),
            "Probes: %d");
        ImGui::SameLine();
        if (ImGui::Button("Fill scene"))
        {
            FillSceneProbes(m_probeFillCount);
        }
        ImGui::SameLine();
        if (ImGui::Button("Maximize probes"))
        {
            m_probeFillCount = static_cast<int>(mukki::kMaxGiProbes);
            FillSceneProbes(m_probeFillCount);
        }
        ImGui::SetItemTooltip(
            "Replaces the probe list with probes distributed across the loaded scene bounds,\n"
            "up to the GPU limit (max %u).",
            mukki::kMaxGiProbes);

        if (m_selectedProbe >= 0 && m_selectedProbe < static_cast<int>(m_scene.probes.size()))
        {
            mukki::SceneProbe& probe = m_scene.probes[static_cast<std::size_t>(m_selectedProbe)];

            char nameBuffer[128];
            std::snprintf(nameBuffer, sizeof(nameBuffer), "%s", probe.name.c_str());
            if (ImGui::InputText("Probe name", nameBuffer, sizeof(nameBuffer)))
            {
                probe.name = nameBuffer;
            }

            bool changed = false;
            changed |= ImGui::DragFloat3("Probe position", &probe.position.x, 0.05f, -200.0f, 200.0f);
            changed |= ImGui::SliderFloat("Probe radius", &probe.radius, 0.05f, 2.0f, "%.2f");
            changed |= ImGui::Checkbox("Probe enabled", &probe.enabled);
            if (changed)
            {
                ApplySceneProbes();
            }
        }
    }

    void DrawShadowSettings()
    {
        ImGui::SeparatorText("Shadows");

        mukki::ShadowSettings settings = m_renderer->GetShadowSettings();
        bool changed = ImGui::Checkbox("Enabled", &settings.enabled);

        changed |= ImGui::SliderFloat("Distance", &settings.distance, 5.0f, 500.0f, "%.1f");
        changed |= ImGui::SliderFloat("Depth bias", &settings.depthBias, 0.0f, 5.0f, "%.2f");
        changed |= ImGui::SliderFloat("Normal bias", &settings.normalBias, 0.0f, 1.0f, "%.3f");

        changed |= ImGui::Checkbox("Ray traced sun shadows", &settings.rtSunShadows);
        if (settings.rtSunShadows && !m_device->IsRayTracingSupported())
        {
            ImGui::SameLine();
            ImGui::TextDisabled("(ray tracing unavailable)");
        }
        int rtSamples = static_cast<int>(settings.rtShadowSamples);
        if (ImGui::SliderInt("RT shadow samples", &rtSamples, 1, 16))
        {
            settings.rtShadowSamples = static_cast<std::uint32_t>(rtSamples);
            changed = true;
        }
        changed |= ImGui::SliderFloat(
            "RT sun angular radius (deg)",
            &settings.rtSunAngularRadiusDegrees,
            0.0f,
            3.0f,
            "%.2f");

        ImGui::Text(
            "%u cascades at %u, %u spot maps, %u point atlases",
            static_cast<unsigned>(settings.cascadeCount),
            static_cast<unsigned>(settings.resolution),
            static_cast<unsigned>(mukki::kMaxSpotShadowLights),
            static_cast<unsigned>(mukki::kMaxPointShadowLights));

        if (changed)
        {
            m_renderer->SetShadowSettings(settings);
        }
    }

    void DrawGiSettings()
    {
        ImGui::SeparatorText("Global illumination");

        mukki::GiSettings settings = m_renderer->GetGiSettings();
        bool changed = ImGui::Checkbox("Ray traced GI", &settings.enabled);

        if (!m_device->IsRayTracingSupported())
        {
            ImGui::SameLine();
            ImGui::TextDisabled("(ray tracing unavailable)");
        }

        changed |= ImGui::SliderFloat("Trace scale", &settings.resolutionScale, 0.25f, 1.0f, "%.2f");
        int traceSamples = static_cast<int>(settings.traceSamples);
        if (ImGui::SliderInt("Trace samples", &traceSamples, 1, 4))
        {
            settings.traceSamples = static_cast<std::uint32_t>(traceSamples);
            changed = true;
        }
        changed |= ImGui::SliderFloat("GI intensity", &settings.intensity, 0.0f, 4.0f, "%.2f");

        changed |= ImGui::Checkbox("Use probes for GI", &settings.useProbes);
        ImGui::SameLine();
        ImGui::TextDisabled("(probe objects)");
        int probeRays = static_cast<int>(settings.probeRays);
        if (ImGui::SliderInt("Probe rays", &probeRays, 16, 256))
        {
            settings.probeRays = static_cast<std::uint32_t>(probeRays);
            changed = true;
        }
        changed |= ImGui::SliderFloat(
            "Probe hysteresis",
            &settings.probeHysteresis,
            0.0f,
            0.98f,
            "%.2f");
        changed |= ImGui::SliderFloat("Max ray distance", &settings.maxDistance, 1.0f, 200.0f, "%.0f");
        changed |= ImGui::SliderFloat("Bounce feedback", &settings.feedbackIntensity, 0.0f, 1.5f, "%.2f");
        changed |= ImGui::SliderFloat(
            "Specular min roughness",
            &settings.specularMinRoughness,
            0.0f,
            0.5f,
            "%.3f");
        changed |= ImGui::SliderFloat("Temporal alpha min", &settings.temporalAlphaMin, 0.005f, 0.5f, "%.3f");
        changed |= ImGui::SliderFloat("Phi depth", &settings.phiDepth, 0.1f, 10.0f, "%.2f");
        changed |= ImGui::SliderFloat("Phi normal", &settings.phiNormal, 1.0f, 128.0f, "%.1f");
        changed |= ImGui::SliderFloat("Phi luminance", &settings.phiLuminance, 0.5f, 16.0f, "%.2f");
        changed |= ImGui::SliderFloat("Phi hit distance", &settings.phiHitDistance, 0.5f, 16.0f, "%.2f");

        int iterations = static_cast<int>(settings.atrousIterations);
        if (ImGui::SliderInt("À-trous iterations", &iterations, 1, 8))
        {
            settings.atrousIterations = static_cast<std::uint32_t>(iterations);
            changed = true;
        }
        changed |= ImGui::Checkbox("Sun shadow ray", &settings.sunShadowRay);

        int sunSamples = static_cast<int>(settings.sunShadowSamples);
        if (ImGui::SliderInt("Sun shadow samples", &sunSamples, 1, 16))
        {
            settings.sunShadowSamples = static_cast<std::uint32_t>(sunSamples);
            changed = true;
        }
        changed |= ImGui::SliderFloat(
            "Sun angular radius (deg)",
            &settings.sunAngularRadiusDegrees,
            0.0f,
            3.0f,
            "%.2f");

        if (changed)
        {
            m_renderer->SetGiSettings(settings);
        }
    }

    void DrawObjectEditor(mukki::SceneObject& object)
    {
        const std::size_t index = static_cast<std::size_t>(std::max(m_selectedObject, 0));
        ImGui::Text("Model: %s", object.modelPath.c_str());

        bool changed = false;
        changed |= ImGui::DragFloat3("Object position", &object.position.x, 0.05f, -100.0f, 100.0f);
        changed |= ImGui::DragFloat3("Object rotation (deg)", &object.rotation.x, 0.5f, -360.0f, 360.0f);
        changed |= ImGui::DragFloat3("Object scale", &object.scale.x, 0.02f, 0.01f, 100.0f);

        if (changed)
        {
            if (index < m_renderer->GetSceneObjectCount())
            {
                m_renderer->UpdateSceneObjectTransform(index, mukki::BuildObjectTransform(object));
                m_sceneStatus = "object '" + object.name + "' updated";
            }
            else
            {
                m_sceneStatus = "object is not instantiated in the renderer";
            }
        }

    }

    void DrawLightEditor(mukki::SceneLight& sceneLight)
    {
        mukki::Light& light = sceneLight.light;

        char nameBuffer[128];
        std::snprintf(nameBuffer, sizeof(nameBuffer), "%s", sceneLight.name.c_str());
        if (ImGui::InputText("Name", nameBuffer, sizeof(nameBuffer)))
        {
            sceneLight.name = nameBuffer;
            m_sceneLightsDirty = true;
        }

        const char* typeNames[] = {"Directional", "Point", "Spot"};
        int type = static_cast<int>(light.type);
        if (ImGui::Combo("Type", &type, typeNames, 3))
        {
            light.type = static_cast<mukki::LightType>(type);
            m_sceneLightsDirty = true;
        }

        if (light.type != mukki::LightType::Directional)
        {
            if (ImGui::DragFloat3("Position", &light.position.x, 0.05f, -100.0f, 100.0f))
            {
                m_sceneLightsDirty = true;
            }
        }

        if (light.type != mukki::LightType::Point)
        {
            // Keep the raw vector so dragging one component does not renormalize the others; the
            // renderer normalizes directions when it uploads lights and builds shadow matrices.
            if (ImGui::DragFloat3("Direction", &light.direction.x, 0.01f, -1.0f, 1.0f))
            {
                m_sceneLightsDirty = true;
            }
        }

        if (ImGui::Checkbox("Casts shadows", &light.castShadows))
        {
            m_sceneLightsDirty = true;
        }
        switch (light.type)
        {
        case mukki::LightType::Directional:
            ImGui::TextUnformatted("(only the first shadow-casting directional light is used)");
            break;
        case mukki::LightType::Spot:
            ImGui::Text("(up to %u spot shadow maps)", static_cast<unsigned>(mukki::kMaxSpotShadowLights));
            break;
        case mukki::LightType::Point:
            ImGui::Text(
                "(up to %u point shadow slots, six faces each)",
                static_cast<unsigned>(mukki::kMaxPointShadowLights));
            break;
        }

        if (ImGui::ColorEdit3("Color", &light.color.x))
        {
            m_sceneLightsDirty = true;
        }

        if (ImGui::DragFloat("Strength", &light.intensity, 0.05f, 0.0f, 100.0f))
        {
            m_sceneLightsDirty = true;
        }

        if (light.type != mukki::LightType::Directional)
        {
            if (ImGui::DragFloat("Range", &light.range, 0.1f, 0.1f, 200.0f))
            {
                m_sceneLightsDirty = true;
            }
        }

        if (light.type == mukki::LightType::Spot)
        {
            float innerDegrees = glm::degrees(light.innerConeAngle);
            float outerDegrees = glm::degrees(light.outerConeAngle);
            bool changed = false;
            changed |= ImGui::DragFloat("Inner cone (deg)", &innerDegrees, 0.5f, 0.0f, 89.0f);
            changed |= ImGui::DragFloat("Outer cone (deg)", &outerDegrees, 0.5f, 1.0f, 89.0f);
            if (changed)
            {
                light.innerConeAngle = glm::radians(std::min(innerDegrees, outerDegrees));
                light.outerConeAngle = glm::radians(std::max(innerDegrees, outerDegrees));
                m_sceneLightsDirty = true;
            }
        }
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
        DrawTexturePreview("Motion vectors (pixels)", m_motionTextureId, previewWidth, aspect);
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
        m_ui.RemoveTexture(m_motionTextureId);
        m_albedoTextureId = 0;
        m_normalTextureId = 0;
        m_positionTextureId = 0;
        m_motionTextureId = 0;
        m_debugTexturesRegistered = false;
    }

    mukki::Camera m_camera{};
    mukki::FlyCameraController m_controller{};
    glm::mat4 m_previousViewProjection{1.0f};
    mukki::JobSystem m_jobSystem{};
    std::unique_ptr<mukki::rhi::Device> m_device{};
    std::unique_ptr<mukki::DeferredRenderer> m_renderer{};
    mukki::vulkan::UiLayer m_ui{};
    mukki::SceneDescription m_scene{};
    std::vector<std::filesystem::path> m_sceneFiles{};
    std::filesystem::path m_scenesRoot{};
    std::filesystem::path m_pendingModelPath{};
    ImTextureID m_albedoTextureId = 0;
    ImTextureID m_normalTextureId = 0;
    ImTextureID m_positionTextureId = 0;
    ImTextureID m_motionTextureId = 0;
    std::filesystem::path m_shaderDirectory{};
    std::filesystem::path m_sceneBinaryDirectory{};
    std::filesystem::path m_sceneSourcePath{};
    std::filesystem::path m_requestedModel{};
    std::filesystem::path m_requestedScene{};
    std::string m_sceneStatus{};
    float m_frameTimeMs = 0.0f;
    int m_selectedLight = 0;
    int m_selectedObject = 0;
    int m_selectedProbe = -1;
    int m_probeFillCount = static_cast<int>(mukki::kMaxGiProbes);
    int m_sceneFileIndex = 0;
    int m_maxFrames = 0;
    int m_frameCount = 0;
    bool m_sceneLightsDirty = false;
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
    std::filesystem::path requestedScene{};
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
        else if (std::strcmp(argv[i], "--scene") == 0 && i + 1 < argc)
        {
            requestedScene = argv[i + 1];
        }
    }

    mukki::Log::Init(mukki::LogConfig{});

    int exitCode = EXIT_SUCCESS;
    try
    {
        const std::filesystem::path shaderDirectory =
            std::filesystem::absolute(argv[0]).parent_path() / "shaders";

        SandboxApp app{maxFrames, shaderDirectory, requestedModel, requestedScene};
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
