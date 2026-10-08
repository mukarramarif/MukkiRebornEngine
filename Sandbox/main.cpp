#include "Application/Application.h"
#include "Application/Camera.h"
#include "Application/Input.h"
#include "Application/Keybinds.h"
#include "Application/PlaneController.h"
#include "Application/Window.h"
#include "Core/JobSystem.h"
#include "Physics/PhysicsWorld.h"
#include "Renderer/Debug/DebugLines.h"
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
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <GLFW/glfw3.h>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <imgui.h>
#include <imgui_internal.h> // DockBuilder API for the default dock layout

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

// Chase camera offset in airplane local space (behind = +Z, above = +Y, in metres).
constexpr glm::vec3 kChaseLocalOffset{0.0f, 3.2f, 10.5f};


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
        std::filesystem::path requestedScene,
        std::filesystem::path requestedEnvironment)
        : m_maxFrames(maxFrames)
        , m_shaderDirectory(std::move(shaderDirectory))
        , m_requestedModel(std::move(requestedModel))
        , m_requestedScene(std::move(requestedScene))
        , m_requestedEnvironment(std::move(requestedEnvironment))
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

        // Flight demo controls (active while flight mode is enabled).
        keybinds.Bind("flight.throttleUp", mukki::Binding::Key(GLFW_KEY_W));
        keybinds.Bind("flight.throttleDown", mukki::Binding::Key(GLFW_KEY_S));
        keybinds.Bind("flight.yawLeft", mukki::Binding::Key(GLFW_KEY_A));
        keybinds.Bind("flight.yawRight", mukki::Binding::Key(GLFW_KEY_D));
        keybinds.Bind("flight.reset", mukki::Binding::Key(GLFW_KEY_R));
        keybinds.Bind("flight.toggle", mukki::Binding::Key(GLFW_KEY_F));

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

        // Keep the dock layout next to the executable so it survives working directory changes.
        // A layout saved by an older window set is stale; rebuild the default layout when the
        // marker window is missing.
        const std::filesystem::path iniPath = m_shaderDirectory.parent_path() / "imgui.ini";
        std::string iniContents{};
        if (std::ifstream stream{iniPath, std::ios::binary}; stream)
        {
            std::ostringstream buffer;
            buffer << stream.rdbuf();
            iniContents = buffer.str();
        }
        m_hasSavedLayout = iniContents.find("[Window][GI & Lights]") != std::string::npos &&
                           iniContents.find("[Window][Flight]") != std::string::npos;
        m_iniPath = mukki::PathToUtf8(iniPath);
        ImGui::GetIO().IniFilename = m_iniPath.c_str();

        mukki::PhysicsSettings physicsSettings{};
        if (!m_physics.Init(physicsSettings))
        {
            MRE_WARN("Failed to initialize the physics world; physics is disabled");
        }

        if (!m_requestedEnvironment.empty())
        {
            mukki::EnvironmentSettings environment = m_renderer->GetEnvironmentSettings();
            environment.environmentMap = m_requestedEnvironment;
            m_renderer->SetEnvironmentSettings(environment);
        }

        ScanSceneFiles();
        ScanEnvironmentMaps();

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

        RebuildPhysics();
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

        if (keybinds.IsActionPressed("flight.toggle"))
        {
            SetFlightEnabled(!m_flightEnabled);
        }

        if (m_flightEnabled)
        {
            UpdateFlight(deltaTime, input, keybinds, lookEnabled);
        }
        else
        {
            mukki::CameraInput cameraInput{};
            cameraInput.move.x = keybinds.GetAxis("camera.left", "camera.right");
            cameraInput.move.y = keybinds.GetAxis("camera.backward", "camera.forward");
            cameraInput.vertical = keybinds.GetAxis("camera.down", "camera.up");
            cameraInput.look = input.GetState().mouseDelta;
            cameraInput.fast = keybinds.IsActionDown("camera.fast");
            cameraInput.lookEnabled = lookEnabled;

            m_controller.Update(m_camera, cameraInput, deltaTime);
        }
        m_frameTimeMs = deltaTime * 1000.0f;

        m_physics.Step(deltaTime);
        SyncDynamicBodies();
        if (m_renderer->ArePhysicsGizmosEnabled())
        {
            mukki::DebugLines lines{};
            m_physics.BuildDebugLines(lines);
            m_renderer->SetPhysicsDebugLines(std::move(lines));
        }

        if (GetWindow().WasResized())
        {
            const std::uint32_t width = static_cast<std::uint32_t>(GetWindow().GetWidth());
            const std::uint32_t height = static_cast<std::uint32_t>(GetWindow().GetHeight());

            ApplyCameraPerspective();
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
        DrawDockspace();
        DrawStatsWindow();
        if (m_showScenesWindow)
        {
            DrawSceneBrowser();
        }
        if (m_showGiLightsWindow)
        {
            DrawGiLightsWindow();
        }
        if (m_showObjectsPhysicsWindow)
        {
            DrawObjectsPhysicsWindow();
        }
        if (m_showVolumetricsWindow)
        {
            DrawVolumetricsWindow();
        }
        if (m_showFlightWindow)
        {
            DrawFlightWindow();
        }
        if (m_showGBuffer)
        {
            DrawGBufferWindow();
        }
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
        // The GPU may still be executing the last submitted frame; wait before releasing any
        // resources (descriptor sets, textures) that frame references.
        if (m_device != nullptr)
        {
            m_device->WaitIdle();
        }
        ReleaseDebugTextures();
        m_ui.Shutdown();
        m_physics.Shutdown();
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

    void ScanEnvironmentMaps()
    {
#ifdef MRE_ASSET_LIBRARY_DIR
        m_assetLibraryRoot = std::filesystem::path{MRE_ASSET_LIBRARY_DIR};
        if (!std::filesystem::exists(m_assetLibraryRoot))
        {
            MRE_WARN(
                "Asset library directory '{}' does not exist",
                mukki::PathToUtf8(m_assetLibraryRoot));
            return;
        }

        std::error_code error{};
        for (const std::filesystem::directory_entry& entry :
             std::filesystem::recursive_directory_iterator(m_assetLibraryRoot, error))
        {
            if (error || !entry.is_regular_file())
            {
                continue;
            }

            std::string extension = mukki::PathToUtf8(entry.path().extension());
            std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
            if (extension == ".hdr")
            {
                m_environmentMaps.push_back(entry.path());
            }
        }

        std::sort(m_environmentMaps.begin(), m_environmentMaps.end());
        MRE_INFO(
            "Found {} HDR environment map(s) in '{}'",
            m_environmentMaps.size(),
            mukki::PathToUtf8(m_assetLibraryRoot));
#else
        MRE_WARN("Asset library directory was not configured at build time");
#endif
    }

    void DrawStatsWindow()
    {
        ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_MenuBar);
        if (ImGui::BeginMenuBar())
        {
            if (ImGui::BeginMenu("View"))
            {
                ImGui::MenuItem("Scenes", nullptr, &m_showScenesWindow);
                ImGui::MenuItem("GI & Lights", nullptr, &m_showGiLightsWindow);
                ImGui::MenuItem("Objects & Physics", nullptr, &m_showObjectsPhysicsWindow);
                ImGui::MenuItem("Volumetrics", nullptr, &m_showVolumetricsWindow);
                ImGui::MenuItem("Flight", nullptr, &m_showFlightWindow);
                ImGui::MenuItem("G-buffer", nullptr, &m_showGBuffer);
                ImGui::MenuItem("ImGui demo", nullptr, &m_showDemoWindow);
                ImGui::Separator();
                if (ImGui::MenuItem("Reset layout"))
                {
                    m_resetLayoutRequested = true;
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        ImGui::Text("Frame time: %.2f ms", static_cast<double>(m_frameTimeMs));
        ImGui::Text(
            "Camera: %.2f, %.2f, %.2f",
            static_cast<double>(m_camera.GetPosition().x),
            static_cast<double>(m_camera.GetPosition().y),
            static_cast<double>(m_camera.GetPosition().z));
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
                ClearFlight();
                m_renderer->ClearScene();
                m_scene = mukki::SceneDescription{};
                m_sceneSourcePath.clear();
                m_physics.ClearBodies();
                m_sceneStatus = "scene unloaded";
            }
        }

        ImGui::SeparatorText("Transform");
        float modelScale = m_renderer->GetModelScale();
        if (ImGui::SliderFloat("Uniform scale", &modelScale, 0.1f, 10.0f, "%.2f"))
        {
            m_renderer->SetModelScale(modelScale);
            // Colliders are fitted at build time; rebuild so they follow the new scale.
            RebuildPhysics();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset"))
        {
            m_renderer->SetModelScale(1.0f);
            RebuildPhysics();
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

        mukki::PhysicsSettings physicsSettings = m_physics.GetSettings();
        physicsSettings.gravity = m_scene.physics.gravity;
        physicsSettings.stepHz = m_scene.physics.stepHz;
        physicsSettings.subSteps = m_scene.physics.subSteps;
        m_physics.SetSettings(physicsSettings);
        ApplySceneEnvironment();
        ResetFloorFromSceneBounds();
        RebuildPhysics();
        PrepareFlight();

        m_sceneLightsDirty = false;
        m_sceneStatus = "loaded '" + m_scene.name + "' (" + std::to_string(m_scene.objects.size()) +
                        " objects, " + std::to_string(m_scene.lights.size()) + " lights, " +
                        std::to_string(m_physics.GetBodyCount()) + " bodies)";
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

    std::uint32_t FindBodyHandleForObject(std::size_t objectIndex) const
    {
        for (const mukki::CollisionBody& body : m_physics.GetBodies())
        {
            if (body.handle != 0 && body.objectIndex == static_cast<std::uint32_t>(objectIndex))
            {
                return body.handle;
            }
        }
        return 0;
    }

    // Recreates the default static floor collider from the adjustable floor settings. Only the
    // ground body is touched so dynamic bodies keep simulating while the floor is edited.
    void UpdateFloorBody()
    {
        if (!m_physics.IsInitialized())
        {
            return;
        }

        if (m_groundBody != 0)
        {
            m_physics.DestroyBody(m_groundBody);
            m_groundBody = 0;
        }
        if (!m_floor.enabled)
        {
            return;
        }

        mukki::BodyDesc ground{};
        ground.type = mukki::BodyType::Static;
        ground.position = glm::vec3{
            m_floor.center.x,
            m_floor.topY - m_floor.thickness * 0.5f,
            m_floor.center.y};
        ground.collider.shape = mukki::ColliderShapeType::Box;
        ground.collider.halfExtents = glm::vec3{
            std::max(m_floor.halfExtentX, 0.5f),
            std::max(m_floor.thickness * 0.5f, 0.05f),
            std::max(m_floor.halfExtentZ, 0.5f)};
        ground.collider.friction = 0.8f;
        m_groundBody = m_physics.CreateBody(ground);
    }

    // Fits the floor defaults to the loaded scene footprint (top surface at the scene minimum Y).
    void ResetFloorFromSceneBounds()
    {
        m_floor.enabled = true;
        m_floor.topY = 0.0f;
        m_floor.center = glm::vec2{0.0f};
        m_floor.halfExtentX = 50.0f;
        m_floor.halfExtentZ = 50.0f;
        m_floor.thickness = 1.0f;

        glm::vec3 boundsMin{0.0f};
        glm::vec3 boundsMax{0.0f};
        if (m_renderer != nullptr && m_renderer->GetSceneBounds(boundsMin, boundsMax))
        {
            const glm::vec3 extent = glm::max(boundsMax - boundsMin, glm::vec3{0.0f});
            m_floor.topY = boundsMin.y;
            m_floor.center = glm::vec2{
                (boundsMin.x + boundsMax.x) * 0.5f,
                (boundsMin.z + boundsMax.z) * 0.5f};
            m_floor.halfExtentX = std::max(extent.x * 0.5f + 1.0f, 10.0f);
            m_floor.halfExtentZ = std::max(extent.z * 0.5f + 1.0f, 10.0f);
        }
    }

    // Recreates every collider from the current scene description. Runs after scene loads and on
    // demand from the physics UI; resets dynamic body poses.
    void RebuildPhysics()
    {
        if (!m_physics.IsInitialized() || m_renderer == nullptr)
        {
            return;
        }

        m_physics.ClearBodies();
        m_groundBody = 0;
        UpdateFloorBody();


        const std::size_t objectCount =
            std::min(m_scene.objects.size(), m_renderer->GetSceneObjectCount());
        for (std::size_t index = 0; index < objectCount; ++index)
        {
            const mukki::SceneObject& object = m_scene.objects[index];
            if (object.physics.body == mukki::ScenePhysicsBody::None)
            {
                continue;
            }

            glm::vec3 fittedMin{0.0f};
            glm::vec3 fittedMax{0.0f};
            if (!m_renderer->GetSceneObjectFittedBounds(index, fittedMin, fittedMax))
            {
                continue;
            }

            const glm::vec3 fittedCenter = (fittedMin + fittedMax) * 0.5f;
            const glm::vec3 fittedHalfExtents =
                glm::max((fittedMax - fittedMin) * 0.5f, glm::vec3{0.01f});
            const glm::quat rotation = mukki::BuildObjectRotation(object);

            mukki::BodyDesc desc{};
            desc.type = object.physics.body == mukki::ScenePhysicsBody::Dynamic
                            ? mukki::BodyType::Dynamic
                            : mukki::BodyType::Static;
            desc.objectIndex = static_cast<std::uint32_t>(index);
            desc.position = object.position;
            desc.rotation = rotation;
            desc.collider.density = object.physics.density;
            desc.collider.friction = object.physics.friction;
            desc.collider.restitution = object.physics.restitution;
            // The auto fitted model is not necessarily centred on the authored object position;
            // keep the difference as a body local offset so colliders track the visuals.
            desc.collider.offset =
                glm::inverse(rotation) * (fittedCenter - object.position) + object.physics.offset;

            const bool authoredExtents = object.physics.halfExtents.x > 0.0f &&
                                         object.physics.halfExtents.y > 0.0f &&
                                         object.physics.halfExtents.z > 0.0f;

            switch (object.physics.shape)
            {
            case mukki::ScenePhysicsShape::Box:
                desc.collider.shape = mukki::ColliderShapeType::Box;
                desc.collider.halfExtents = authoredExtents ? object.physics.halfExtents
                                                            : fittedHalfExtents;
                break;
            case mukki::ScenePhysicsShape::Sphere:
                desc.collider.shape = mukki::ColliderShapeType::Sphere;
                desc.collider.radius = object.physics.radius > 0.0f
                                           ? object.physics.radius
                                           : std::max(
                                                 fittedHalfExtents.x,
                                                 std::max(fittedHalfExtents.y, fittedHalfExtents.z));
                break;
            case mukki::ScenePhysicsShape::Capsule:
                desc.collider.shape = mukki::ColliderShapeType::Capsule;
                desc.collider.radius = object.physics.radius > 0.0f
                                           ? object.physics.radius
                                           : std::max(fittedHalfExtents.x, fittedHalfExtents.z);
                desc.collider.height = object.physics.height > 0.0f
                                           ? object.physics.height
                                           : std::max(
                                                 fittedHalfExtents.y * 2.0f,
                                                 desc.collider.radius * 2.0f + 0.05f);
                break;
            }

            if (m_physics.CreateBody(desc) == 0)
            {
                MRE_WARN("Failed to create a physics body for scene object '{}'", object.name);
            }
        }
    }

    // Post-step transform sync: dynamic bodies drive the scene objects and the renderer.
    void SyncDynamicBodies()
    {
        for (const mukki::CollisionBody& body : m_physics.GetBodies())
        {
            if (body.handle == 0 || body.type != mukki::BodyType::Dynamic ||
                body.objectIndex >= m_scene.objects.size())
            {
                continue;
            }

            glm::vec3 position{0.0f};
            glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
            if (!m_physics.GetBodyTransform(body.handle, position, rotation))
            {
                continue;
            }

            mukki::SceneObject& object = m_scene.objects[body.objectIndex];
            object.position = position;
            object.rotation = mukki::ExtractObjectRotationEuler(rotation);

            if (body.objectIndex < m_renderer->GetSceneObjectCount())
            {
                m_renderer->UpdateSceneObjectTransform(
                    body.objectIndex,
                    mukki::BuildObjectTransform(object));
            }
        }
    }

    std::filesystem::path ResolveSceneAssetPath(const std::filesystem::path& path) const
    {
        if (path.empty() || path.is_absolute() || m_scene.assetRoot.empty())
        {
            return path;
        }
        return m_scene.assetRoot / path;
    }

    // Applies the optional scene authored environment and cloud settings. Absent blocks leave the
    // renderer's current settings alone, so --env and UI edits survive older scenes.
    void ApplySceneEnvironment()
    {
        if (m_renderer == nullptr)
        {
            return;
        }
        if (m_scene.hasEnvironment)
        {
            mukki::EnvironmentSettings environment = m_renderer->GetEnvironmentSettings();
            environment.environmentMap = ResolveSceneAssetPath(m_scene.environment.map);
            environment.intensity = m_scene.environment.intensity;
            m_renderer->SetEnvironmentSettings(environment);
        }
        if (m_scene.hasClouds)
        {
            m_renderer->SetCloudSettings(m_scene.clouds);
        }
    }

    void ApplyCameraPerspective()
    {
        const float aspect = GetWindow().GetAspectRatio();
        if (m_flightEnabled)
        {
            m_camera.SetPerspective(glm::radians(70.0f), aspect, 1.0f, 25000.0f);
        }
        else
        {
            m_camera.SetPerspective(1.0471975512f, aspect, 0.1f, 1000.0f);
        }
    }

    // Recreates the plane at the authored spawn pose and snaps the chase camera behind it.
    void ResetPlane()
    {
        m_plane.Reset(m_planeStartPosition, m_planeStartRotation, 140.0f, 0.6f);
        m_flightStickX = 0.0f;
        m_flightStickY = 0.0f;
        m_crashed = false;
        m_crashTimer = 0.0f;
        m_flightStatus = "flying";

        const glm::quat orientation = m_plane.GetOrientation();
        m_chaseCameraPosition = m_plane.GetPosition() + orientation * kChaseLocalOffset;
        SyncPlaneObject();
    }

    void SetFlightEnabled(bool enabled)
    {
        if (enabled && m_planeObjectIndex < 0)
        {
            m_sceneStatus = "flight mode needs a 'playerPlane' object in the scene";
            return;
        }

        m_flightEnabled = enabled;
        if (enabled)
        {
            ResetPlane();
            m_renderer->SetGroundPlaneExtent(9000.0f);
        }
        else
        {
            m_crashed = false;
            m_renderer->SetGroundPlaneExtent(10.0f);
        }
        ApplyCameraPerspective();
    }

    // Keeps the rendered airplane in sync with the controller state.
    void SyncPlaneObject()
    {
        if (m_planeObjectIndex < 0 || m_renderer == nullptr ||
            static_cast<std::size_t>(m_planeObjectIndex) >= m_renderer->GetSceneObjectCount())
        {
            return;
        }

        const glm::mat4 transform =
            glm::translate(glm::mat4{1.0f}, m_plane.GetPosition()) *
            glm::mat4_cast(m_plane.GetOrientation());
        m_renderer->UpdateSceneObjectTransform(
            static_cast<std::size_t>(m_planeObjectIndex),
            transform);
    }

    void CacheCollisionBounds()
    {
        m_collisionBoundsMin.clear();
        m_collisionBoundsMax.clear();
        if (m_renderer == nullptr)
        {
            return;
        }

        const std::size_t count =
            std::min(m_scene.objects.size(), m_renderer->GetSceneObjectCount());
        for (std::size_t index = 0; index < count; ++index)
        {
            if (m_scene.objects[index].physics.body != mukki::ScenePhysicsBody::Static)
            {
                continue;
            }

            glm::vec3 boundsMin{0.0f};
            glm::vec3 boundsMax{0.0f};
            if (m_renderer->GetSceneObjectFittedBounds(index, boundsMin, boundsMax))
            {
                m_collisionBoundsMin.push_back(boundsMin);
                m_collisionBoundsMax.push_back(boundsMax);
            }
        }
    }

    void PrepareFlight()
    {
        m_planeObjectIndex = -1;
        m_flightEnabled = false;
        m_crashed = false;
        m_crashTimer = 0.0f;
        m_flightStatus.clear();

        if (!m_scene.playerPlane.empty())
        {
            for (std::size_t index = 0; index < m_scene.objects.size(); ++index)
            {
                if (m_scene.objects[index].name == m_scene.playerPlane)
                {
                    m_planeObjectIndex = static_cast<int>(index);
                    m_planeStartPosition = m_scene.objects[index].position;
                    m_planeStartRotation = mukki::BuildObjectRotation(m_scene.objects[index]);
                    break;
                }
            }
            if (m_planeObjectIndex < 0)
            {
                MRE_WARN("playerPlane '{}' was not found in the scene", m_scene.playerPlane);
            }
        }

        CacheCollisionBounds();
        if (m_planeObjectIndex >= 0)
        {
            SetFlightEnabled(true);
        }
    }

    void ClearFlight()
    {
        m_flightEnabled = false;
        m_planeObjectIndex = -1;
        m_crashed = false;
        m_crashTimer = 0.0f;
        m_flightStatus.clear();
        m_collisionBoundsMin.clear();
        m_collisionBoundsMax.clear();
        if (m_renderer != nullptr)
        {
            m_renderer->SetGroundPlaneExtent(10.0f);
        }
        ApplyCameraPerspective();
    }

    void CrashPlane(const char* reason)
    {
        if (m_crashed)
        {
            return;
        }
        m_crashed = true;
        m_crashTimer = 1.6f;
        ++m_crashCount;
        m_flightStatus = std::string{reason} + " - respawning";
        MRE_INFO("Flight: {}", m_flightStatus);
    }

    void CheckPlaneCrash()
    {
        const glm::vec3 position = m_plane.GetPosition();
        if (position.y <= m_floor.topY + 3.0f)
        {
            CrashPlane("crashed into the ground");
            return;
        }

        constexpr float kPlaneRadius = 3.0f;
        for (std::size_t index = 0; index < m_collisionBoundsMin.size(); ++index)
        {
            const glm::vec3 min = m_collisionBoundsMin[index];
            const glm::vec3 max = m_collisionBoundsMax[index];
            const bool inside =
                position.x + kPlaneRadius >= min.x && position.x - kPlaneRadius <= max.x &&
                position.y + kPlaneRadius >= min.y && position.y - kPlaneRadius <= max.y &&
                position.z + kPlaneRadius >= min.z && position.z - kPlaneRadius <= max.z;
            if (inside)
            {
                CrashPlane("crashed into a skyscraper");
                return;
            }
        }

        if (glm::length(glm::vec2{position.x, position.z}) > 12000.0f)
        {
            CrashPlane("left the city airspace");
        }
    }

    void UpdateChaseCamera(float deltaTime)
    {
        const glm::quat orientation = m_plane.GetOrientation();
        const glm::vec3 desired = m_plane.GetPosition() + orientation * kChaseLocalOffset;
        const float blend = 1.0f - std::exp(-6.0f * deltaTime);
        m_chaseCameraPosition = glm::mix(m_chaseCameraPosition, desired, blend);

        const glm::vec3 lookTarget =
            m_plane.GetPosition() + orientation * glm::vec3{0.0f, 1.5f, -12.0f};
        const glm::vec3 forward = lookTarget - m_chaseCameraPosition;
        if (glm::length(forward) > 1.0e-3f)
        {
            m_camera.SetView(m_chaseCameraPosition, forward, orientation * glm::vec3{0.0f, 1.0f, 0.0f});
        }
    }

    void UpdateFlight(
        float deltaTime,
        const mukki::Input& input,
        const mukki::Keybinds& keybinds,
        bool lookEnabled)
    {
        if (keybinds.IsActionPressed("flight.reset"))
        {
            ResetPlane();
        }

        // Mouse virtual stick: mouse deltas push the stick and it springs back toward center.
        if (lookEnabled)
        {
            const glm::vec2 mouseDelta = input.GetState().mouseDelta;
            m_flightStickX = std::clamp(m_flightStickX + mouseDelta.x * m_flightMouseGain, -1.0f, 1.0f);
            m_flightStickY = std::clamp(m_flightStickY - mouseDelta.y * m_flightMouseGain, -1.0f, 1.0f);
        }
        const float returnFactor = std::exp(-m_flightStickReturn * deltaTime);
        m_flightStickX *= returnFactor;
        m_flightStickY *= returnFactor;

        if (m_crashed)
        {
            m_crashTimer -= deltaTime;
            if (m_crashTimer <= 0.0f)
            {
                ResetPlane();
            }
        }
        else
        {
            mukki::FlightInput flight{};
            flight.roll = m_flightStickX;
            flight.pitch = m_flightStickY;
            flight.yaw = keybinds.GetAxis("flight.yawLeft", "flight.yawRight");
            flight.throttle = keybinds.GetAxis("flight.throttleDown", "flight.throttleUp");
            m_plane.Update(flight, deltaTime);
            SyncPlaneObject();
            CheckPlaneCrash();
        }

        UpdateChaseCamera(deltaTime);
    }

    void DrawDockspace()
    {
        // Passthru keeps the central node transparent so the rendered scene shows through the
        // empty dock space instead of the host window background.
        const ImGuiID dockspaceId =
            ImGui::DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode);
        if (!m_layoutInitialized)
        {
            m_layoutInitialized = true;
            if (!m_hasSavedLayout)
            {
                BuildDefaultLayout(dockspaceId);
            }
        }
        if (m_resetLayoutRequested)
        {
            m_resetLayoutRequested = false;
            BuildDefaultLayout(dockspaceId);
        }
    }

    // Typed tab groups: Stats/Scenes/G-buffer on the left, the lighting/objects/volumetrics
    // editors tabbed together on the right.
    void BuildDefaultLayout(ImGuiID dockspaceId)
    {
        ImGui::DockBuilderRemoveNode(dockspaceId);
        ImGui::DockBuilderAddNode(
            dockspaceId,
            static_cast<ImGuiDockNodeFlags>(
                ImGuiDockNodeFlags_DockSpace | ImGuiDockNodeFlags_PassthruCentralNode));
        ImGui::DockBuilderSetNodeSize(dockspaceId, ImGui::GetMainViewport()->WorkSize);

        ImGuiID leftNode = 0;
        ImGuiID mainNode = 0;
        ImGui::DockBuilderSplitNode(dockspaceId, ImGuiDir_Left, 0.28f, &leftNode, &mainNode);

        ImGui::DockBuilderDockWindow("Stats", leftNode);
        ImGui::DockBuilderDockWindow("Scenes", leftNode);
        ImGui::DockBuilderDockWindow("G-buffer", leftNode);
        ImGui::DockBuilderDockWindow("GI & Lights", mainNode);
        ImGui::DockBuilderDockWindow("Objects & Physics", mainNode);
        ImGui::DockBuilderDockWindow("Volumetrics", mainNode);
        ImGui::DockBuilderDockWindow("Flight", mainNode);
        ImGui::DockBuilderFinish(dockspaceId);
    }

    void DrawGiLightsWindow()
    {
        ImGui::Begin("GI & Lights", &m_showGiLightsWindow);

        ImGui::Text(
            "Lights: %zu in scene, %u in renderer",
            m_scene.lights.size(),
            m_renderer->GetLightCount());

        bool debugClusters = m_renderer->IsClusterDebugViewEnabled();
        if (ImGui::Checkbox("Visualize cluster light counts", &debugClusters))
        {
            m_renderer->SetClusterDebugView(debugClusters);
        }
        bool showLightGizmos = m_renderer->AreLightGizmosEnabled();
        if (ImGui::Checkbox("Show light gizmos", &showLightGizmos))
        {
            m_renderer->SetGizmoView(showLightGizmos, m_renderer->AreProbeGizmosEnabled());
        }

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

    void DrawObjectsPhysicsWindow()
    {
        ImGui::Begin("Objects & Physics", &m_showObjectsPhysicsWindow);

        ImGui::Text("Scene: %s", m_scene.name.empty() ? "(none)" : m_scene.name.c_str());
        ImGui::Text(
            "Objects: %zu   Lights: %zu",
            m_scene.objects.size(),
            m_scene.lights.size());
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

        ImGui::Checkbox("Show G-buffer window", &m_showGBuffer);
        ImGui::SetItemTooltip(
            "Shows the G-buffer debug tab (albedo, normals, position, motion vectors) in the\n"
            "Stats/Scenes group.");

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

        bool showProbeGizmos = m_renderer->AreProbeGizmosEnabled();
        if (ImGui::Checkbox("Show probe gizmos", &showProbeGizmos))
        {
            m_renderer->SetGizmoView(m_renderer->AreLightGizmosEnabled(), showProbeGizmos);
        }

        DrawPhysicsSettings();

        ImGui::End();
    }

    void DrawVolumetricsWindow()
    {
        ImGui::Begin("Volumetrics", &m_showVolumetricsWindow);
        DrawEnvironmentSettings();
        DrawCloudSettings();
        ImGui::End();
    }

    void DrawFlightWindow()
    {
        ImGui::Begin("Flight", &m_showFlightWindow);

        const bool hasPlane = m_planeObjectIndex >= 0;
        bool enabled = m_flightEnabled;
        if (ImGui::Checkbox("Flight mode (F)", &enabled))
        {
            SetFlightEnabled(enabled);
        }
        if (!hasPlane)
        {
            ImGui::SameLine();
            ImGui::TextDisabled("(author \"playerPlane\" in the scene)");
        }

        ImGui::Text("Status: %s", m_flightStatus.empty() ? "-" : m_flightStatus.c_str());
        ImGui::Text(
            "Speed: %.0f m/s (%.0f km/h)",
            static_cast<double>(m_plane.GetSpeed()),
            static_cast<double>(m_plane.GetSpeed() * 3.6f));
        ImGui::Text("Altitude: %.0f m", static_cast<double>(m_plane.GetPosition().y));
        ImGui::Text("Throttle: %.0f %%", static_cast<double>(m_plane.GetThrottle() * 100.0f));
        ImGui::Text("Bank: %.0f deg", static_cast<double>(glm::degrees(m_plane.GetBankRadians())));
        ImGui::Text("Crashes: %d", m_crashCount);

        if (ImGui::Button("Reset plane (R)"))
        {
            ResetPlane();
        }
        ImGui::SetItemTooltip("Respawns at the authored start pose and clears the crash state.");

        ImGui::SeparatorText("Handling");
        mukki::FlightSettings settings = m_plane.GetSettings();
        bool changed = ImGui::SliderFloat("Max speed (m/s)", &settings.maxSpeed, 120.0f, 500.0f, "%.0f");
        changed |= ImGui::SliderFloat("Pitch rate", &settings.pitchRate, 0.4f, 3.0f, "%.2f");
        changed |= ImGui::SliderFloat("Roll rate", &settings.rollRate, 0.8f, 5.0f, "%.2f");
        changed |= ImGui::SliderFloat("Yaw rate", &settings.yawRate, 0.2f, 2.0f, "%.2f");
        changed |= ImGui::SliderFloat("Auto level", &settings.autoLevelGain, 0.0f, 2.0f, "%.2f");
        changed |= ImGui::SliderFloat("Mouse stick gain", &m_flightMouseGain, 0.001f, 0.02f, "%.3f");
        changed |= ImGui::SliderFloat("Stick return", &m_flightStickReturn, 0.5f, 8.0f, "%.1f");
        if (changed)
        {
            m_plane.GetSettings() = settings;
        }

        ImGui::SeparatorText("Controls");
        ImGui::TextUnformatted("Mouse: pitch / roll (captured cursor)");
        ImGui::TextUnformatted("W/S: throttle, A/D: rudder");
        ImGui::TextUnformatted("F: toggle flight mode, R: reset, Tab: free cursor");
        ImGui::TextUnformatted(
            "Hitting a building, the ground or the 12 km boundary respawns you.");
        ImGui::End();
    }

    void DrawPhysicsSettings()
    {
        ImGui::SeparatorText("Physics (Box3D)");

        mukki::PhysicsSettings settings = m_physics.GetSettings();
        bool changed = ImGui::Checkbox("Physics enabled", &settings.enabled);
        changed |= ImGui::DragFloat3("Gravity", &settings.gravity.x, 0.05f, -100.0f, 100.0f);
        changed |= ImGui::SliderFloat("Step rate (Hz)", &settings.stepHz, 15.0f, 240.0f, "%.0f");
        int subSteps = static_cast<int>(settings.subSteps);
        if (ImGui::SliderInt("Sub-steps", &subSteps, 1, 8))
        {
            settings.subSteps = static_cast<std::uint32_t>(subSteps);
            changed = true;
        }
        if (changed)
        {
            m_physics.SetSettings(settings);
            // Mirror into the scene so Save JSON persists gravity and solver settings.
            m_scene.physics.gravity = settings.gravity;
            m_scene.physics.stepHz = settings.stepHz;
            m_scene.physics.subSteps = settings.subSteps;
        }

        std::size_t staticCount = 0;
        std::size_t dynamicCount = 0;
        std::size_t kinematicCount = 0;
        for (const mukki::CollisionBody& body : m_physics.GetBodies())
        {
            if (body.handle == 0)
            {
                continue;
            }
            switch (body.type)
            {
            case mukki::BodyType::Static:
                ++staticCount;
                break;
            case mukki::BodyType::Dynamic:
                ++dynamicCount;
                break;
            case mukki::BodyType::Kinematic:
                ++kinematicCount;
                break;
            }
        }
        ImGui::Text(
            "Bodies: %zu (static %zu, dynamic %zu, kinematic %zu)",
            m_physics.GetBodyCount(),
            staticCount,
            dynamicCount,
            kinematicCount);
        ImGui::Text(
            "Box3D memory: %.1f KiB",
            static_cast<double>(m_physics.GetByteCount()) / 1024.0);

        ImGui::SeparatorText("Static floor");
        bool floorChanged = ImGui::Checkbox("Floor collider", &m_floor.enabled);
        ImGui::SetItemTooltip(
            "The ground collider dynamic bodies rest on. Disable it to drop them into the void.");
        floorChanged |= ImGui::DragFloat("Floor top (Y)", &m_floor.topY, 0.05f, -1000.0f, 1000.0f, "%.2f");
        floorChanged |= ImGui::DragFloat2("Floor center (XZ)", &m_floor.center.x, 0.1f, -5000.0f, 5000.0f, "%.1f");
        floorChanged |= ImGui::DragFloat("Floor half X", &m_floor.halfExtentX, 0.5f, 0.5f, 10000.0f, "%.1f");
        floorChanged |= ImGui::DragFloat("Floor half Z", &m_floor.halfExtentZ, 0.5f, 0.5f, 10000.0f, "%.1f");
        floorChanged |= ImGui::DragFloat("Floor thickness", &m_floor.thickness, 0.05f, 0.1f, 100.0f, "%.2f");
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset floor"))
        {
            ResetFloorFromSceneBounds();
            floorChanged = true;
        }
        ImGui::SetItemTooltip("Fits the floor to the loaded scene bounds (top surface at the scene minimum Y).");
        if (floorChanged)
        {
            UpdateFloorBody();
        }

        bool showPhysicsGizmos = m_renderer->ArePhysicsGizmosEnabled();
        if (ImGui::Checkbox("Show collision gizmos", &showPhysicsGizmos))
        {
            m_renderer->SetPhysicsGizmoView(showPhysicsGizmos);
        }
        ImGui::SameLine();
        if (ImGui::Button("Rebuild bodies"))
        {
            RebuildPhysics();
            m_sceneStatus = "physics bodies rebuilt";
        }
        ImGui::SetItemTooltip(
            "Recreates every collider from the scene description and resets dynamic body poses.");
        ImGui::TextDisabled("Gravity and step rate are saved with Save JSON; the floor is runtime only.");
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

    void DrawEnvironmentSettings()
    {
        ImGui::SeparatorText("Environment (IBL)");

        mukki::EnvironmentSettings settings = m_renderer->GetEnvironmentSettings();
        bool changed = ImGui::Checkbox("Image based lighting", &settings.enabled);
        ImGui::SetItemTooltip(
            "Procedural sky providing diffuse SH irradiance, prefiltered specular reflections,\n"
            "the visible background and the sky term of the ray traced GI.");
        changed |= ImGui::SliderFloat("Sky intensity", &settings.intensity, 0.0f, 4.0f, "%.2f");

        // Radiance maps discovered in AssetLibrary/ at startup.
        const std::string currentMap = settings.environmentMap.empty()
                                           ? std::string{"(gradient sky)"}
                                           : mukki::PathToUtf8(settings.environmentMap.filename());
        if (ImGui::BeginCombo("HDR map", currentMap.c_str()))
        {
            if (ImGui::Selectable("(gradient sky)", settings.environmentMap.empty()))
            {
                settings.environmentMap.clear();
                changed = true;
            }

            for (const std::filesystem::path& map : m_environmentMaps)
            {
                const bool selected = map == settings.environmentMap;
                if (ImGui::Selectable(mukki::PathToUtf8(map.filename()).c_str(), selected))
                {
                    settings.environmentMap = map;
                    changed = true;
                }
                if (selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }

            // Keep a map selected through --env selectable even when it is outside the library.
            const bool currentInList = std::any_of(
                m_environmentMaps.begin(),
                m_environmentMaps.end(),
                [&settings](const std::filesystem::path& map) { return map == settings.environmentMap; });
            if (!settings.environmentMap.empty() && !currentInList &&
                ImGui::Selectable(currentMap.c_str(), true))
            {
            }

            ImGui::EndCombo();
        }
        ImGui::SetItemTooltip(
            "Radiance .hdr equirectangular maps found in AssetLibrary.\n"
            "The gradient colors below are ignored while a map is selected.");

        changed |= ImGui::ColorEdit3("Sky zenith", &settings.zenithColor.x);
        changed |= ImGui::ColorEdit3("Sky horizon", &settings.horizonColor.x);
        changed |= ImGui::ColorEdit3("Sky ground", &settings.groundColor.x);
        changed |= ImGui::SliderFloat("Sun disc", &settings.sunDiscIntensity, 0.0f, 50.0f, "%.0f");

        if (changed)
        {
            m_renderer->SetEnvironmentSettings(settings);
        }
    }

    void DrawCloudSettings()
    {
        ImGui::SeparatorText("Volumetric clouds (Nubis)");

        mukki::CloudSettings settings = m_renderer->GetCloudSettings();
        bool changed = ImGui::Checkbox("Clouds", &settings.enabled);
        ImGui::SetItemTooltip(
            "Ray marched volumetric clouds over the sky (inflight generated noise + weather).");

        changed |= ImGui::SliderFloat("Coverage", &settings.coverage, 0.0f, 1.0f, "%.2f");
        changed |= ImGui::SliderFloat("Cloud type", &settings.cloudType, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("0 stratus, 0.33 cumulus, 0.66 cumulonimbus, 1 cirrus");
        changed |= ImGui::SliderFloat("Density", &settings.density, 0.002f, 0.08f, "%.3f");
        changed |= ImGui::SliderFloat("Detail erosion", &settings.detailStrength, 0.0f, 1.0f, "%.2f");
        changed |= ImGui::SliderFloat("Coverage variation", &settings.coverageVariation, 0.0f, 1.0f, "%.2f");
        changed |= ImGui::SliderFloat("Layer bottom (km)", &settings.layerBottomKm, 0.5f, 4.0f, "%.2f");
        changed |= ImGui::SliderFloat("Layer top (km)", &settings.layerTopKm, 1.0f, 8.0f, "%.2f");
        changed |= ImGui::SliderFloat("Altitude variation (km)", &settings.altitudeVariationKm, 0.0f, 2.0f, "%.2f");
        changed |= ImGui::SliderFloat("Wind speed (m/s)", &settings.windSpeed, 0.0f, 80.0f, "%.0f");
        changed |= ImGui::SliderAngle("Wind direction", &settings.windDirectionRadians);
        changed |= ImGui::SliderFloat("Curl strength", &settings.curlStrength, 0.0f, 3.0f, "%.2f");

        int steps = static_cast<int>(settings.primarySteps);
        if (ImGui::SliderInt("Primary steps", &steps, 16, 128))
        {
            settings.primarySteps = static_cast<std::uint32_t>(steps);
            changed = true;
        }
        int lightSteps = static_cast<int>(settings.lightSteps);
        if (ImGui::SliderInt("Light steps", &lightSteps, 1, 8))
        {
            settings.lightSteps = static_cast<std::uint32_t>(lightSteps);
            changed = true;
        }
        changed |= ImGui::SliderFloat("Cloud trace scale", &settings.resolutionScale, 0.25f, 1.0f, "%.2f");
        changed |= ImGui::SliderFloat("Cloud temporal blend", &settings.temporalBlend, 0.0f, 0.98f, "%.2f");
        changed |= ImGui::SliderFloat("God rays", &settings.godRayIntensity, 0.0f, 3.0f, "%.2f");

        if (changed)
        {
            m_renderer->SetCloudSettings(settings);
        }
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

        changed |= ImGui::Checkbox("Dynamic point shadows", &settings.dynamicPointShadows);
        ImGui::SetItemTooltip(
            "When enabled, each point light's shadow map only re-renders when the light or the\n"
            "scene geometry changes (moving lights still update every frame).");

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

        ImGui::SeparatorText("GI punctual lights");
        changed |= ImGui::Checkbox("Point/spot lights in GI", &settings.pointLights);
        int giLights = static_cast<int>(settings.maxGiLights);
        if (ImGui::SliderInt("GI light count", &giLights, 1, static_cast<int>(mukki::kMaxGiLights)))
        {
            settings.maxGiLights = static_cast<std::uint32_t>(giLights);
            changed = true;
        }
        changed |= ImGui::Checkbox("GI light shadows", &settings.pointLightShadows);
        changed |= ImGui::SliderFloat(
            "GI light shadow bias",
            &settings.pointLightShadowBias,
            0.0f,
            0.2f,
            "%.3f");

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
        const bool dynamicBody = object.physics.body == mukki::ScenePhysicsBody::Dynamic;
        ImGui::Text("Model: %s", object.modelPath.c_str());

        bool changed = false;
        changed |= ImGui::DragFloat3("Object position", &object.position.x, 0.05f, -100.0f, 100.0f);
        changed |= ImGui::DragFloat3("Object rotation (deg)", &object.rotation.x, 0.5f, -360.0f, 360.0f);
        if (dynamicBody)
        {
            ImGui::BeginDisabled();
        }
        changed |= ImGui::DragFloat3("Object scale", &object.scale.x, 0.02f, 0.01f, 100.0f);
        if (dynamicBody)
        {
            ImGui::EndDisabled();
            ImGui::SetItemTooltip(
                "Dynamic bodies bake the scale into their collider. Author halfExtents/radius\n"
                "instead so the collider matches the visual.");
        }

        if (changed)
        {
            if (index < m_renderer->GetSceneObjectCount())
            {
                m_renderer->UpdateSceneObjectTransform(index, mukki::BuildObjectTransform(object));
                if (dynamicBody)
                {
                    const std::uint32_t handle = FindBodyHandleForObject(index);
                    if (handle != 0)
                    {
                        m_physics.TeleportBody(
                            handle,
                            object.position,
                            mukki::BuildObjectRotation(object));
                    }
                }
                m_sceneStatus = "object '" + object.name + "' updated";
            }
            else
            {
                m_sceneStatus = "object is not instantiated in the renderer";
            }
        }

        ImGui::SeparatorText("Physics collider");
        bool physicsChanged = false;

        static const char* kBodyNames[] = {"None", "Static", "Dynamic"};
        int body = static_cast<int>(object.physics.body);
        if (ImGui::Combo("Body", &body, kBodyNames, 3))
        {
            object.physics.body = static_cast<mukki::ScenePhysicsBody>(body);
            physicsChanged = true;
        }

        static const char* kShapeNames[] = {"Box", "Sphere", "Capsule"};
        int shape = static_cast<int>(object.physics.shape);
        if (ImGui::Combo("Shape", &shape, kShapeNames, 3))
        {
            object.physics.shape = static_cast<mukki::ScenePhysicsShape>(shape);
            physicsChanged = true;
        }

        switch (object.physics.shape)
        {
        case mukki::ScenePhysicsShape::Box:
            physicsChanged |= ImGui::DragFloat3(
                "Half extents (0 = fit)",
                &object.physics.halfExtents.x,
                0.02f,
                0.0f,
                100.0f);
            break;
        case mukki::ScenePhysicsShape::Sphere:
            physicsChanged |= ImGui::DragFloat(
                "Radius (0 = fit)",
                &object.physics.radius,
                0.02f,
                0.0f,
                100.0f,
                "%.3f");
            break;
        case mukki::ScenePhysicsShape::Capsule:
            physicsChanged |= ImGui::DragFloat(
                "Radius (0 = fit)",
                &object.physics.radius,
                0.02f,
                0.0f,
                100.0f,
                "%.3f");
            physicsChanged |= ImGui::DragFloat(
                "Height (0 = fit)",
                &object.physics.height,
                0.02f,
                0.0f,
                200.0f,
                "%.3f");
            break;
        }
        physicsChanged |= ImGui::DragFloat3("Collider offset", &object.physics.offset.x, 0.01f, -20.0f, 20.0f);
        physicsChanged |= ImGui::DragFloat("Density", &object.physics.density, 5.0f, 0.0f, 100000.0f, "%.0f");
        physicsChanged |= ImGui::DragFloat("Friction", &object.physics.friction, 0.005f, 0.0f, 1.0f, "%.3f");
        physicsChanged |= ImGui::DragFloat("Restitution", &object.physics.restitution, 0.005f, 0.0f, 1.0f, "%.3f");

        if (physicsChanged)
        {
            m_physicsDirty = true;
        }
        if (m_physicsDirty)
        {
            m_physicsDirty = false;
            RebuildPhysics();
            m_sceneStatus = "physics bodies rebuilt";
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

    struct FloorSettings
    {
        bool enabled = true;
        float topY = 0.0f;
        glm::vec2 center{0.0f};
        float halfExtentX = 50.0f;
        float halfExtentZ = 50.0f;
        float thickness = 1.0f;
    };

    mukki::Camera m_camera{};
    mukki::FlyCameraController m_controller{};
    glm::mat4 m_previousViewProjection{1.0f};
    mukki::JobSystem m_jobSystem{};
    std::unique_ptr<mukki::rhi::Device> m_device{};
    std::unique_ptr<mukki::DeferredRenderer> m_renderer{};
    mukki::vulkan::UiLayer m_ui{};
    mukki::PhysicsWorld m_physics{};
    mukki::SceneDescription m_scene{};
    FloorSettings m_floor{};
    std::uint32_t m_groundBody = 0;
    mukki::PlaneController m_plane{};
    glm::vec3 m_planeStartPosition{0.0f};
    glm::quat m_planeStartRotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 m_chaseCameraPosition{0.0f};
    std::vector<glm::vec3> m_collisionBoundsMin{};
    std::vector<glm::vec3> m_collisionBoundsMax{};
    std::string m_flightStatus{};
    float m_flightStickX = 0.0f;
    float m_flightStickY = 0.0f;
    float m_flightMouseGain = 0.006f;
    float m_flightStickReturn = 2.5f;
    float m_crashTimer = 0.0f;
    int m_planeObjectIndex = -1;
    int m_crashCount = 0;
    std::string m_iniPath{};
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
    std::filesystem::path m_requestedEnvironment{};
    std::filesystem::path m_assetLibraryRoot{};
    std::vector<std::filesystem::path> m_environmentMaps{};
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
    bool m_physicsDirty = false;
    bool m_debugTexturesRegistered = false;
    bool m_renderedFrame = false;
    bool m_cursorReleased = false;
    bool m_showScenesWindow = true;
    bool m_showGiLightsWindow = true;
    bool m_showObjectsPhysicsWindow = true;
    bool m_showVolumetricsWindow = true;
    bool m_showFlightWindow = true;
    bool m_showGBuffer = true;
    bool m_showDemoWindow = false;
    bool m_layoutInitialized = false;
    bool m_hasSavedLayout = false;
    bool m_resetLayoutRequested = false;
    bool m_flightEnabled = false;
    bool m_crashed = false;
};

} // namespace

int main(int argc, char** argv)
{
    int maxFrames = 0;
    std::filesystem::path requestedModel{};
    std::filesystem::path requestedScene{};
    std::filesystem::path requestedEnvironment{};
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
        else if (std::strcmp(argv[i], "--env") == 0 && i + 1 < argc)
        {
            // Absolute so the environment dropdown can match the scanned library entries.
            std::error_code absoluteError{};
            const std::filesystem::path requested = argv[i + 1];
            requestedEnvironment = std::filesystem::absolute(requested, absoluteError);
            if (absoluteError)
            {
                requestedEnvironment = requested;
            }
        }
    }

    mukki::Log::Init(mukki::LogConfig{});

    int exitCode = EXIT_SUCCESS;
    try
    {
        const std::filesystem::path shaderDirectory =
            std::filesystem::absolute(argv[0]).parent_path() / "shaders";

        SandboxApp app{maxFrames, shaderDirectory, requestedModel, requestedScene, requestedEnvironment};
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
