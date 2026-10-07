#include "Renderer/Deferred/DeferredRenderer.h"

#include "Assets/GltfLoader.h"
#include "Assets/HdrLoader.h"
#include "Renderer/Clouds/CloudNoise.h"
#include "Core/JobSystem.h"
#include "Renderer/GlobalIllumination/GiAtrousSchedule.h"
#include "Renderer/RHI/Device.h"
#include "Scene/SceneDescription.h"
#include "Utilities/Assert.h"
#include "Utilities/Log.h"
#include "Utilities/PathUtils.h"
#include "Utilities/Profiling.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <future>
#include <iterator>
#include <string>

#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_inverse.hpp>

namespace mukki
{
namespace
{

struct Vertex
{
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f};
    glm::vec2 uv{0.0f};
};

float MaximumScale(const glm::mat4& transform)
{
    return std::max(
        {glm::length(glm::vec3{transform[0]}),
         glm::length(glm::vec3{transform[1]}),
         glm::length(glm::vec3{transform[2]})});
}

constexpr Vertex kPlaneVertices[] = {
    {{-10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
    {{-10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
};

constexpr std::uint32_t kPlaneIndices[] = {0, 1, 2, 0, 2, 3};

constexpr std::uint32_t kWhitePixel = 0xFFFFFFFFu;

// Split-sum BRDF integration LUT resolution (see BuildEnvironmentBrdfLut).
constexpr std::uint32_t kBrdfLutSize = 64;

rhi::AddressMode ToRhiAddressMode(GltfWrapMode mode)
{
    switch (mode)
    {
    case GltfWrapMode::ClampToEdge:
        return rhi::AddressMode::ClampToEdge;
    case GltfWrapMode::MirroredRepeat:
        return rhi::AddressMode::MirroredRepeat;
    case GltfWrapMode::Repeat:
    default:
        return rhi::AddressMode::Repeat;
    }
}

} // namespace

bool DeferredRenderer::Init(rhi::Device& device, const Desc& desc)
{
    MRE_ASSERT(!m_initialized, "DeferredRenderer::Init called twice");
    MRE_ASSERT(!desc.shaderDirectory.empty(), "DeferredRenderer requires a shader directory");

    m_device = &device;
    m_jobSystem = desc.jobSystem;

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    if (std::getenv("MRE_GI_OFF") != nullptr)
    {
        m_giSettings.enabled = false;
    }
    if (std::getenv("MRE_REFL_OFF") != nullptr)
    {
        m_reflectionSettings.enabled = false;
    }
    if (std::getenv("MRE_PROBES_OFF") != nullptr)
    {
        m_giSettings.useProbes = false;
    }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    if (m_jobSystem != nullptr)
    {
        m_asyncLoader.SetJobSystem(*m_jobSystem);
        m_secondaryBatchCount = std::max(1u, m_jobSystem->GetThreadCount());
    }

    const rhi::Extent2D extent = device.GetSwapchainInfo().extent;
    if (!m_gbuffer.Init(device, std::max(1u, extent.width), std::max(1u, extent.height)))
    {
        return false;
    }
    m_clusterGrid = ComputeClusterGrid(m_gbuffer.GetWidth(), m_gbuffer.GetHeight());

    m_bufferManager = std::make_unique<BufferManager>(device);
    m_gbufferSampler = device.CreateSampler(rhi::SamplerDesc{});

    const rhi::TextureDesc whiteTextureDesc{
        1,
        1,
        rhi::Format::R8G8B8A8Unorm,
        rhi::TextureUsage::Sampled,
        &kWhitePixel,
        sizeof(kWhitePixel)};
    m_whiteTexture = device.CreateTexture(whiteTextureDesc);
    if (m_whiteTexture == nullptr)
    {
        MRE_CRITICAL("Failed to create the default white texture");
        Shutdown();
        return false;
    }

    const std::array<std::uint8_t, 4> blackPixel{0u, 0u, 0u, 0u};
    const rhi::TextureDesc blackTextureDesc{
        1,
        1,
        rhi::Format::R8G8B8A8Unorm,
        rhi::TextureUsage::Sampled,
        blackPixel.data(),
        blackPixel.size()};
    m_blackTexture = device.CreateTexture(blackTextureDesc);
    if (m_blackTexture == nullptr)
    {
        MRE_CRITICAL("Failed to create the default black texture");
        Shutdown();
        return false;
    }

    // Split-sum BRDF LUT for the image based specular term: r = scale, g = bias.
    {
        std::vector<float> brdfLutPixels{};
        BuildEnvironmentBrdfLut(brdfLutPixels, kBrdfLutSize);
        const rhi::TextureDesc brdfLutDesc{
            kBrdfLutSize,
            kBrdfLutSize,
            rhi::Format::R32G32Sfloat,
            rhi::TextureUsage::Sampled,
            brdfLutPixels.data(),
            brdfLutPixels.size() * sizeof(float)};
        m_brdfLut = device.CreateTexture(brdfLutDesc);
        rhi::SamplerDesc brdfLutSamplerDesc{};
        brdfLutSamplerDesc.addressModeU = rhi::AddressMode::ClampToEdge;
        brdfLutSamplerDesc.addressModeV = rhi::AddressMode::ClampToEdge;
        m_brdfLutSampler = device.CreateSampler(brdfLutSamplerDesc);
        if (m_brdfLut == nullptr || m_brdfLutSampler == nullptr)
        {
            MRE_CRITICAL("Failed to create the environment BRDF LUT");
            Shutdown();
            return false;
        }
        m_environmentSh = ComputeEnvironmentSh(m_environmentSettings);
    }

    if (!LoadShaders(device, desc.shaderDirectory))
    {
        Shutdown();
        return false;
    }

    // The bindless set is created by hand because descriptor indexing with update-after-bind cannot
    // be expressed in shader reflection. Every other set layout comes from the compiled shaders.
    const std::vector<rhi::DescriptorBinding> bindlessBindings{
        rhi::DescriptorBinding{
            0,
            rhi::DescriptorType::SampledImage,
            rhi::ShaderStageMask::FragmentCompute,
            rhi::kMaxBindlessTextures,
            true},
        rhi::DescriptorBinding{
            1,
            rhi::DescriptorType::Sampler,
            rhi::ShaderStageMask::FragmentCompute,
            rhi::kMaxBindlessSamplers},
        rhi::DescriptorBinding{
            2,
            rhi::DescriptorType::StorageBuffer,
            rhi::ShaderStageMask::FragmentCompute,
            rhi::kMaxBindlessBuffers,
            true},
    };
    const std::array<const rhi::Shader*, 2> sceneStages{
        m_gbufferVertexShader.get(),
        m_gbufferFragmentShader.get()};
    const std::array<const rhi::Shader*, 1> lightingStages{m_lightingFragmentShader.get()};
    const std::array<const rhi::Shader*, 1> clusterStages{m_clusterCullingShader.get()};
    const std::array<const rhi::Shader*, 2> shadowStages{
        m_shadowVertexShader.get(),
        m_shadowFragmentShader.get()};
    const std::array<const rhi::Shader*, 1> giTraceStages{m_giTraceShader.get()};
    const std::array<const rhi::Shader*, 1> giTemporalStages{m_giTemporalShader.get()};
    const std::array<const rhi::Shader*, 1> giAtrousStages{m_giAtrousShader.get()};
    const std::array<const rhi::Shader*, 1> giUpsampleStages{m_giUpsampleShader.get()};

    m_sceneSetLayout =
        device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(sceneStages, 0));
    m_bindlessSetLayout = device.CreateDescriptorSetLayout(bindlessBindings);
    m_lightingSetLayout =
        device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(lightingStages, 0));
    m_clusterSetLayout =
        device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(clusterStages, 0));
    m_shadowSetLayout =
        device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(shadowStages, 0));
    m_giTraceSetLayout =
        device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(giTraceStages, 0));
    m_giTemporalSetLayout =
        device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(giTemporalStages, 0));
    m_giAtrousSetLayout =
        device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(giAtrousStages, 0));
    m_giUpsampleSetLayout =
        device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(giUpsampleStages, 0));

    // Ray traced sun shadows are optional; only devices with ray query support get the resources.
    if (device.IsRayTracingSupported())
    {
        const std::array<const rhi::Shader*, 1> rtShadowStages{m_rtShadowShader.get()};
        m_rtShadowSetLayout =
            device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(rtShadowStages, 0));

        const std::array<const rhi::Shader*, 1> giProbeStages{m_giProbeShader.get()};
        m_giProbeSetLayout =
            device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(giProbeStages, 0));

        const std::array<const rhi::Shader*, 1> reflectionTraceStages{m_rtReflectionShader.get()};
        m_reflectionTraceSetLayout =
            device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(reflectionTraceStages, 0));

        const std::array<const rhi::Shader*, 1> reflectionTemporalStages{
            m_reflectionTemporalShader.get()};
        m_reflectionTemporalSetLayout =
            device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(reflectionTemporalStages, 0));

        const std::array<const rhi::Shader*, 1> reflectionUpsampleStages{
            m_reflectionUpsampleShader.get()};
        m_reflectionUpsampleSetLayout =
            device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(reflectionUpsampleStages, 0));
    }

    const std::array<const rhi::Shader*, 1> cloudStages{m_cloudShader.get()};
    m_cloudSetLayout =
        device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(cloudStages, 0));
    const std::array<const rhi::Shader*, 1> cloudUpsampleStages{m_cloudUpsampleShader.get()};
    m_cloudUpsampleSetLayout =
        device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(cloudUpsampleStages, 0));
    const std::array<const rhi::Shader*, 1> cloudReprojectStages{m_cloudReprojectShader.get()};
    m_cloudReprojectSetLayout =
        device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(cloudReprojectStages, 0));
    const std::array<const rhi::Shader*, 1> cloudGodRayStages{m_cloudGodRayShader.get()};
    m_cloudGodRaySetLayout =
        device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(cloudGodRayStages, 0));

    MRE_ASSERT(
        m_sceneSetLayout != nullptr && m_bindlessSetLayout != nullptr &&
            m_lightingSetLayout != nullptr && m_clusterSetLayout != nullptr && m_shadowSetLayout != nullptr,
        "Failed to create a descriptor set layout");

    m_bindlessSet = device.CreateDescriptorSet(*m_bindlessSetLayout);
    m_bindlessSet->BindTexture(0, *m_whiteTexture, m_whiteTexture->GetBindlessIndex());
    // Slot 0 doubles as the default for zero-initialized texture indices; unwritten descriptor
    // array slots are undefined and sampling them can fault the GPU.
    m_bindlessSet->BindTexture(0, *m_whiteTexture, 0u);
    static_cast<void>(GetSamplerIndex(rhi::AddressMode::ClampToEdge, rhi::AddressMode::ClampToEdge));

    const std::uint32_t framesInFlight = device.GetFramesInFlight();
    m_tlas.resize(framesInFlight);
    m_tlasInstanceCounts.assign(framesInFlight, 0);
    m_geometryDataBuffers.resize(framesInFlight);
    m_objectDataBuffers.clear();
    m_uniformBuffers.reserve(framesInFlight);
    m_sceneSets.reserve(framesInFlight);
    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        m_uniformBuffers.push_back(device.CreateBuffer(
            rhi::BufferDesc{sizeof(SceneUniform), rhi::BufferUsage::Uniform, true},
            &m_sceneUniform,
            sizeof(SceneUniform)));

        m_objectDataBuffers.push_back(device.CreateBuffer(
            rhi::BufferDesc{kMaxObjectData * sizeof(GpuObjectData), rhi::BufferUsage::Storage, true}));
        MRE_ASSERT(m_objectDataBuffers.back() != nullptr, "Failed to create the object data buffer");

        std::unique_ptr<rhi::DescriptorSet> sceneSet = device.CreateDescriptorSet(*m_sceneSetLayout);
        sceneSet->BindBuffer(0, *m_uniformBuffers.back());
        sceneSet->BindBuffer(1, *m_objectDataBuffers.back());
        m_sceneSets.push_back(std::move(sceneSet));
    }

    if (!CreateClusterResources(device) || !CreateShadowResources(device))
    {
        Shutdown();
        return false;
    }

    if (!CreateGiResources(device))
    {
        Shutdown();
        return false;
    }

    if (!CreateCloudResources(device))
    {
        Shutdown();
        return false;
    }

    m_gbufferSecondaries.clear();
    m_gbufferSecondaries.resize(framesInFlight);
    m_shadowSecondaries.clear();
    m_shadowSecondaries.resize(framesInFlight);
    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        m_gbufferSecondaries[frame].reserve(m_secondaryBatchCount);
        for (std::uint32_t batch = 0; batch < m_secondaryBatchCount; ++batch)
        {
            m_gbufferSecondaries[frame].push_back(device.CreateSecondaryCommandList());
        }

        m_shadowSecondaries[frame].reserve(kMaxShadowViews);
        for (std::uint32_t view = 0; view < kMaxShadowViews; ++view)
        {
            m_shadowSecondaries[frame].push_back(device.CreateSecondaryCommandList());
        }
    }

    if (!CreateScene() || !CreatePipelines(device))
    {
        Shutdown();
        return false;
    }

    if (!CreateRtShadowResources(device))
    {
        Shutdown();
        return false;
    }

    if (!CreateGiProbeResources(device))
    {
        Shutdown();
        return false;
    }

    m_sceneUniform.ambientColor = glm::vec4(0.05f, 0.06f, 0.09f, 1.0f);

    m_initialized = true;
    MRE_INFO("Deferred renderer initialized ({}x{})", m_gbuffer.GetWidth(), m_gbuffer.GetHeight());
    return true;
}

void DeferredRenderer::Shutdown()
{
    UnloadModel();

    m_lightingSets.clear();
    m_clusterSets.clear();
    m_sceneSets.clear();
    m_shadowSets.clear();
    m_shadowPipeline.reset();
    m_clusterCullingPipeline.reset();
    m_lightingPipeline.reset();
    m_gbufferPipeline.reset();
    m_shadowFragmentShader.reset();
    m_shadowVertexShader.reset();
    m_clusterCullingShader.reset();
    m_lightingFragmentShader.reset();
    m_lightingVertexShader.reset();
    m_gbufferFragmentShader.reset();
    m_gbufferVertexShader.reset();
    m_sceneDrawItems.clear();
    m_gbufferSecondaries.clear();
    m_shadowSecondaries.clear();
    m_tlas.clear();
    m_geometryDataBuffers.clear();
    m_geometryData.clear();
    m_objectDataBuffers.clear();
    m_previousModels.clear();
    m_planeBlas.reset();
    m_tlasInstanceCounts.clear();
    m_blasDirty = false;
    m_gi.clear();
    m_giLightBuffers.clear();
    m_giLights.clear();
    m_giAtrousPipeline.reset();
    m_giUpsamplePipeline.reset();
    m_giTemporalPipeline.reset();
    m_giTracePipeline.reset();
    m_giAtrousShader.reset();
    m_giUpsampleShader.reset();
    m_giTemporalShader.reset();
    m_giTraceShader.reset();
    m_giUpsampleSetLayout.reset();
    m_giAtrousSetLayout.reset();
    m_giTemporalSetLayout.reset();
    m_giTraceSetLayout.reset();
    m_shadowMap.reset();
    m_spotShadowMap.reset();
    m_pointShadowMaps.clear();
    for (PointShadowSlotState& state : m_pointShadowSlotStates)
    {
        state = PointShadowSlotState{};
    }
    m_shadowSampler.reset();
    m_shadowUniformBuffers.clear();
    m_shadowViews.clear();
    m_shadowCullItems.clear();
    m_spotShadowLights.clear();
    m_pointShadowLights.clear();
    m_lightBuffers.clear();
    m_clusterUniformBuffers.clear();
    m_clusterCountBuffers.clear();
    m_clusterIndexBuffers.clear();
    m_lights.clear();
    m_gpuLights.clear();
    m_zeroClusterCounts.clear();
    m_shadowSetLayout.reset();
    m_clusterSetLayout.reset();
    m_lightingSetLayout.reset();
    m_bindlessSetLayout.reset();
    m_sceneSetLayout.reset();
    m_bindlessSet.reset();
    m_whiteTexture.reset();
    m_blackTexture.reset();
    m_brdfLut.reset();
    m_brdfLutSampler.reset();
    m_environmentMap.reset();
    m_environmentMapBound = false;
    m_loadedEnvironmentMapPath.clear();
    m_cloudPipeline.reset();
    m_cloudUpsamplePipeline.reset();
    m_cloudReprojectPipeline.reset();
    m_cloudGodRayPipeline.reset();
    m_cloudSetLayout.reset();
    m_cloudUpsampleSetLayout.reset();
    m_cloudReprojectSetLayout.reset();
    m_cloudGodRaySetLayout.reset();
    m_cloudShader.reset();
    m_cloudUpsampleShader.reset();
    m_cloudReprojectShader.reset();
    m_cloudGodRayShader.reset();
    m_cloudBaseNoise.reset();
    m_cloudDetailNoise.reset();
    m_cloudCurlNoise.reset();
    m_cloudWeatherMap.reset();
    m_cloudSampler.reset();
    m_cloudEmpty.reset();
    m_cloudUniformBuffers.clear();
    m_cloudSets.clear();
    m_cloudUpsampleSets.clear();
    m_cloudReprojectSets.clear();
    m_cloudGodRaySets.clear();
    m_cloudCurrent.clear();
    m_cloudDistance.clear();
    m_cloudMask.clear();
    m_cloudFull.clear();
    m_cloudHistory.clear();
    m_cloudHistoryDistance.clear();
    m_cloudHistoryMask.clear();
    m_cloudGodRays.clear();
    m_cloudHistoryParity.clear();
    m_cloudHistoryValid.clear();
    m_cloudNoiseReady = false;
    m_cloudFrameCounter = 0;
    m_bindlessSamplers.clear();
    m_samplerCache.clear();
    m_gbufferSampler.reset();
    m_planeIndexBuffer = nullptr;
    m_planeVertexBuffer = nullptr;
    m_uniformBuffers.clear();
    m_bufferManager.reset();
    m_gbuffer.Shutdown();
    m_device = nullptr;
    m_jobSystem = nullptr;
    m_modelScale = 1.0f;
    m_lightingSetsDirty = true;
    m_clusterDebugView = false;
    m_initialized = false;
}

void DeferredRenderer::Resize(std::uint32_t width, std::uint32_t height)
{
    if (!m_initialized || width == 0 || height == 0)
    {
        return;
    }

    if (width == m_gbuffer.GetWidth() && height == m_gbuffer.GetHeight())
    {
        return;
    }

    MRE_ASSERT(m_device != nullptr, "DeferredRenderer requires a device");

    m_device->WaitIdle();

    m_lightingSets.clear();
    m_clusterSets.clear();
    m_lightingSetsDirty = true;

    if (!m_gbuffer.Resize(*m_device, width, height))
    {
        MRE_CRITICAL("Failed to resize the G-buffer");
        return;
    }

    m_clusterGrid = ComputeClusterGrid(m_gbuffer.GetWidth(), m_gbuffer.GetHeight());
    if (!CreateClusterResources(*m_device))
    {
        MRE_CRITICAL("Failed to recreate the clustered light resources");
    }

    if (!CreateGiResources(*m_device))
    {
        MRE_CRITICAL("Failed to recreate the global illumination resources");
    }

    if (!CreateCloudResources(*m_device))
    {
        MRE_CRITICAL("Failed to recreate the cloud resources");
    }

    if (!CreateRtShadowResources(*m_device))
    {
        MRE_CRITICAL("Failed to recreate the ray traced shadow resources");
    }
}

bool DeferredRenderer::LoadModel(const std::filesystem::path& path)
{
    MRE_ASSERT(m_initialized, "DeferredRenderer::LoadModel called before DeferredRenderer::Init");
    MRE_ASSERT(m_device != nullptr, "DeferredRenderer requires a device");

    GltfModel model{};
    if (!LoadGltfModel(path, model))
    {
        return false;
    }

    m_device->WaitIdle();
    ClearModelInstances();
    ClearModels();

    const std::optional<std::size_t> modelIndex = UploadModel(path, model);
    if (!modelIndex.has_value())
    {
        return false;
    }

    AddModelInstance(*modelIndex, glm::mat4{1.0f});
    return true;
}

bool DeferredRenderer::LoadScene(const SceneDescription& scene)
{
    MRE_ASSERT(m_initialized, "DeferredRenderer::LoadScene called before DeferredRenderer::Init");
    MRE_ASSERT(m_device != nullptr, "DeferredRenderer requires a device");

    m_device->WaitIdle();
    ClearModelInstances();
    ClearModels();

    std::vector<std::size_t> modelIndices{};
    modelIndices.reserve(scene.objects.size());
    for (const SceneObject& object : scene.objects)
    {
        const std::filesystem::path modelPath = ResolveModelPath(scene, object);
        const std::optional<std::size_t> modelIndex = LoadModelResources(modelPath);
        if (!modelIndex.has_value())
        {
            MRE_ERROR(
                "Failed to load model '{}' for scene object '{}'",
                PathToUtf8(modelPath),
                object.name);
            return false;
        }
        modelIndices.push_back(*modelIndex);
    }

    std::vector<Light> lights{};
    lights.reserve(scene.lights.size());
    for (const SceneLight& sceneLight : scene.lights)
    {
        lights.push_back(sceneLight.light);
    }
    SetLights(lights);

    for (std::size_t index = 0; index < scene.objects.size(); ++index)
    {
        AddModelInstance(modelIndices[index], BuildObjectTransform(scene.objects[index]));
    }

    m_modelName = scene.name;
    m_modelVertexCount = 0;
    m_modelTriangleCount = 0;
    for (const std::size_t modelIndex : modelIndices)
    {
        m_modelVertexCount += m_models[modelIndex]->vertexCount;
        m_modelTriangleCount += m_models[modelIndex]->triangleCount;
    }

    MRE_INFO(
        "Loaded scene '{}' ({} objects, {} lights, {} vertices)",
        scene.name,
        scene.objects.size(),
        scene.lights.size(),
        m_modelVertexCount);
    return true;
}

void DeferredRenderer::ClearScene()
{
    UnloadModel();
    SetLights({});
}

void DeferredRenderer::RequestModelLoad(const std::filesystem::path& path)
{
    MRE_ASSERT(m_initialized, "DeferredRenderer::RequestModelLoad called before DeferredRenderer::Init");
    m_asyncLoader.Load(path);
}

bool DeferredRenderer::PollModelLoad()
{
    MRE_ASSERT(m_initialized, "DeferredRenderer::PollModelLoad called before DeferredRenderer::Init");

    if (!m_asyncLoader.IsReady())
    {
        return false;
    }

    GltfLoadResult result = m_asyncLoader.Consume();
    if (!result.success)
    {
        MRE_ERROR("Asynchronous load of '{}' failed", PathToUtf8(result.path));
        return false;
    }

    m_device->WaitIdle();
    ClearModelInstances();
    ClearModels();

    const std::optional<std::size_t> modelIndex = UploadModel(result.path, result.model);
    if (!modelIndex.has_value())
    {
        return false;
    }

    AddModelInstance(*modelIndex, glm::mat4{1.0f});
    return true;
}

std::optional<std::size_t> DeferredRenderer::UploadModel(const std::filesystem::path& path, GltfModel& model)
{
    MRE_PROFILE_ZONE_N("UploadModel");
    MRE_ASSERT(m_device != nullptr, "DeferredRenderer requires a device");

    auto resources = std::make_unique<ModelResources>();
    resources->name = PathToUtf8(path.filename());
    resources->vertexCount = model.vertexCount;
    resources->triangleCount = model.triangleCount;

    const glm::vec3 center = (model.minBounds + model.maxBounds) * 0.5f;
    const glm::vec3 extent = model.maxBounds - model.minBounds;
    const float maxExtent = std::max({extent.x, extent.y, extent.z});
    const float scale = maxExtent > 0.0f ? 2.5f / maxExtent : 1.0f;
    const float lift = (center.y - model.minBounds.y) * scale;

    const glm::mat4 fit = glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, lift, 0.0f}) *
                          glm::scale(glm::mat4{1.0f}, glm::vec3{scale}) *
                          glm::translate(glm::mat4{1.0f}, -center);

    const glm::vec3 fittedMin = glm::vec3{fit * glm::vec4{model.minBounds, 1.0f}};
    const glm::vec3 fittedMax = glm::vec3{fit * glm::vec4{model.maxBounds, 1.0f}};
    resources->fittedBounds.center = (fittedMin + fittedMax) * 0.5f;
    resources->fittedBounds.radius = glm::length(fittedMax - fittedMin) * 0.5f;

    const auto createTextureFromImage = [this, &model](int imageIndex, rhi::Format format) {
        std::unique_ptr<rhi::Texture> texture{};
        if (imageIndex < 0 || imageIndex >= static_cast<int>(model.images.size()))
        {
            return texture;
        }

        const GltfImage& image = model.images[static_cast<std::size_t>(imageIndex)];
        if (image.width == 0 || image.height == 0 || image.pixels.empty())
        {
            return texture;
        }

        const rhi::TextureDesc textureDesc{
            image.width,
            image.height,
            format,
            rhi::TextureUsage::Sampled,
            image.pixels.data(),
            image.pixels.size()};
        return m_device->CreateTexture(textureDesc);
    };

    const auto textureImageIndex = [&model](int textureIndex) {
        if (textureIndex < 0 || textureIndex >= static_cast<int>(model.textures.size()))
        {
            return -1;
        }
        return model.textures[static_cast<std::size_t>(textureIndex)].imageIndex;
    };

    resources->materials.reserve(model.materials.size());
    for (const GltfMaterial& source : model.materials)
    {
        Material material{};
        material.baseColorTexture =
            createTextureFromImage(textureImageIndex(source.baseColorTexture), rhi::Format::R8G8B8A8Srgb);
        material.metallicRoughnessTexture =
            createTextureFromImage(textureImageIndex(source.metallicRoughnessTexture), rhi::Format::R8G8B8A8Unorm);
        material.specularTexture =
            createTextureFromImage(textureImageIndex(source.specularTexture), rhi::Format::R8G8B8A8Unorm);
        material.specularColorTexture =
            createTextureFromImage(textureImageIndex(source.specularColorTexture), rhi::Format::R8G8B8A8Unorm);
        material.transmissionTexture =
            createTextureFromImage(textureImageIndex(source.transmissionTexture), rhi::Format::R8G8B8A8Unorm);
        material.thicknessTexture =
            createTextureFromImage(textureImageIndex(source.thicknessTexture), rhi::Format::R8G8B8A8Unorm);
        material.sheenColorTexture =
            createTextureFromImage(textureImageIndex(source.sheenColorTexture), rhi::Format::R8G8B8A8Srgb);
        material.sheenRoughnessTexture =
            createTextureFromImage(textureImageIndex(source.sheenRoughnessTexture), rhi::Format::R8G8B8A8Unorm);
        // Emissive textures carry light, so they stay linear instead of sRGB.
        material.emissiveTexture =
            createTextureFromImage(textureImageIndex(source.emissiveTexture), rhi::Format::R8G8B8A8Unorm);

        if (material.baseColorTexture != nullptr)
        {
            m_bindlessSet->BindTexture(0, *material.baseColorTexture, material.baseColorTexture->GetBindlessIndex());
        }
        if (material.metallicRoughnessTexture != nullptr)
        {
            m_bindlessSet->BindTexture(
                0,
                *material.metallicRoughnessTexture,
                material.metallicRoughnessTexture->GetBindlessIndex());
        }
        if (material.specularTexture != nullptr)
        {
            m_bindlessSet->BindTexture(
                0,
                *material.specularTexture,
                material.specularTexture->GetBindlessIndex());
        }
        if (material.specularColorTexture != nullptr)
        {
            m_bindlessSet->BindTexture(
                0,
                *material.specularColorTexture,
                material.specularColorTexture->GetBindlessIndex());
        }
        if (material.transmissionTexture != nullptr)
        {
            m_bindlessSet->BindTexture(
                0,
                *material.transmissionTexture,
                material.transmissionTexture->GetBindlessIndex());
        }
        if (material.thicknessTexture != nullptr)
        {
            m_bindlessSet->BindTexture(
                0,
                *material.thicknessTexture,
                material.thicknessTexture->GetBindlessIndex());
        }
        if (material.sheenColorTexture != nullptr)
        {
            m_bindlessSet->BindTexture(
                0,
                *material.sheenColorTexture,
                material.sheenColorTexture->GetBindlessIndex());
        }
        if (material.sheenRoughnessTexture != nullptr)
        {
            m_bindlessSet->BindTexture(
                0,
                *material.sheenRoughnessTexture,
                material.sheenRoughnessTexture->GetBindlessIndex());
        }
        if (material.emissiveTexture != nullptr)
        {
            m_bindlessSet->BindTexture(
                0,
                *material.emissiveTexture,
                material.emissiveTexture->GetBindlessIndex());
        }

        resources->materials.push_back(std::move(material));
    }

    const bool rayTracingSupported = m_device->IsRayTracingSupported();
    resources->vertexBuffers.reserve(model.primitives.size());
    resources->indexBuffers.reserve(model.primitives.size());
    resources->accelerationStructures.reserve(model.primitives.size());
    for (const GltfPrimitive& primitive : model.primitives)
    {
        const std::uint64_t vertexBytes = sizeof(GltfVertex) * primitive.vertices.size();
        const std::uint64_t indexBytes = sizeof(std::uint32_t) * primitive.indices.size();

        resources->vertexBuffers.push_back(m_device->CreateBuffer(
            rhi::BufferDesc{vertexBytes, rhi::BufferUsage::Vertex, false, rayTracingSupported},
            primitive.vertices.data(),
            vertexBytes));
        resources->indexBuffers.push_back(m_device->CreateBuffer(
            rhi::BufferDesc{indexBytes, rhi::BufferUsage::Index, false, rayTracingSupported},
            primitive.indices.data(),
            indexBytes));

        // Register the buffers in the bindless storage buffer array used by ray traced hit shading.
        resources->vertexBufferIndices.push_back(m_nextBindlessBuffer++);
        resources->indexBufferIndices.push_back(m_nextBindlessBuffer++);
        m_bindlessSet->BindStorageBuffer(2, *resources->vertexBuffers.back(), resources->vertexBufferIndices.back());
        m_bindlessSet->BindStorageBuffer(2, *resources->indexBuffers.back(), resources->indexBufferIndices.back());

        if (rayTracingSupported)
        {
            const rhi::BottomLevelAccelerationStructureDesc blasDesc{
                resources->vertexBuffers.back().get(),
                static_cast<std::uint32_t>(sizeof(GltfVertex)),
                static_cast<std::uint32_t>(primitive.vertices.size()),
                resources->indexBuffers.back().get(),
                static_cast<std::uint32_t>(primitive.indices.size()),
                true};
            resources->accelerationStructures.push_back(
                m_device->CreateBottomLevelAccelerationStructure(blasDesc));
        }
    }
    m_blasDirty = true;

    for (const GltfNode& node : model.nodes)
    {
        if (node.meshIndex < 0 || node.meshIndex >= static_cast<int>(model.meshPrimitives.size()))
        {
            continue;
        }

        for (const std::uint32_t primitiveIndex : model.meshPrimitives[static_cast<std::size_t>(node.meshIndex)])
        {
            const GltfPrimitive& primitive = model.primitives[primitiveIndex];

            DrawItem item{};
            item.vertexBuffer = resources->vertexBuffers[primitiveIndex].get();
            item.indexBuffer = resources->indexBuffers[primitiveIndex].get();
            item.indexCount = static_cast<std::uint32_t>(primitive.indices.size());
            item.firstIndex = 0;
            item.pushConstants.model = fit * node.worldTransform;
            if (primitiveIndex < resources->accelerationStructures.size())
            {
                item.blas = resources->accelerationStructures[primitiveIndex].get();
            }
            item.vertexBufferIndex = resources->vertexBufferIndices[primitiveIndex];
            item.indexBufferIndex = resources->indexBufferIndices[primitiveIndex];

            const std::uint32_t whiteIndex = m_whiteTexture->GetBindlessIndex();
            const std::uint32_t defaultSamplerIndex =
                GetSamplerIndex(rhi::AddressMode::Repeat, rhi::AddressMode::Repeat);
            item.pushConstants.textureIndices = glm::uvec2{whiteIndex, whiteIndex};
            item.specular = glm::vec4{1.0f, 1.0f, 1.0f, 1.0f};
            item.specularTextureIndices = glm::uvec4{whiteIndex, whiteIndex, 0u, 0u};
            item.specularSamplerIndices =
                glm::uvec4{defaultSamplerIndex, defaultSamplerIndex, 0u, 0u};
            item.volume = glm::vec4{0.0f, 1.5f, 0.0f, 0.0f};
            item.attenuation = glm::vec4{1.0f, 1.0f, 1.0f, 0.0f};
            item.volumeTextureIndices = glm::uvec4{whiteIndex, whiteIndex, 0u, 0u};
            item.volumeSamplerIndices = glm::uvec4{defaultSamplerIndex, defaultSamplerIndex, 0u, 0u};
            item.alpha = glm::vec4{0.0f, 0.5f, 0.0f, 0.0f};
            item.sheen = glm::vec4{0.0f, 0.0f, 0.0f, 0.0f};
            item.sheenTextureIndices = glm::uvec4{whiteIndex, whiteIndex, 0u, 0u};
            item.sheenSamplerIndices = glm::uvec4{defaultSamplerIndex, defaultSamplerIndex, 0u, 0u};
            item.emissive = glm::vec4{0.0f};
            item.emissiveTextureIndices = glm::uvec4{whiteIndex, 0u, 0u, 0u};
            item.emissiveSamplerIndices = glm::uvec4{defaultSamplerIndex, 0u, 0u, 0u};

            if (primitive.materialIndex >= 0 &&
                primitive.materialIndex < static_cast<int>(resources->materials.size()))
            {
                const GltfMaterial& material = model.materials[static_cast<std::size_t>(primitive.materialIndex)];
                item.pushConstants.baseColorFactor = material.baseColorFactor;
                item.pushConstants.params = glm::vec4{
                    material.roughnessFactor,
                    material.metallicFactor,
                    material.alphaMask ? 1.0f : 0.0f,
                    material.alphaCutoff};
                item.specular = glm::vec4{
                    material.specularFactor,
                    material.specularColorFactor.r,
                    material.specularColorFactor.g,
                    material.specularColorFactor.b};
                item.volume = glm::vec4{
                    material.transmissionFactor,
                    material.ior,
                    material.thicknessFactor,
                    material.dispersion};
                item.attenuation = glm::vec4{
                    material.attenuationColor.r,
                    material.attenuationColor.g,
                    material.attenuationColor.b,
                    material.attenuationDistance};
                item.alpha = glm::vec4{
                    material.alphaBlend ? 1.0f : 0.0f,
                    material.alphaCutoff,
                    0.0f,
                    0.0f};
                item.alphaBlend = material.alphaBlend;
                item.sheen = glm::vec4{
                    material.sheenColorFactor.r,
                    material.sheenColorFactor.g,
                    material.sheenColorFactor.b,
                    material.sheenRoughnessFactor};
                item.emissive = glm::vec4{
                    material.emissiveFactor.r,
                    material.emissiveFactor.g,
                    material.emissiveFactor.b,
                    0.0f};

                const Material& gpuMaterial = resources->materials[static_cast<std::size_t>(primitive.materialIndex)];
                if (gpuMaterial.baseColorTexture != nullptr)
                {
                    item.pushConstants.textureIndices.x = gpuMaterial.baseColorTexture->GetBindlessIndex();
                }
                if (gpuMaterial.metallicRoughnessTexture != nullptr)
                {
                    item.pushConstants.textureIndices.y = gpuMaterial.metallicRoughnessTexture->GetBindlessIndex();
                }

                if (material.baseColorTexture >= 0 &&
                    material.baseColorTexture < static_cast<int>(model.textures.size()))
                {
                    const GltfTexture& texture = model.textures[static_cast<std::size_t>(material.baseColorTexture)];
                    item.pushConstants.samplerIndices.x =
                        GetSamplerIndex(ToRhiAddressMode(texture.wrapS), ToRhiAddressMode(texture.wrapT));
                }
                if (material.metallicRoughnessTexture >= 0 &&
                    material.metallicRoughnessTexture < static_cast<int>(model.textures.size()))
                {
                    const GltfTexture& texture =
                        model.textures[static_cast<std::size_t>(material.metallicRoughnessTexture)];
                    item.pushConstants.samplerIndices.y =
                        GetSamplerIndex(ToRhiAddressMode(texture.wrapS), ToRhiAddressMode(texture.wrapT));
                }

                if (gpuMaterial.specularTexture != nullptr)
                {
                    item.specularTextureIndices.x = gpuMaterial.specularTexture->GetBindlessIndex();
                }
                if (gpuMaterial.specularColorTexture != nullptr)
                {
                    item.specularTextureIndices.y = gpuMaterial.specularColorTexture->GetBindlessIndex();
                }
                if (material.specularTexture >= 0 &&
                    material.specularTexture < static_cast<int>(model.textures.size()))
                {
                    const GltfTexture& texture =
                        model.textures[static_cast<std::size_t>(material.specularTexture)];
                    item.specularSamplerIndices.x =
                        GetSamplerIndex(ToRhiAddressMode(texture.wrapS), ToRhiAddressMode(texture.wrapT));
                }
                if (material.specularColorTexture >= 0 &&
                    material.specularColorTexture < static_cast<int>(model.textures.size()))
                {
                    const GltfTexture& texture =
                        model.textures[static_cast<std::size_t>(material.specularColorTexture)];
                    item.specularSamplerIndices.y =
                        GetSamplerIndex(ToRhiAddressMode(texture.wrapS), ToRhiAddressMode(texture.wrapT));
                }

                if (gpuMaterial.transmissionTexture != nullptr)
                {
                    item.volumeTextureIndices.x = gpuMaterial.transmissionTexture->GetBindlessIndex();
                }
                if (gpuMaterial.thicknessTexture != nullptr)
                {
                    item.volumeTextureIndices.y = gpuMaterial.thicknessTexture->GetBindlessIndex();
                }
                if (material.transmissionTexture >= 0 &&
                    material.transmissionTexture < static_cast<int>(model.textures.size()))
                {
                    const GltfTexture& texture =
                        model.textures[static_cast<std::size_t>(material.transmissionTexture)];
                    item.volumeSamplerIndices.x =
                        GetSamplerIndex(ToRhiAddressMode(texture.wrapS), ToRhiAddressMode(texture.wrapT));
                }
                if (material.thicknessTexture >= 0 &&
                    material.thicknessTexture < static_cast<int>(model.textures.size()))
                {
                    const GltfTexture& texture =
                        model.textures[static_cast<std::size_t>(material.thicknessTexture)];
                    item.volumeSamplerIndices.y =
                        GetSamplerIndex(ToRhiAddressMode(texture.wrapS), ToRhiAddressMode(texture.wrapT));
                }

                if (gpuMaterial.sheenColorTexture != nullptr)
                {
                    item.sheenTextureIndices.x = gpuMaterial.sheenColorTexture->GetBindlessIndex();
                }
                if (gpuMaterial.sheenRoughnessTexture != nullptr)
                {
                    item.sheenTextureIndices.y = gpuMaterial.sheenRoughnessTexture->GetBindlessIndex();
                }
                if (material.sheenColorTexture >= 0 &&
                    material.sheenColorTexture < static_cast<int>(model.textures.size()))
                {
                    const GltfTexture& texture =
                        model.textures[static_cast<std::size_t>(material.sheenColorTexture)];
                    item.sheenSamplerIndices.x =
                        GetSamplerIndex(ToRhiAddressMode(texture.wrapS), ToRhiAddressMode(texture.wrapT));
                }
                if (material.sheenRoughnessTexture >= 0 &&
                    material.sheenRoughnessTexture < static_cast<int>(model.textures.size()))
                {
                    const GltfTexture& texture =
                        model.textures[static_cast<std::size_t>(material.sheenRoughnessTexture)];
                    item.sheenSamplerIndices.y =
                        GetSamplerIndex(ToRhiAddressMode(texture.wrapS), ToRhiAddressMode(texture.wrapT));
                }

                if (gpuMaterial.emissiveTexture != nullptr)
                {
                    item.emissiveTextureIndices.x = gpuMaterial.emissiveTexture->GetBindlessIndex();
                }
                if (material.emissiveTexture >= 0 &&
                    material.emissiveTexture < static_cast<int>(model.textures.size()))
                {
                    const GltfTexture& texture =
                        model.textures[static_cast<std::size_t>(material.emissiveTexture)];
                    item.emissiveSamplerIndices.x =
                        GetSamplerIndex(ToRhiAddressMode(texture.wrapS), ToRhiAddressMode(texture.wrapT));
                }
            }
            else
            {
                item.pushConstants.baseColorFactor = glm::vec4{0.8f, 0.8f, 0.8f, 1.0f};
                item.pushConstants.params = glm::vec4{0.6f, 0.0f, 0.0f, 0.0f};
            }

            resources->drawItems.push_back(item);
        }
    }

    std::size_t textureCount = 0;
    for (const Material& material : resources->materials)
    {
        if (material.baseColorTexture != nullptr)
        {
            ++textureCount;
        }
        if (material.metallicRoughnessTexture != nullptr)
        {
            ++textureCount;
        }
        if (material.specularTexture != nullptr)
        {
            ++textureCount;
        }
        if (material.specularColorTexture != nullptr)
        {
            ++textureCount;
        }
        if (material.emissiveTexture != nullptr)
        {
            ++textureCount;
        }
    }

    const std::size_t modelIndex = m_models.size();
    m_models.push_back(std::move(resources));

    m_modelName = m_models[modelIndex]->name;
    m_modelVertexCount = m_models[modelIndex]->vertexCount;
    m_modelTriangleCount = m_models[modelIndex]->triangleCount;

    MRE_INFO(
        "Model '{}' uploaded ({} draw items, {} textures)",
        m_modelName,
        m_models[modelIndex]->drawItems.size(),
        textureCount);
    return modelIndex;
}

std::optional<std::size_t> DeferredRenderer::LoadModelResources(const std::filesystem::path& path)
{
    std::error_code absoluteError{};
    const std::filesystem::path absolute = std::filesystem::absolute(path, absoluteError);
    const std::string key =
        PathToUtf8((absoluteError ? path : absolute).lexically_normal());

    if (const auto existing = m_modelCache.find(key); existing != m_modelCache.end())
    {
        return existing->second;
    }

    GltfModel model{};
    if (!LoadGltfModel(path, model))
    {
        return std::nullopt;
    }

    const std::optional<std::size_t> modelIndex = UploadModel(path, model);
    if (modelIndex.has_value())
    {
        m_modelCache.emplace(key, *modelIndex);
    }
    return modelIndex;
}

void DeferredRenderer::AddModelInstance(std::size_t modelIndex, const glm::mat4& transform)
{
    MRE_ASSERT(modelIndex < m_models.size(), "Model index out of range");

    const ModelResources& model = *m_models[modelIndex];
    const glm::mat4 scaleMatrix = glm::scale(glm::mat4{1.0f}, glm::vec3{m_modelScale});

    SceneObjectInstance sceneObject{};
    sceneObject.modelIndex = modelIndex;
    sceneObject.firstDrawItem = m_modelDrawItems.size();
    sceneObject.transform = transform;
    sceneObject.bounds.center = glm::vec3{transform * glm::vec4{model.fittedBounds.center, 1.0f}};
    sceneObject.bounds.radius = model.fittedBounds.radius * MaximumScale(transform);

    for (const DrawItem& source : model.drawItems)
    {
        DrawItem instance = source;
        const glm::mat4 baseTransform = transform * source.pushConstants.model;
        instance.pushConstants.model = scaleMatrix * baseTransform;
        m_modelDrawItems.push_back(instance);
        m_modelBaseTransforms.push_back(baseTransform);
    }

    sceneObject.drawItemCount = m_modelDrawItems.size() - sceneObject.firstDrawItem;
    m_objectInstances.push_back(sceneObject);
    ++m_geometryRevision;
}

void DeferredRenderer::UpdateSceneObjectTransform(std::size_t objectIndex, const glm::mat4& transform)
{
    MRE_ASSERT(objectIndex < m_objectInstances.size(), "Scene object index out of range");

    SceneObjectInstance& sceneObject = m_objectInstances[objectIndex];
    sceneObject.transform = transform;

    const ModelResources& model = *m_models[sceneObject.modelIndex];
    sceneObject.bounds.center = glm::vec3{transform * glm::vec4{model.fittedBounds.center, 1.0f}};
    sceneObject.bounds.radius = model.fittedBounds.radius * MaximumScale(transform);
    MRE_ASSERT(
        sceneObject.firstDrawItem + sceneObject.drawItemCount <= m_modelDrawItems.size(),
        "Scene object draw items are out of range");

    const glm::mat4 scaleMatrix = glm::scale(glm::mat4{1.0f}, glm::vec3{m_modelScale});
    for (std::size_t index = 0; index < sceneObject.drawItemCount; ++index)
    {
        const DrawItem& source = model.drawItems[index];
        const std::size_t drawIndex = sceneObject.firstDrawItem + index;
        const glm::mat4 baseTransform = transform * source.pushConstants.model;
        m_modelDrawItems[drawIndex].pushConstants.model = scaleMatrix * baseTransform;
        m_modelBaseTransforms[drawIndex] = baseTransform;
    }
    ++m_geometryRevision;
}

bool DeferredRenderer::GetSceneBounds(glm::vec3& outMin, glm::vec3& outMax) const
{
    if (m_objectInstances.empty())
    {
        return false;
    }

    glm::vec3 boundsMin{std::numeric_limits<float>::max()};
    glm::vec3 boundsMax{std::numeric_limits<float>::lowest()};
    for (const SceneObjectInstance& sceneObject : m_objectInstances)
    {
        const glm::vec3 center = sceneObject.bounds.center * m_modelScale;
        const float radius = sceneObject.bounds.radius * m_modelScale;
        boundsMin = glm::min(boundsMin, center - radius);
        boundsMax = glm::max(boundsMax, center + radius);
    }

    outMin = boundsMin;
    outMax = boundsMax;
    return true;
}

void DeferredRenderer::ClearModelInstances()
{
    m_modelDrawItems.clear();
    m_modelBaseTransforms.clear();
    m_objectInstances.clear();
    ++m_geometryRevision;
}

void DeferredRenderer::ClearModels()
{
    m_modelCache.clear();
    m_models.clear();
    m_blasDirty = true;
    ++m_geometryRevision;
}

void DeferredRenderer::UnloadModel()
{
    m_asyncLoader.Reset();
    if (m_device != nullptr)
    {
        m_device->WaitIdle();
    }
    ClearModelInstances();
    ClearModels();
    m_modelName.clear();
    m_modelVertexCount = 0;
    m_modelTriangleCount = 0;
}

void DeferredRenderer::SetModelScale(float scale)
{
    MRE_ASSERT(scale > 0.0f, "Model scale must be greater than zero");
    m_modelScale = std::clamp(scale, 0.01f, 100.0f);
    ApplyModelScale();
    ++m_geometryRevision;
}

void DeferredRenderer::UpdateObjectData(std::uint32_t frameIndex)
{
    MRE_ASSERT(frameIndex < m_objectDataBuffers.size(), "Frame index out of range");
    MRE_ASSERT(m_objectDataBuffers[frameIndex] != nullptr, "Object data buffer is missing");

    const std::size_t itemCount = m_sceneDrawItems.size() + m_modelDrawItems.size();
    MRE_ASSERT(itemCount <= kMaxObjectData, "Too many draw items for the object data buffer");
    if (itemCount > kMaxObjectData)
    {
        return;
    }

    if (m_previousModels.size() != itemCount)
    {
        // Draw items were added or removed: assume no motion for one frame.
        m_previousModels.clear();
        m_previousModels.reserve(itemCount);
        for (const DrawItem& item : m_sceneDrawItems)
        {
            m_previousModels.push_back(item.pushConstants.model);
        }
        for (const DrawItem& item : m_modelDrawItems)
        {
            m_previousModels.push_back(item.pushConstants.model);
        }
    }

    std::vector<GpuObjectData> objectData{};
    objectData.resize(itemCount);

    std::size_t index = 0;
    const auto updateItem = [this, &objectData, &index](DrawItem& item) {
        objectData[index].previousModel = m_previousModels[index];
        objectData[index].specular = item.specular;
        objectData[index].specularTextureIndices = item.specularTextureIndices;
        objectData[index].specularSamplerIndices = item.specularSamplerIndices;
        objectData[index].volume = item.volume;
        objectData[index].attenuation = item.attenuation;
        objectData[index].volumeTextureIndices = item.volumeTextureIndices;
        objectData[index].volumeSamplerIndices = item.volumeSamplerIndices;
        objectData[index].alpha = item.alpha;
        objectData[index].sheen = item.sheen;
        objectData[index].sheenTextureIndices = item.sheenTextureIndices;
        objectData[index].sheenSamplerIndices = item.sheenSamplerIndices;
        objectData[index].emissive = item.emissive;
        objectData[index].emissiveTextureIndices = item.emissiveTextureIndices;
        objectData[index].emissiveSamplerIndices = item.emissiveSamplerIndices;
        item.pushConstants.drawIndex = static_cast<std::uint32_t>(index);
        m_previousModels[index] = item.pushConstants.model;
        ++index;
    };

    for (DrawItem& item : m_sceneDrawItems)
    {
        updateItem(item);
    }
    for (DrawItem& item : m_modelDrawItems)
    {
        updateItem(item);
    }

    if (!objectData.empty())
    {
        m_objectDataBuffers[frameIndex]->Write(objectData.data(), objectData.size() * sizeof(GpuObjectData));
    }
}

void DeferredRenderer::BuildAccelerationStructures(rhi::CommandList& commandList, std::uint32_t frameIndex)
{
    MRE_PROFILE_ZONE_N("BuildAccelerationStructures");

    if (!m_device->IsRayTracingSupported())
    {
        return;
    }

    if (m_blasDirty)
    {
        for (const std::unique_ptr<ModelResources>& model : m_models)
        {
            for (const std::unique_ptr<rhi::AccelerationStructure>& blas : model->accelerationStructures)
            {
                if (blas != nullptr)
                {
                    commandList.BuildAccelerationStructure(*blas);
                }
            }
        }
        if (m_planeBlas != nullptr)
        {
            commandList.BuildAccelerationStructure(*m_planeBlas);
        }
        commandList.AccelerationStructureBarrier();
        m_blasDirty = false;
    }

    MRE_ASSERT(frameIndex < m_tlas.size() && frameIndex < m_geometryDataBuffers.size(), "Frame index out of range");

    // Gather the triangle geometry of every draw item as a top level acceleration structure
    // instance. The instance custom index points into the per frame geometry data buffer.
    m_geometryData.clear();
    std::vector<rhi::AccelerationStructureInstance> instances{};
    instances.reserve(m_sceneDrawItems.size() + m_modelDrawItems.size());

    const auto appendItem = [this, &instances](const DrawItem& item) {
        if (item.blas == nullptr)
        {
            return;
        }

        rhi::AccelerationStructureInstance instance{};
        const glm::mat4& model = item.pushConstants.model;
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 4; ++column)
            {
                instance.transform[row * 4 + column] = model[column][row];
            }
        }
        instance.instanceCustomIndex = static_cast<std::uint32_t>(m_geometryData.size());
        instance.instanceMask = 0xFF;
        instance.bottomLevel = item.blas;
        instances.push_back(instance);

        GpuGeometryData geometry{};
        geometry.baseColorFactor = item.pushConstants.baseColorFactor;
        geometry.bufferIndices = glm::uvec4{
            item.vertexBufferIndex,
            item.indexBufferIndex,
            static_cast<std::uint32_t>(sizeof(GltfVertex)),
            0u};
        geometry.textureIndices = glm::uvec4{item.pushConstants.textureIndices, 0u, 0u};
        geometry.samplerIndices = glm::uvec4{item.pushConstants.samplerIndices, 0u, 0u};
        // params.z encodes the alpha mode: 0 opaque, 1 MASK, 2 BLEND.
        const float alphaMode = item.pushConstants.params.z > 0.5f
                                    ? 1.0f
                                    : (item.alpha.x > 0.5f ? 2.0f : 0.0f);
        geometry.params = glm::vec4{
            item.pushConstants.params.x,
            item.pushConstants.params.y,
            alphaMode,
            item.alpha.y};
        geometry.volume = item.volume;
        geometry.attenuation = item.attenuation;
        geometry.volumeTextureIndices = item.volumeTextureIndices;
        geometry.volumeSamplerIndices = item.volumeSamplerIndices;
        geometry.sheen = item.sheen;
        geometry.sheenTextureIndices = item.sheenTextureIndices;
        geometry.sheenSamplerIndices = item.sheenSamplerIndices;
        geometry.emissive = item.emissive;
        geometry.emissiveTextureIndices = item.emissiveTextureIndices;
        geometry.emissiveSamplerIndices = item.emissiveSamplerIndices;
        m_geometryData.push_back(geometry);
    };

    for (const DrawItem& item : m_sceneDrawItems)
    {
        appendItem(item);
    }
    for (const DrawItem& item : m_modelDrawItems)
    {
        appendItem(item);
    }

    if (instances.empty())
    {
        return;
    }

    if (m_tlasInstanceCounts.size() <= frameIndex)
    {
        m_tlasInstanceCounts.resize(m_device->GetFramesInFlight());
    }
    if (m_tlas[frameIndex] == nullptr || m_tlasInstanceCounts[frameIndex] != instances.size())
    {
        const rhi::TopLevelAccelerationStructureDesc tlasDesc{
            std::span<const rhi::AccelerationStructureInstance>{instances},
            true};
        m_tlas[frameIndex] = m_device->CreateTopLevelAccelerationStructure(tlasDesc);
        m_tlasInstanceCounts[frameIndex] = instances.size();
        if (m_tlas[frameIndex] == nullptr)
        {
            MRE_ERROR("Failed to create the top level acceleration structure");
            return;
        }
    }
    else
    {
        m_device->UpdateTopLevelAccelerationStructure(*m_tlas[frameIndex], instances);
    }

    commandList.BuildAccelerationStructure(*m_tlas[frameIndex]);
    commandList.AccelerationStructureBarrier();

    const std::uint64_t geometryBytes = m_geometryData.size() * sizeof(GpuGeometryData);
    if (m_geometryDataBuffers[frameIndex] == nullptr ||
        m_geometryDataBuffers[frameIndex]->GetSize() < geometryBytes)
    {
        m_geometryDataBuffers[frameIndex] =
            m_device->CreateBuffer(rhi::BufferDesc{geometryBytes, rhi::BufferUsage::Storage, true});
    }
    if (m_geometryDataBuffers[frameIndex] != nullptr)
    {
        m_geometryDataBuffers[frameIndex]->Write(m_geometryData.data(), geometryBytes);
    }
}

bool DeferredRenderer::CreateGiResources(rhi::Device& device)
{
    MRE_ASSERT(m_giTraceSetLayout != nullptr, "GI set layouts must exist before creating resources");

    m_gi.clear();
    m_gi.resize(device.GetFramesInFlight());

    // Per-frame buffer with the punctual lights evaluated by the ray traced hit shading. The cap
    // keeps the per-hit loop bounded; the buffer is always large enough for kMaxGiLights entries.
    m_giLightBuffers.clear();
    m_giLightBuffers.reserve(device.GetFramesInFlight());
    for (std::uint32_t frame = 0; frame < device.GetFramesInFlight(); ++frame)
    {
        m_giLightBuffers.push_back(device.CreateBuffer(
            rhi::BufferDesc{sizeof(GpuLight) * kMaxGiLights, rhi::BufferUsage::Storage, true}));
    }

    const std::uint32_t fullWidth = m_gbuffer.GetWidth();
    const std::uint32_t fullHeight = m_gbuffer.GetHeight();
    const auto halfWidth = std::max(
        1u,
        static_cast<std::uint32_t>(std::lround(static_cast<double>(fullWidth) * m_giSettings.resolutionScale)));
    const auto halfHeight = std::max(
        1u,
        static_cast<std::uint32_t>(std::lround(static_cast<double>(fullHeight) * m_giSettings.resolutionScale)));

    const auto createStorage = [&device](std::uint32_t width, std::uint32_t height, rhi::Format format) {
        return device.CreateTexture(rhi::TextureDesc{width, height, format, rhi::TextureUsage::Storage});
    };

    for (GiFrameResources& gi : m_gi)
    {
        gi.diffuseNoisy = createStorage(halfWidth, halfHeight, rhi::Format::R16G16B16A16Sfloat);
        gi.specularNoisy = createStorage(halfWidth, halfHeight, rhi::Format::R16G16B16A16Sfloat);
        gi.hitDistance = createStorage(halfWidth, halfHeight, rhi::Format::R16Sfloat);
        for (std::uint32_t parity = 0; parity < 2; ++parity)
        {
            gi.diffuseHistory[parity] = createStorage(halfWidth, halfHeight, rhi::Format::R16G16B16A16Sfloat);
            gi.specularHistory[parity] = createStorage(halfWidth, halfHeight, rhi::Format::R16G16B16A16Sfloat);
            gi.diffuseMoments[parity] = createStorage(halfWidth, halfHeight, rhi::Format::R16G16B16A16Sfloat);
            gi.specularMoments[parity] = createStorage(halfWidth, halfHeight, rhi::Format::R16G16B16A16Sfloat);
        }
        gi.diffuseFiltered = createStorage(halfWidth, halfHeight, rhi::Format::R16G16B16A16Sfloat);
        gi.specularFiltered = createStorage(halfWidth, halfHeight, rhi::Format::R16G16B16A16Sfloat);
        gi.indirect = createStorage(fullWidth, fullHeight, rhi::Format::R16G16B16A16Sfloat);
        gi.historyParity = 0;
        gi.historyValid = false;

        if (gi.diffuseNoisy == nullptr || gi.specularNoisy == nullptr || gi.hitDistance == nullptr ||
            gi.diffuseHistory[0] == nullptr || gi.diffuseHistory[1] == nullptr ||
            gi.specularHistory[0] == nullptr || gi.specularHistory[1] == nullptr ||
            gi.diffuseMoments[0] == nullptr || gi.diffuseMoments[1] == nullptr ||
            gi.specularMoments[0] == nullptr || gi.specularMoments[1] == nullptr ||
            gi.diffuseFiltered == nullptr || gi.specularFiltered == nullptr || gi.indirect == nullptr)
        {
            MRE_CRITICAL("Failed to create the global illumination textures");
            return false;
        }

        gi.uniformBuffer = device.CreateBuffer(
            rhi::BufferDesc{sizeof(GiUniform), rhi::BufferUsage::Uniform, true});
        gi.traceSet = device.CreateDescriptorSet(*m_giTraceSetLayout);
        gi.temporalSet = device.CreateDescriptorSet(*m_giTemporalSetLayout);
        gi.upsampleSet = device.CreateDescriptorSet(*m_giUpsampleSetLayout);
        for (std::unique_ptr<rhi::DescriptorSet>& atrousSet : gi.atrousSets)
        {
            atrousSet = device.CreateDescriptorSet(*m_giAtrousSetLayout);
        }

        const bool atrousSetsCreated = std::all_of(
            gi.atrousSets.begin(),
            gi.atrousSets.end(),
            [](const std::unique_ptr<rhi::DescriptorSet>& atrousSet) { return atrousSet != nullptr; });

        if (gi.uniformBuffer == nullptr || gi.traceSet == nullptr || gi.temporalSet == nullptr ||
            !atrousSetsCreated || gi.upsampleSet == nullptr)
        {
            MRE_CRITICAL("Failed to create the global illumination descriptor sets");
            return false;
        }
    }

    if (!CreateReflectionResources(device))
    {
        return false;
    }

    m_lightingSetsDirty = true;
    MRE_INFO("Global illumination resources created ({}x{} half resolution)", halfWidth, halfHeight);
    return true;
}

bool DeferredRenderer::CreateReflectionResources(rhi::Device& device)
{
    if (!device.IsRayTracingSupported() || m_reflectionTraceSetLayout == nullptr ||
        m_reflectionTemporalSetLayout == nullptr || m_reflectionUpsampleSetLayout == nullptr ||
        m_gi.empty())
    {
        return true;
    }

    const std::uint32_t fullWidth = m_gbuffer.GetWidth();
    const std::uint32_t fullHeight = m_gbuffer.GetHeight();
    const float resolutionScale = std::clamp(m_reflectionSettings.resolutionScale, 0.25f, 1.0f);
    const std::uint32_t halfWidth =
        std::max(1u, static_cast<std::uint32_t>(std::lround(static_cast<double>(fullWidth) * resolutionScale)));
    const std::uint32_t halfHeight =
        std::max(1u, static_cast<std::uint32_t>(std::lround(static_cast<double>(fullHeight) * resolutionScale)));

    const auto createStorage = [&device](std::uint32_t width, std::uint32_t height, rhi::Format format) {
        return device.CreateTexture(rhi::TextureDesc{width, height, format, rhi::TextureUsage::Storage});
    };

    for (GiFrameResources& gi : m_gi)
    {
        gi.reflectionNoisy = createStorage(halfWidth, halfHeight, rhi::Format::R16G16B16A16Sfloat);
        gi.reflectionHitDistance = createStorage(halfWidth, halfHeight, rhi::Format::R16Sfloat);
        for (std::uint32_t parity = 0; parity < 2; ++parity)
        {
            gi.reflectionHistory[parity] =
                createStorage(halfWidth, halfHeight, rhi::Format::R16G16B16A16Sfloat);
            gi.reflectionMoments[parity] =
                createStorage(halfWidth, halfHeight, rhi::Format::R16G16B16A16Sfloat);
        }
        gi.reflectionFiltered = createStorage(halfWidth, halfHeight, rhi::Format::R16G16B16A16Sfloat);
        gi.reflections = createStorage(fullWidth, fullHeight, rhi::Format::R16G16B16A16Sfloat);

        if (gi.reflectionNoisy == nullptr || gi.reflectionHitDistance == nullptr ||
            gi.reflectionHistory[0] == nullptr || gi.reflectionHistory[1] == nullptr ||
            gi.reflectionMoments[0] == nullptr || gi.reflectionMoments[1] == nullptr ||
            gi.reflectionFiltered == nullptr || gi.reflections == nullptr)
        {
            MRE_CRITICAL("Failed to create the ray traced reflection textures");
            return false;
        }

        gi.reflectionTraceSet = device.CreateDescriptorSet(*m_reflectionTraceSetLayout);
        gi.reflectionTemporalSet = device.CreateDescriptorSet(*m_reflectionTemporalSetLayout);
        gi.reflectionUpsampleSet = device.CreateDescriptorSet(*m_reflectionUpsampleSetLayout);
        for (std::unique_ptr<rhi::DescriptorSet>& atrousSet : gi.reflectionAtrousSets)
        {
            atrousSet = device.CreateDescriptorSet(*m_giAtrousSetLayout);
        }

        gi.reflectionParity = 0;
        gi.reflectionValid = false;
    }

    MRE_INFO("Ray traced reflection resources created ({}x{})", halfWidth, halfHeight);
    return true;
}

void DeferredRenderer::SetGiSettings(const GiSettings& settings)
{
    const bool scaleChanged =
        std::abs(settings.resolutionScale - m_giSettings.resolutionScale) > 1e-4f;
    m_giSettings = settings;

    if (m_initialized && scaleChanged && m_device != nullptr)
    {
        m_device->WaitIdle();
        if (!CreateGiResources(*m_device))
        {
            MRE_CRITICAL("Failed to recreate the global illumination resources");
        }
    }
}

bool DeferredRenderer::IsGiActive() const
{
    return m_giSettings.enabled && m_device != nullptr && m_device->IsRayTracingSupported();
}

void DeferredRenderer::SetReflectionSettings(const ReflectionSettings& settings)
{
    const bool scaleChanged =
        std::abs(settings.resolutionScale - m_reflectionSettings.resolutionScale) > 1e-4f;
    m_reflectionSettings = settings;

    if (scaleChanged && m_initialized && m_device != nullptr &&
        m_device->IsRayTracingSupported())
    {
        m_device->WaitIdle();
        if (!CreateReflectionResources(*m_device))
        {
            MRE_CRITICAL("Failed to recreate the ray traced reflection resources");
        }
    }

    // Reset the temporal history so the new cutoff/intensity converges immediately.
    for (GiFrameResources& gi : m_gi)
    {
        gi.reflectionValid = false;
    }
}

void DeferredRenderer::SetEnvironmentSettings(const EnvironmentSettings& settings)
{
    const bool mapChanged = settings.environmentMap != m_loadedEnvironmentMapPath;
    m_environmentSettings = settings;
    if (mapChanged)
    {
        ReloadEnvironmentMap();
    }

    // The map/colour SH is stored unscaled; fold the intensity in per update so the slider does not
    // re-run the projection over the whole image.
    EnvironmentSh base = m_environmentMapBound ? m_environmentMapShBase : EnvironmentSh{};
    if (!m_environmentMapBound)
    {
        EnvironmentSettings gradient = m_environmentSettings;
        gradient.intensity = 1.0f;
        base = ComputeEnvironmentSh(gradient);
    }

    const float intensity = std::max(m_environmentSettings.intensity, 0.0f);
    for (std::size_t band = 0; band < 9; ++band)
    {
        m_environmentSh.bands[band] = base.bands[band] * intensity;
    }
}

void DeferredRenderer::ReloadEnvironmentMap()
{
    if (m_device != nullptr)
    {
        m_device->WaitIdle();
    }

    m_environmentMap.reset();
    m_environmentMapBound = false;
    m_environmentMapShBase = EnvironmentSh{};
    m_loadedEnvironmentMapPath = m_environmentSettings.environmentMap;

    const std::filesystem::path& path = m_environmentSettings.environmentMap;
    if (path.empty())
    {
        return;
    }

    HdrImage image{};
    if (!LoadHdrImage(path, image))
    {
        MRE_WARN("Falling back to the gradient sky; '{}' could not be loaded", PathToUtf8(path));
        return;
    }

    const std::vector<std::uint16_t> packed = PackHdrToRgba16(image);
    const rhi::TextureDesc textureDesc{
        image.width,
        image.height,
        rhi::Format::R16G16B16A16Sfloat,
        rhi::TextureUsage::Sampled,
        packed.data(),
        packed.size() * sizeof(std::uint16_t)};
    m_environmentMap = m_device->CreateTexture(textureDesc);
    if (m_environmentMap == nullptr)
    {
        MRE_WARN("Falling back to the gradient sky; the environment texture could not be created");
        return;
    }

    m_environmentMapIndex = m_environmentMap->GetBindlessIndex();
    m_bindlessSet->BindTexture(0, *m_environmentMap, m_environmentMapIndex);
    m_environmentMapSamplerIndex =
        GetSamplerIndex(rhi::AddressMode::ClampToEdge, rhi::AddressMode::ClampToEdge);
    m_environmentMapShBase =
        ComputeEnvironmentSh(image.pixels.data(), image.width, image.height, 1.0f);
    m_environmentMapBound = true;
}

bool DeferredRenderer::IsReflectionActive() const
{
    return m_reflectionSettings.enabled && m_device != nullptr &&
           m_device->IsRayTracingSupported() && m_rtReflectionPipeline != nullptr &&
           m_reflectionTemporalPipeline != nullptr && m_reflectionUpsamplePipeline != nullptr &&
           !m_gi.empty();
}

bool DeferredRenderer::IsCloudActive() const
{
    return m_cloudSettings.enabled && m_device != nullptr && m_cloudPipeline != nullptr &&
           m_cloudUpsamplePipeline != nullptr && !m_cloudSets.empty();
}

void DeferredRenderer::SetCloudSettings(const CloudSettings& settings)
{
    const bool scaleChanged =
        std::abs(settings.resolutionScale - m_cloudSettings.resolutionScale) > 1e-4f;
    const bool enabledChanged = settings.enabled != m_cloudSettings.enabled;
    m_cloudSettings = settings;
    m_cloudSettings.resolutionScale = std::clamp(m_cloudSettings.resolutionScale, 0.25f, 1.0f);

    if (scaleChanged && m_initialized && m_device != nullptr)
    {
        m_device->WaitIdle();
        if (!CreateCloudResources(*m_device))
        {
            MRE_CRITICAL("Failed to recreate the cloud resources");
        }
    }
    if (enabledChanged)
    {
        m_lightingSetsDirty = true;
    }
    // Setting changes invalidate the cloud history so the previous weather/layers do not ghost.
    for (std::size_t index = 0; index < m_cloudHistoryValid.size(); ++index)
    {
        m_cloudHistoryValid[index] = false;
    }
}

void DeferredRenderer::GenerateCloudNoiseTextures()
{
    const auto start = std::chrono::steady_clock::now();

    constexpr std::uint32_t kBaseResolution = 128;
    constexpr std::uint32_t kDetailResolution = 32;
    constexpr std::uint32_t kCurlResolution = 128;
    constexpr std::uint32_t kWeatherResolution = 256;
    constexpr std::uint32_t kNoiseSeed = 1337u;
    constexpr float kWeatherVariation = 0.5f;

    std::vector<std::uint8_t> basePixels(
        static_cast<std::size_t>(kBaseResolution) * kBaseResolution * kBaseResolution * 4);
    std::vector<std::uint8_t> detailPixels(
        static_cast<std::size_t>(kDetailResolution) * kDetailResolution * kDetailResolution * 4);
    std::vector<std::uint8_t> curlPixels(
        static_cast<std::size_t>(kCurlResolution) * kCurlResolution * 4);
    std::vector<std::uint8_t> weatherPixels(
        static_cast<std::size_t>(kWeatherResolution) * kWeatherResolution * 4);

    const auto dispatchSlices = [this](
                                    std::uint32_t slices,
                                    const auto& generate) {
        const std::uint32_t jobCount =
            m_jobSystem != nullptr ? std::max(1u, m_jobSystem->GetThreadCount()) : 1u;
        for (std::uint32_t job = 0; job < jobCount; ++job)
        {
            const std::uint32_t begin = slices * job / jobCount;
            const std::uint32_t end = slices * (job + 1) / jobCount;
            if (begin >= end)
            {
                continue;
            }
            if (m_jobSystem != nullptr)
            {
                m_jobSystem->Submit([&generate, begin, end]() { generate(begin, end); });
            }
            else
            {
                generate(begin, end);
            }
        }
        if (m_jobSystem != nullptr)
        {
            m_jobSystem->WaitIdle();
        }
    };

    dispatchSlices(kBaseResolution, [&](std::uint32_t begin, std::uint32_t end) {
        GenerateCloudBaseShape(kBaseResolution, kNoiseSeed, begin, end, basePixels);
    });
    dispatchSlices(kDetailResolution, [&](std::uint32_t begin, std::uint32_t end) {
        GenerateCloudDetail(kDetailResolution, kNoiseSeed + 17u, begin, end, detailPixels);
    });
    dispatchSlices(kCurlResolution, [&](std::uint32_t begin, std::uint32_t end) {
        GenerateCloudCurl(kCurlResolution, kNoiseSeed + 29u, begin, end, curlPixels);
    });
    dispatchSlices(kWeatherResolution, [&](std::uint32_t begin, std::uint32_t end) {
        GenerateCloudWeather(
            kWeatherResolution,
            kWeatherVariation,
            kNoiseSeed + 41u,
            begin,
            end,
            weatherPixels);
    });

    const auto createNoise = [this](
                                 std::uint32_t width,
                                 std::uint32_t height,
                                 std::uint32_t depth,
                                 const std::vector<std::uint8_t>& pixels) {
        const rhi::TextureDesc desc{
            width,
            height,
            rhi::Format::R8G8B8A8Unorm,
            rhi::TextureUsage::Sampled,
            pixels.data(),
            pixels.size(),
            1,
            depth};
        return m_device->CreateTexture(desc);
    };

    m_cloudBaseNoise = createNoise(kBaseResolution, kBaseResolution, kBaseResolution, basePixels);
    m_cloudDetailNoise = createNoise(kDetailResolution, kDetailResolution, kDetailResolution, detailPixels);
    m_cloudCurlNoise = createNoise(kCurlResolution, kCurlResolution, 1, curlPixels);
    m_cloudWeatherMap = createNoise(kWeatherResolution, kWeatherResolution, 1, weatherPixels);

    m_cloudNoiseReady = m_cloudBaseNoise != nullptr && m_cloudDetailNoise != nullptr &&
                        m_cloudCurlNoise != nullptr && m_cloudWeatherMap != nullptr;

    const float milliseconds =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
    MRE_INFO("Generated cloud noise textures in {:.0f} ms", static_cast<double>(milliseconds));
}

bool DeferredRenderer::CreateCloudResources(rhi::Device& device)
{
    MRE_ASSERT(m_cloudSetLayout != nullptr, "Cloud set layouts must exist before the resources");

    if (!m_cloudNoiseReady)
    {
        GenerateCloudNoiseTextures();
        if (!m_cloudNoiseReady)
        {
            MRE_CRITICAL("Failed to generate the cloud noise textures");
            return false;
        }
    }

    if (m_cloudSampler == nullptr)
    {
        rhi::SamplerDesc samplerDesc{};
        samplerDesc.addressModeU = rhi::AddressMode::Repeat;
        samplerDesc.addressModeV = rhi::AddressMode::Repeat;
        m_cloudSampler = device.CreateSampler(samplerDesc);
        if (m_cloudSampler == nullptr)
        {
            MRE_CRITICAL("Failed to create the cloud sampler");
            return false;
        }
    }

    const auto createEmpty = [&device]() {
        const std::array<std::uint16_t, 4> emptyPixel{0u, 0u, 0u, 0x3C00u}; // 0,0,0,1 in half
        const rhi::TextureDesc desc{
            1,
            1,
            rhi::Format::R16G16B16A16Sfloat,
            rhi::TextureUsage::Sampled,
            emptyPixel.data(),
            emptyPixel.size() * sizeof(std::uint16_t)};
        return device.CreateTexture(desc);
    };
    if (m_cloudEmpty == nullptr)
    {
        m_cloudEmpty = createEmpty();
        if (m_cloudEmpty == nullptr)
        {
            MRE_CRITICAL("Failed to create the empty cloud texture");
            return false;
        }
    }

    const std::uint32_t framesInFlight = device.GetFramesInFlight();
    const std::uint32_t fullWidth = m_gbuffer.GetWidth();
    const std::uint32_t fullHeight = m_gbuffer.GetHeight();
    m_cloudWidth = std::max(
        1u,
        static_cast<std::uint32_t>(
            std::lround(static_cast<double>(fullWidth) * m_cloudSettings.resolutionScale)));
    m_cloudHeight = std::max(
        1u,
        static_cast<std::uint32_t>(
            std::lround(static_cast<double>(fullHeight) * m_cloudSettings.resolutionScale)));

    const auto createStorage = [&device](std::uint32_t width, std::uint32_t height, rhi::Format format) {
        return device.CreateTexture(rhi::TextureDesc{width, height, format, rhi::TextureUsage::Storage});
    };

    m_cloudUniformBuffers.clear();
    m_cloudSets.clear();
    m_cloudUpsampleSets.clear();
    m_cloudReprojectSets.clear();
    m_cloudGodRaySets.clear();
    m_cloudCurrent.clear();
    m_cloudDistance.clear();
    m_cloudMask.clear();
    m_cloudFull.clear();
    m_cloudHistory.clear();
    m_cloudHistoryDistance.clear();
    m_cloudHistoryMask.clear();
    m_cloudGodRays.clear();
    m_cloudHistoryParity.clear();
    m_cloudHistoryValid.clear();
    m_cloudUniformBuffers.reserve(framesInFlight);
    m_cloudSets.reserve(framesInFlight);
    m_cloudUpsampleSets.reserve(framesInFlight);
    m_cloudReprojectSets.reserve(framesInFlight);
    m_cloudGodRaySets.reserve(framesInFlight);
    m_cloudCurrent.reserve(framesInFlight);
    m_cloudDistance.reserve(framesInFlight);
    m_cloudMask.reserve(framesInFlight);
    m_cloudFull.reserve(framesInFlight);

    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        m_cloudUniformBuffers.push_back(
            device.CreateBuffer(rhi::BufferDesc{sizeof(CloudUniform), rhi::BufferUsage::Uniform, true}));
        m_cloudSets.push_back(device.CreateDescriptorSet(*m_cloudSetLayout));
        m_cloudUpsampleSets.push_back(device.CreateDescriptorSet(*m_cloudUpsampleSetLayout));
        m_cloudReprojectSets.push_back(device.CreateDescriptorSet(*m_cloudReprojectSetLayout));
        m_cloudGodRaySets.push_back(device.CreateDescriptorSet(*m_cloudGodRaySetLayout));
        m_cloudCurrent.push_back(createStorage(m_cloudWidth, m_cloudHeight, rhi::Format::R16G16B16A16Sfloat));
        m_cloudDistance.push_back(createStorage(m_cloudWidth, m_cloudHeight, rhi::Format::R16Sfloat));
        m_cloudMask.push_back(createStorage(m_cloudWidth, m_cloudHeight, rhi::Format::R16Sfloat));
        m_cloudFull.push_back(createStorage(fullWidth, fullHeight, rhi::Format::R16G16B16A16Sfloat));
        m_cloudGodRays.push_back(createStorage(m_cloudWidth, m_cloudHeight, rhi::Format::R16G16B16A16Sfloat));

        std::array<std::unique_ptr<rhi::Texture>, 2> history{};
        std::array<std::unique_ptr<rhi::Texture>, 2> historyDistance{};
        std::array<std::unique_ptr<rhi::Texture>, 2> historyMask{};
        for (std::uint32_t parity = 0; parity < 2; ++parity)
        {
            history[parity] = createStorage(m_cloudWidth, m_cloudHeight, rhi::Format::R16G16B16A16Sfloat);
            historyDistance[parity] = createStorage(m_cloudWidth, m_cloudHeight, rhi::Format::R16Sfloat);
            historyMask[parity] = createStorage(m_cloudWidth, m_cloudHeight, rhi::Format::R16Sfloat);
        }
        m_cloudHistory.push_back(std::move(history));
        m_cloudHistoryDistance.push_back(std::move(historyDistance));
        m_cloudHistoryMask.push_back(std::move(historyMask));
        m_cloudHistoryParity.push_back(0u);
        m_cloudHistoryValid.push_back(false);

        const bool frameReady =
            m_cloudUniformBuffers.back() != nullptr && m_cloudSets.back() != nullptr &&
            m_cloudUpsampleSets.back() != nullptr && m_cloudReprojectSets.back() != nullptr &&
            m_cloudGodRaySets.back() != nullptr && m_cloudCurrent.back() != nullptr &&
            m_cloudDistance.back() != nullptr && m_cloudMask.back() != nullptr &&
            m_cloudFull.back() != nullptr && m_cloudGodRays.back() != nullptr &&
            m_cloudHistory.back()[0] != nullptr && m_cloudHistory.back()[1] != nullptr &&
            m_cloudHistoryDistance.back()[0] != nullptr &&
            m_cloudHistoryDistance.back()[1] != nullptr &&
            m_cloudHistoryMask.back()[0] != nullptr && m_cloudHistoryMask.back()[1] != nullptr;
        if (!frameReady)
        {
            MRE_CRITICAL("Failed to create the cloud resources");
            return false;
        }
    }

    m_lightingSetsDirty = true;
    MRE_INFO("Cloud resources created ({}x{} trace, {}x{} full)", m_cloudWidth, m_cloudHeight, fullWidth, fullHeight);
    return true;
}

void DeferredRenderer::UpdateCloudUniform(std::uint32_t frameIndex, const FrameView& view)
{
    if (frameIndex >= m_cloudUniformBuffers.size() || m_cloudUniformBuffers[frameIndex] == nullptr)
    {
        return;
    }

    m_cloudUniform.inverseViewProjection = glm::inverse(view.viewProjection);
    m_cloudUniform.previousViewProjection = view.previousViewProjection;
    m_cloudUniform.viewProjection = view.viewProjection;
    m_cloudUniform.cameraPosition = glm::vec4{view.position, 1.0f};
    m_cloudUniform.screenSize = glm::vec4{
        static_cast<float>(m_gbuffer.GetWidth()),
        static_cast<float>(m_gbuffer.GetHeight()),
        static_cast<float>(m_cloudWidth),
        static_cast<float>(m_cloudHeight)};
    m_cloudUniform.sunDirectionIntensity =
        glm::vec4{glm::vec3{m_sceneUniform.sunDirection}, 1.0f};
    m_cloudUniform.sunColor = m_sceneUniform.sunColor;
    m_cloudUniform.cloudParams = glm::vec4{
        m_cloudSettings.coverage,
        m_cloudSettings.cloudType,
        m_cloudSettings.density,
        m_cloudSettings.detailStrength};
    m_cloudUniform.cloudParams2 = glm::vec4{
        m_cloudSettings.layerBottomKm,
        m_cloudSettings.layerTopKm,
        m_cloudSettings.altitudeVariationKm,
        m_cloudSettings.curlStrength};
    m_cloudUniform.cloudParams3 = glm::vec4{
        m_cloudSettings.windDirectionRadians,
        m_cloudSettings.windSpeed,
        static_cast<float>(m_cloudFrameCounter) / 60.0f,
        0.0f};
    for (std::size_t band = 0; band < 9; ++band)
    {
        m_cloudUniform.skySh[band] = m_environmentSh.bands[band];
    }
    m_cloudUniform.skyParams = glm::vec4{
        1.0f,
        m_cloudSettings.coverageVariation,
        m_cloudSettings.godRayIntensity,
        0.0f};
    // Noise scales in 1 / metres: base features are kilometres wide, detail a few hundred metres.
    m_cloudUniform.noiseScales = glm::vec4{
        1.0f / 8000.0f,
        1.0f / 900.0f,
        1.0f / 3000.0f,
        1.0f / 60000.0f};
    const bool historyValid =
        frameIndex < m_cloudHistoryValid.size() && m_cloudHistoryValid[frameIndex];
    m_cloudUniform.temporalParams = glm::vec4{
        std::clamp(m_cloudSettings.temporalBlend, 0.0f, 0.98f),
        historyValid ? 0.0f : 1.0f,
        0.0f,
        0.0f};
    m_cloudUniform.cloudParams4 = glm::vec4{
        static_cast<float>(std::clamp(m_cloudSettings.primarySteps, 1u, 128u)),
        static_cast<float>(std::clamp(m_cloudSettings.lightSteps, 1u, 8u)),
        static_cast<float>(m_cloudFrameCounter),
        0.0f};

    m_cloudUniformBuffers[frameIndex]->Write(&m_cloudUniform, sizeof(CloudUniform));
    ++m_cloudFrameCounter;
}

void DeferredRenderer::RecordCloudPasses(rhi::CommandList& commandList, std::uint32_t frameIndex)
{
    if (!IsCloudActive() || frameIndex >= m_cloudSets.size() ||
        frameIndex >= m_cloudReprojectSets.size() || frameIndex >= m_cloudGodRaySets.size() ||
        frameIndex >= m_cloudHistory.size() || frameIndex >= m_cloudHistoryParity.size() ||
        m_cloudCurrent[frameIndex] == nullptr || m_cloudUpsampleSets[frameIndex] == nullptr ||
        m_cloudGodRays[frameIndex] == nullptr)
    {
        return;
    }

    rhi::DescriptorSet& set = *m_cloudSets[frameIndex];
    set.BindBuffer(0, *m_cloudUniformBuffers[frameIndex]);
    set.BindTexture(1, m_gbuffer.GetDepth(), 0u);
    set.BindTexture(2, *m_cloudBaseNoise, 0u);
    set.BindTexture(3, *m_cloudDetailNoise, 0u);
    set.BindTexture(4, *m_cloudCurlNoise, 0u);
    set.BindTexture(5, *m_cloudWeatherMap, 0u);
    set.BindStorageImage(6, *m_cloudCurrent[frameIndex]);
    set.BindStorageImage(7, *m_cloudMask[frameIndex]);
    set.BindStorageImage(8, *m_cloudDistance[frameIndex]);
    set.BindSampler(9, *m_cloudSampler);

    commandList.BeginGpuZone("CloudMarch");
    commandList.BindPipeline(*m_cloudPipeline);
    commandList.BindDescriptorSet(0, set);
    commandList.Dispatch((m_cloudWidth + 7) / 8, (m_cloudHeight + 7) / 8, 1);
    commandList.ShaderMemoryBarrier(rhi::ShaderStageMask::Compute, rhi::ShaderStageMask::Compute);

    const std::uint32_t parity = m_cloudHistoryParity[frameIndex];
    const std::uint32_t nextParity = parity ^ 1u;

    // Temporal reprojection: read the history for this slot, write the other parity.
    rhi::DescriptorSet& reproject = *m_cloudReprojectSets[frameIndex];
    reproject.BindBuffer(0, *m_cloudUniformBuffers[frameIndex]);
    reproject.BindTexture(1, *m_cloudCurrent[frameIndex], 0u);
    reproject.BindTexture(2, *m_cloudDistance[frameIndex], 0u);
    reproject.BindTexture(3, *m_cloudMask[frameIndex], 0u);
    reproject.BindTexture(4, *m_cloudHistory[frameIndex][parity], 0u);
    reproject.BindTexture(5, *m_cloudHistoryDistance[frameIndex][parity], 0u);
    reproject.BindTexture(6, *m_cloudHistoryMask[frameIndex][parity], 0u);
    reproject.BindStorageImage(7, *m_cloudHistory[frameIndex][nextParity]);
    reproject.BindStorageImage(8, *m_cloudHistoryDistance[frameIndex][nextParity]);
    reproject.BindStorageImage(9, *m_cloudHistoryMask[frameIndex][nextParity]);

    commandList.BindPipeline(*m_cloudReprojectPipeline);
    commandList.BindDescriptorSet(0, reproject);
    commandList.Dispatch((m_cloudWidth + 7) / 8, (m_cloudHeight + 7) / 8, 1);
    commandList.ShaderMemoryBarrier(rhi::ShaderStageMask::Compute, rhi::ShaderStageMask::Compute);

    // God rays: radial blur of the (temporally resolved) sun occlusion mask.
    rhi::DescriptorSet& godRay = *m_cloudGodRaySets[frameIndex];
    godRay.BindBuffer(0, *m_cloudUniformBuffers[frameIndex]);
    godRay.BindTexture(1, *m_cloudMask[frameIndex], 0u);
    godRay.BindTexture(2, *m_cloudHistoryMask[frameIndex][nextParity], 0u);
    godRay.BindStorageImage(3, *m_cloudGodRays[frameIndex]);

    commandList.BindPipeline(*m_cloudGodRayPipeline);
    commandList.BindDescriptorSet(0, godRay);
    commandList.Dispatch((m_cloudWidth + 7) / 8, (m_cloudHeight + 7) / 8, 1);
    commandList.ShaderMemoryBarrier(rhi::ShaderStageMask::Compute, rhi::ShaderStageMask::Compute);

    rhi::DescriptorSet& upsample = *m_cloudUpsampleSets[frameIndex];
    upsample.BindBuffer(0, *m_cloudUniformBuffers[frameIndex]);
    upsample.BindTexture(1, *m_cloudHistory[frameIndex][nextParity], 0u);
    upsample.BindTexture(2, *m_cloudHistoryDistance[frameIndex][nextParity], 0u);
    upsample.BindTexture(3, m_gbuffer.GetDepth(), 0u);
    upsample.BindStorageImage(4, *m_cloudFull[frameIndex]);

    commandList.BindPipeline(*m_cloudUpsamplePipeline);
    commandList.BindDescriptorSet(0, upsample);
    commandList.Dispatch(
        (m_gbuffer.GetWidth() + 7) / 8,
        (m_gbuffer.GetHeight() + 7) / 8,
        1);
    commandList.ShaderMemoryBarrier(rhi::ShaderStageMask::Compute, rhi::ShaderStageMask::Fragment);
    commandList.EndGpuZone();

    m_cloudHistoryParity[frameIndex] = nextParity;
    m_cloudHistoryValid[frameIndex] = true;
}

void DeferredRenderer::UpdateGiUniform(std::uint32_t frameIndex, const FrameView& view)
{
    if (m_gi.size() <= frameIndex)
    {
        return;
    }

    const Light* sun = FindShadowLight();
    if (sun == nullptr)
    {
        for (const Light& light : m_lights)
        {
            if (light.type == LightType::Directional)
            {
                sun = &light;
                break;
            }
        }
    }

    m_giUniform = GiUniform{};
    m_giUniform.inverseViewProjection = glm::inverse(view.viewProjection);
    m_giUniform.previousViewProjection = view.previousViewProjection;
    m_giUniform.viewProjection = view.viewProjection;
    m_giUniform.screenSize = glm::vec4{
        static_cast<float>(m_gbuffer.GetWidth()),
        static_cast<float>(m_gbuffer.GetHeight()),
        0.0f,
        0.0f};
    m_giUniform.halfSize = glm::vec4{
        static_cast<float>(m_gi[frameIndex].diffuseNoisy->GetWidth()),
        static_cast<float>(m_gi[frameIndex].diffuseNoisy->GetHeight()),
        0.0f,
        0.0f};
    m_giUniform.cameraPosition = glm::vec4{view.position, 1.0f};
    m_giUniform.skyColor = glm::vec4{glm::vec3{m_sceneUniform.ambientColor}, 1.0f};
    if (sun != nullptr)
    {
        const float length = glm::length(sun->direction);
        const glm::vec3 direction = length > 1e-4f ? -sun->direction / length : glm::vec3{0.0f, -1.0f, 0.0f};
        m_giUniform.sunDirectionIntensity = glm::vec4{direction, sun->intensity};
        m_giUniform.sunColor = glm::vec4{sun->color, 1.0f};
    }
    m_giUniform.params = glm::vec4{
        m_giSettings.maxDistance,
        static_cast<float>(m_giFrameCounter),
        m_giSettings.specularMinRoughness,
        m_giSettings.feedbackIntensity};
    m_giUniform.params2 = glm::vec4{
        m_giSettings.temporalAlphaMin,
        m_giSettings.phiDepth,
        m_giSettings.phiNormal,
        m_giSettings.phiLuminance};
    m_giUniform.params3 = glm::vec4{
        m_giSettings.phiHitDistance,
        m_giSettings.intensity,
        (m_giSettings.sunShadowRay && sun != nullptr && sun->castShadows) ? 1.0f : 0.0f,
        m_gi[frameIndex].historyValid ? 0.0f : 1.0f};
    m_giUniform.params4 = glm::vec4{
        static_cast<float>(m_giSettings.sunShadowSamples),
        m_giSettings.sunAngularRadiusDegrees * 3.14159265358979323846f / 180.0f,
        static_cast<float>(m_giSettings.traceSamples),
        IsRtShadowActive() ? 1.0f : 0.0f};
    m_giUniform.params5 = glm::vec4{
        static_cast<float>(std::clamp(m_giSettings.probeRays, 1u, 256u)),
        std::clamp(m_giSettings.probeHysteresis, 0.0f, 0.98f),
        IsProbeGiActive() ? static_cast<float>(m_giProbeData.size()) : 0.0f,
        m_gi[frameIndex].reflectionValid ? 0.0f : 1.0f};
    m_giUniform.params6 = glm::vec4{
        std::max(m_reflectionSettings.maxDistance, 0.1f),
        std::clamp(m_reflectionSettings.maxRoughness, 0.0f, 1.0f),
        static_cast<float>(std::clamp(m_reflectionSettings.samples, 1u, 8u)),
        // Zero disables the reflection signal, which makes the GI trace cover smooth specular.
        IsReflectionActive() ? std::max(m_reflectionSettings.intensity, 0.0f) : 0.0f};
    m_giUniform.params7 = glm::vec4{
        static_cast<float>(std::clamp(m_reflectionSettings.refractionBounces, 1u, 8u)),
        m_reflectionSettings.dispersion ? 1.0f : 0.0f,
        std::max(m_reflectionSettings.volumeAbsorptionScale, 0.0f),
        0.0f};
    m_giUniform.params8 = glm::vec4{
        static_cast<float>(m_giLights.size()),
        m_giSettings.pointLightShadows ? 1.0f : 0.0f,
        std::max(m_giSettings.pointLightShadowBias, 0.0f),
        0.0f};
    for (std::size_t band = 0; band < 9; ++band)
    {
        m_giUniform.skySh[band] = m_environmentSh.bands[band];
    }
    m_giUniform.skyParams = glm::vec4{
        m_environmentSettings.enabled ? 1.0f : 0.0f,
        m_environmentSettings.intensity,
        m_environmentSettings.sunDiscIntensity,
        0.0f};
    m_giUniform.skyZenith = glm::vec4{m_environmentSettings.zenithColor, 0.0f};
    m_giUniform.skyHorizon = glm::vec4{m_environmentSettings.horizonColor, 0.0f};
    m_giUniform.skyGround = glm::vec4{m_environmentSettings.groundColor, 0.0f};
    m_giUniform.skyMap = glm::vec4{
        static_cast<float>(m_environmentMapIndex),
        static_cast<float>(m_environmentMapSamplerIndex),
        m_environmentMapBound ? 1.0f : 0.0f,
        0.0f};
    if (!m_giLights.empty() && frameIndex < m_giLightBuffers.size() &&
        m_giLightBuffers[frameIndex] != nullptr)
    {
        m_giLightBuffers[frameIndex]->Write(m_giLights.data(), m_giLights.size() * sizeof(GpuLight));
    }
    if (m_gi[frameIndex].reflectionNoisy != nullptr)
    {
        m_giUniform.reflectionSize = glm::vec4{
            static_cast<float>(m_gi[frameIndex].reflectionNoisy->GetWidth()),
            static_cast<float>(m_gi[frameIndex].reflectionNoisy->GetHeight()),
            0.0f,
            0.0f};
    }

    m_gi[frameIndex].uniformBuffer->Write(&m_giUniform, sizeof(GiUniform));

    // Probes never use the screen space feedback term.
    if (frameIndex < m_giProbeUniformBuffers.size() && m_giProbeUniformBuffers[frameIndex] != nullptr)
    {
        m_giProbeUniform = m_giUniform;
        m_giProbeUniform.params.w = 0.0f;
        m_giProbeUniformBuffers[frameIndex]->Write(&m_giProbeUniform, sizeof(GiUniform));
    }

    ++m_giFrameCounter;
}

void DeferredRenderer::BindGiDescriptors(std::uint32_t frameIndex)
{
    if (m_gi.size() <= frameIndex)
    {
        return;
    }

    GiFrameResources& gi = m_gi[frameIndex];
    const std::uint32_t parity = gi.historyParity;

    // Trace.
    gi.traceSet->BindBuffer(0, *gi.uniformBuffer);
    if (m_tlas[frameIndex] != nullptr)
    {
        gi.traceSet->BindAccelerationStructure(1, *m_tlas[frameIndex]);
    }
    if (m_geometryDataBuffers[frameIndex] != nullptr)
    {
        gi.traceSet->BindBuffer(2, *m_geometryDataBuffers[frameIndex]);
    }
    gi.traceSet->BindTexture(3, m_gbuffer.GetDepth(), 0u);
    gi.traceSet->BindTexture(4, m_gbuffer.GetNormal(), 0u);
    gi.traceSet->BindTexture(5, *gi.indirect, 0u);
    gi.traceSet->BindStorageImage(6, *gi.diffuseNoisy);
    gi.traceSet->BindStorageImage(7, *gi.specularNoisy);
    gi.traceSet->BindStorageImage(8, *gi.hitDistance);
    gi.traceSet->BindTexture(
        9,
        m_rtSunShadowMask != nullptr ? *m_rtSunShadowMask : *m_whiteTexture,
        0u);
    if (frameIndex < m_giLightBuffers.size() && m_giLightBuffers[frameIndex] != nullptr)
    {
        gi.traceSet->BindBuffer(13, *m_giLightBuffers[frameIndex]);
    }

    // Temporal: reads history[parity], writes history[parity ^ 1].
    gi.temporalSet->BindBuffer(0, *gi.uniformBuffer);
    gi.temporalSet->BindTexture(1, m_gbuffer.GetDepth(), 0u);
    gi.temporalSet->BindTexture(2, m_gbuffer.GetNormal(), 0u);
    gi.temporalSet->BindTexture(3, m_gbuffer.GetMotion(), 0u);
    gi.temporalSet->BindTexture(4, *gi.diffuseNoisy, 0u);
    gi.temporalSet->BindTexture(5, *gi.specularNoisy, 0u);
    gi.temporalSet->BindTexture(6, *gi.diffuseHistory[parity], 0u);
    gi.temporalSet->BindTexture(7, *gi.specularHistory[parity], 0u);
    gi.temporalSet->BindTexture(8, *gi.diffuseMoments[parity], 0u);
    gi.temporalSet->BindTexture(9, *gi.specularMoments[parity], 0u);
    gi.temporalSet->BindStorageImage(10, *gi.diffuseHistory[parity ^ 1]);
    gi.temporalSet->BindStorageImage(11, *gi.specularHistory[parity ^ 1]);
    gi.temporalSet->BindStorageImage(12, *gi.diffuseMoments[parity ^ 1]);
    gi.temporalSet->BindStorageImage(13, *gi.specularMoments[parity ^ 1]);

    // À-trous: one descriptor set per iteration so a set bound to the command buffer is never
    // updated afterwards. The schedule always ends on the filtered texture.
    const std::vector<GiAtrousStep> atrousSteps =
        BuildGiAtrousSchedule(m_giSettings.atrousIterations, kMaxAtrousIterations);
    for (std::uint32_t signal = 0; signal < 2; ++signal)
    {
        const auto atrousTexture = [&](GiAtrousTexture which) -> rhi::Texture* {
            if (which == GiAtrousTexture::History)
            {
                return signal == 0 ? gi.diffuseHistory[parity ^ 1].get()
                                   : gi.specularHistory[parity ^ 1].get();
            }
            if (which == GiAtrousTexture::Noisy)
            {
                return signal == 0 ? gi.diffuseNoisy.get() : gi.specularNoisy.get();
            }
            return signal == 0 ? gi.diffuseFiltered.get() : gi.specularFiltered.get();
        };
        rhi::Texture* const moments =
            signal == 0 ? gi.diffuseMoments[parity ^ 1].get() : gi.specularMoments[parity ^ 1].get();

        for (std::uint32_t stepIndex = 0; stepIndex < atrousSteps.size(); ++stepIndex)
        {
            const GiAtrousStep& step = atrousSteps[stepIndex];
            rhi::DescriptorSet& set = *gi.atrousSets[signal * kMaxAtrousIterations + stepIndex];
            set.BindBuffer(0, *gi.uniformBuffer);
            set.BindTexture(1, m_gbuffer.GetDepth(), 0u);
            set.BindTexture(2, m_gbuffer.GetNormal(), 0u);
            set.BindTexture(3, *gi.hitDistance, 0u);
            set.BindTexture(4, *atrousTexture(step.input), 0u);
            set.BindTexture(5, *moments, 0u);
            set.BindStorageImage(6, *atrousTexture(step.output));
        }
    }

    gi.upsampleSet->BindBuffer(0, *gi.uniformBuffer);
    gi.upsampleSet->BindTexture(1, *gi.diffuseFiltered, 0u);
    gi.upsampleSet->BindTexture(2, *gi.specularFiltered, 0u);
    gi.upsampleSet->BindTexture(3, m_gbuffer.GetAlbedo(), 0u);
    gi.upsampleSet->BindTexture(4, m_gbuffer.GetNormal(), 0u);
    gi.upsampleSet->BindTexture(5, m_gbuffer.GetDepth(), 0u);
    gi.upsampleSet->BindStorageImage(6, *gi.indirect);
}

void DeferredRenderer::RecordGiPasses(rhi::CommandList& commandList, std::uint32_t frameIndex)
{
    if (!IsPerPixelGiActive() || m_gi.size() <= frameIndex)
    {
        return;
    }

    GiFrameResources& gi = m_gi[frameIndex];
    if (gi.traceSet == nullptr || m_tlas[frameIndex] == nullptr ||
        m_geometryDataBuffers[frameIndex] == nullptr)
    {
        return;
    }

    const std::uint32_t halfWidth = gi.diffuseNoisy->GetWidth();
    const std::uint32_t halfHeight = gi.diffuseNoisy->GetHeight();
    const std::uint32_t groupsX = (halfWidth + 7) / 8;
    const std::uint32_t groupsY = (halfHeight + 7) / 8;
    const std::uint32_t iterations =
        std::clamp(m_giSettings.atrousIterations, 1u, kMaxAtrousIterations);

    // Bind after the G-buffer pass so the G-buffer textures are in a sampled layout.
    BindGiDescriptors(frameIndex);

    commandList.BeginGpuZone("GlobalIllumination");

    commandList.BindPipeline(*m_giTracePipeline);
    commandList.BindDescriptorSet(0, *gi.traceSet);
    commandList.BindDescriptorSet(1, *m_bindlessSet);
    commandList.Dispatch(groupsX, groupsY, 1);
    commandList.ShaderMemoryBarrier(rhi::ShaderStageMask::Compute, rhi::ShaderStageMask::Compute);

    const bool skipTemporal = std::getenv("MRE_NOTEMPORAL") != nullptr;
    const bool skipAtrous = std::getenv("MRE_NOATROUS") != nullptr;
    const bool skipUpsample = std::getenv("MRE_NOUPSAMPLE") != nullptr;

    if (!skipTemporal)
    {
        commandList.BindPipeline(*m_giTemporalPipeline);
        commandList.BindDescriptorSet(0, *gi.temporalSet);
        commandList.Dispatch(groupsX, groupsY, 1);
        commandList.ShaderMemoryBarrier(rhi::ShaderStageMask::Compute, rhi::ShaderStageMask::Compute);
    }

    if (!skipAtrous)
    {
        commandList.BindPipeline(*m_giAtrousPipeline);
        for (std::uint32_t signal = 0; signal < 2; ++signal)
        {
            for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
            {
                commandList.BindDescriptorSet(
                    0,
                    *gi.atrousSets[signal * kMaxAtrousIterations + iteration]);
                const GiAtrousPushConstants pushConstants{signal, iteration};
                commandList.SetPushConstants(
                    &pushConstants,
                    sizeof(pushConstants),
                    rhi::ShaderStageMask::Compute);
                commandList.Dispatch(groupsX, groupsY, 1);
                commandList.ShaderMemoryBarrier(
                    rhi::ShaderStageMask::Compute,
                    rhi::ShaderStageMask::Compute);
            }
        }
    }

    if (!skipUpsample)
    {
        commandList.BindPipeline(*m_giUpsamplePipeline);
        commandList.BindDescriptorSet(0, *gi.upsampleSet);
        commandList.Dispatch(
            (m_gbuffer.GetWidth() + 7) / 8,
            (m_gbuffer.GetHeight() + 7) / 8,
            1);
        commandList.ShaderMemoryBarrier(
            rhi::ShaderStageMask::Compute,
            rhi::ShaderStageMask::Fragment);
    }

    gi.historyParity ^= 1;
    gi.historyValid = true;

    commandList.EndGpuZone();
}

void DeferredRenderer::ApplyModelScale()
{
    MRE_ASSERT(
        m_modelBaseTransforms.size() == m_modelDrawItems.size(),
        "Model draw items and base transforms are out of sync");

    const glm::mat4 scaleMatrix = glm::scale(glm::mat4{1.0f}, glm::vec3{m_modelScale});
    for (std::size_t index = 0; index < m_modelDrawItems.size(); ++index)
    {
        m_modelDrawItems[index].pushConstants.model = scaleMatrix * m_modelBaseTransforms[index];
    }
}

void DeferredRenderer::Render(rhi::CommandList& commandList, const FrameView& view)
{
    MRE_PROFILE_ZONE_N("DeferredRenderer::Render");
    MRE_ASSERT(m_initialized, "DeferredRenderer::Render called before DeferredRenderer::Init");
    MRE_ASSERT(m_device != nullptr, "DeferredRenderer requires a device");

    const std::uint32_t frameIndex = m_device->GetFrameIndex();
    MRE_ASSERT(frameIndex < m_sceneSets.size(), "Frame index out of range");

    UpdateUniforms(frameIndex, view);
    UpdateObjectData(frameIndex);
    BuildAccelerationStructures(commandList, frameIndex);

    rhi::RenderingDesc gbufferDesc{};
    gbufferDesc.colorAttachments = {
        rhi::ColorAttachment{&m_gbuffer.GetAlbedo(), rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f}, true},
        rhi::ColorAttachment{&m_gbuffer.GetNormal(), rhi::ClearColor{0.5f, 0.5f, 1.0f, 1.0f}, true},
        rhi::ColorAttachment{&m_gbuffer.GetPosition(), rhi::ClearColor{0.0f, 0.0f, 0.0f, 0.0f}, true},
        rhi::ColorAttachment{&m_gbuffer.GetMotion(), rhi::ClearColor{0.0f, 0.0f, 0.0f, 0.0f}, true},
        rhi::ColorAttachment{
            &m_gbuffer.GetSpecularF0(),
            rhi::ClearColor{0.04f, 0.04f, 0.04f, 1.0f},
            true},
        rhi::ColorAttachment{
            &m_gbuffer.GetTransmission(),
            rhi::ClearColor{0.0f, 1.5f, 0.0f, 0.0f},
            true},
        rhi::ColorAttachment{
            &m_gbuffer.GetAttenuation(),
            rhi::ClearColor{1.0f, 1.0f, 1.0f, 0.0f},
            true},
        rhi::ColorAttachment{
            &m_gbuffer.GetSheen(),
            rhi::ClearColor{0.0f, 0.0f, 0.0f, 0.0f},
            true},
    };
    gbufferDesc.depthAttachment = rhi::DepthAttachment{&m_gbuffer.GetDepth(), 1.0f, true};

    // Alpha blended materials skip the opaque G-buffer and are drawn back-to-front after lighting.
    std::vector<const DrawItem*> gbufferItems{};
    std::vector<const DrawItem*> transparentItems{};
    gbufferItems.reserve(m_sceneDrawItems.size() + m_modelDrawItems.size());
    transparentItems.reserve(m_modelDrawItems.size());
    const auto classifyItem = [&gbufferItems, &transparentItems](const DrawItem& item) {
        if (item.alphaBlend)
        {
            transparentItems.push_back(&item);
        }
        else
        {
            gbufferItems.push_back(&item);
        }
    };
    for (const DrawItem& item : m_sceneDrawItems)
    {
        classifyItem(item);
    }
    for (const DrawItem& item : m_modelDrawItems)
    {
        classifyItem(item);
    }

    if (!transparentItems.empty())
    {
        const glm::vec3 cameraForward =
            -glm::vec3{view.view[0][2], view.view[1][2], view.view[2][2]};
        std::sort(
            transparentItems.begin(),
            transparentItems.end(),
            [&view, &cameraForward](const DrawItem* left, const DrawItem* right) {
                const float leftDepth = glm::dot(
                    glm::vec3{left->pushConstants.model[3]} - view.position,
                    cameraForward);
                const float rightDepth = glm::dot(
                    glm::vec3{right->pushConstants.model[3]} - view.position,
                    cameraForward);
                return leftDepth > rightDepth;
            });
    }

    BuildShadowCullItems();
    BuildShadowViews();

    if (!m_shadowViews.empty())
    {
        commandList.BeginGpuZone("ShadowPass");

        const bool useShadowSecondaries =
            m_jobSystem != nullptr && frameIndex < m_shadowSecondaries.size() &&
            m_shadowViews.size() <= m_shadowSecondaries[frameIndex].size();

        if (useShadowSecondaries)
        {
            RecordShadowViews(frameIndex);

            for (std::size_t index = 0; index < m_shadowViews.size(); ++index)
            {
                const ShadowView& shadowView = m_shadowViews[index];

                rhi::RenderingDesc shadowDesc{};
                shadowDesc.depthAttachment.texture = shadowView.texture;
                shadowDesc.depthAttachment.clearDepth = 1.0f;
                shadowDesc.depthAttachment.clear = true;
                shadowDesc.depthAttachment.layer = shadowView.layer;
                shadowDesc.area = shadowView.area;
                shadowDesc.secondaryCommandBuffers = true;

                commandList.BeginRendering(shadowDesc);
                commandList.ExecuteCommands({m_shadowSecondaries[frameIndex][index].get()});
                commandList.EndRendering();
            }
        }
        else
        {
            for (const ShadowView& shadowView : m_shadowViews)
            {
                rhi::RenderingDesc shadowDesc{};
                shadowDesc.depthAttachment.texture = shadowView.texture;
                shadowDesc.depthAttachment.clearDepth = 1.0f;
                shadowDesc.depthAttachment.clear = true;
                shadowDesc.depthAttachment.layer = shadowView.layer;
                shadowDesc.area = shadowView.area;

                commandList.BeginRendering(shadowDesc);
                commandList.BindPipeline(*m_shadowPipeline);
                commandList.BindDescriptorSet(0, *m_shadowSets[frameIndex]);
                commandList.BindDescriptorSet(1, *m_bindlessSet);
                commandList.SetDepthBias(
                    m_shadowSettings.depthBias,
                    m_shadowSettings.depthBias * 1.5f,
                    0.0f);

                ShadowPushConstants pushConstants{};
                pushConstants.viewIndex = shadowView.viewIndex;
                const std::uint32_t pushConstantSize = m_shadowPipeline->GetPushConstantSize();
                for (const DrawItem* item : shadowView.items)
                {
                    pushConstants.object = item->pushConstants;
                    commandList.BindVertexBuffer(*item->vertexBuffer);
                    commandList.BindIndexBuffer(*item->indexBuffer);
                    commandList.SetPushConstants(
                        &pushConstants,
                        pushConstantSize,
                        rhi::ShaderStageMask::VertexFragment);
                    commandList.DrawIndexed(item->indexCount, item->firstIndex);
                }
                commandList.EndRendering();
            }
        }

        commandList.EndGpuZone();
    }

    const auto drawItems = [&commandList, frameIndex](const std::vector<const DrawItem*>& items) {
        for (const DrawItem* item : items)
        {
            commandList.BindVertexBuffer(*item->vertexBuffer);
            commandList.BindIndexBuffer(*item->indexBuffer);
            commandList.SetPushConstants(
                &item->pushConstants,
                sizeof(ObjectPushConstants),
                rhi::ShaderStageMask::VertexFragment);
            commandList.DrawIndexed(item->indexCount, item->firstIndex);
        }
    };

    const bool useSecondaries =
        m_jobSystem != nullptr && m_secondaryBatchCount > 1 && gbufferItems.size() > 1 &&
        frameIndex < m_gbufferSecondaries.size();

    commandList.BeginGpuZone("GBuffer");

    if (useSecondaries)
    {
        const std::uint32_t batchCount = RecordGBufferSecondaries(frameIndex, gbufferDesc, gbufferItems);

        std::vector<const rhi::CommandList*> secondaries{};
        secondaries.reserve(batchCount);
        for (std::uint32_t batch = 0; batch < batchCount; ++batch)
        {
            secondaries.push_back(m_gbufferSecondaries[frameIndex][batch].get());
        }

        gbufferDesc.secondaryCommandBuffers = true;
        commandList.BeginRendering(gbufferDesc);
        commandList.ExecuteCommands(secondaries);
        commandList.EndRendering();
    }
    else
    {
        commandList.BeginRendering(gbufferDesc);
        commandList.BindPipeline(*m_gbufferPipeline);
        commandList.BindDescriptorSet(0, *m_sceneSets[frameIndex]);
        commandList.BindDescriptorSet(1, *m_bindlessSet);
        drawItems(gbufferItems);
        commandList.EndRendering();
    }

    commandList.EndGpuZone();

    const bool probeGiActive = IsProbeGiActive();
    if (probeGiActive != m_probeGiActiveLastFrame)
    {
        // Switching modes invalidates the per-pixel GI history.
        for (GiFrameResources& gi : m_gi)
        {
            gi.historyValid = false;
        }
        m_probeGiActiveLastFrame = probeGiActive;
    }

    RecordRtShadowPass(commandList, frameIndex);
    RecordGiProbes(commandList, frameIndex);

    if (m_lightingSetsDirty)
    {
        CreateLightingSets(*m_device);
        m_lightingSetsDirty = false;
    }

    if (m_clusterGrid.count > 0 && !m_gpuLights.empty())
    {
        commandList.BeginGpuZone("LightCulling");
        commandList.BindPipeline(*m_clusterCullingPipeline);
        commandList.BindDescriptorSet(0, *m_clusterSets[frameIndex]);
        commandList.Dispatch(
            (m_clusterGrid.x + 3) / 4,
            (m_clusterGrid.y + 3) / 4,
            m_clusterGrid.z);
        commandList.EndGpuZone();
        commandList.ShaderMemoryBarrier(rhi::ShaderStageMask::Compute, rhi::ShaderStageMask::Fragment);
    }

    RecordGiPasses(commandList, frameIndex);
    RecordReflectionPasses(commandList, frameIndex);
    RecordCloudPasses(commandList, frameIndex);

    commandList.BeginGpuZone("Lighting");
    commandList.BeginSwapchainRendering(
        rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f},
        rhi::DepthAttachment{&m_gbuffer.GetDepth(), 1.0f, false},
        true);
    commandList.BindPipeline(*m_lightingPipeline);
    commandList.BindDescriptorSet(0, *m_lightingSets[frameIndex]);
    commandList.BindDescriptorSet(1, *m_bindlessSet);
    commandList.Draw(3);

    if (!transparentItems.empty())
    {
        commandList.BeginGpuZone("Transparent");
        commandList.BindPipeline(*m_transparentPipeline);
        commandList.BindDescriptorSet(0, *m_sceneSets[frameIndex]);
        commandList.BindDescriptorSet(1, *m_bindlessSet);
        for (const DrawItem* item : transparentItems)
        {
            commandList.BindVertexBuffer(*item->vertexBuffer);
            commandList.BindIndexBuffer(*item->indexBuffer);
            commandList.SetPushConstants(
                &item->pushConstants,
                sizeof(ObjectPushConstants),
                rhi::ShaderStageMask::VertexFragment);
            commandList.DrawIndexed(item->indexCount, item->firstIndex);
        }
        commandList.EndGpuZone();
    }

    RecordDebugLines(commandList, view);
    commandList.EndSwapchainRendering();
    commandList.EndGpuZone();
}

void DeferredRenderer::RecordDebugLines(rhi::CommandList& commandList, const FrameView& view)
{
    if (m_debugLinePipeline == nullptr || (!m_showLightGizmos && !m_showProbeGizmos))
    {
        return;
    }

    DebugLines lines{};
    if (m_showLightGizmos)
    {
        for (const Light& light : m_lights)
        {
            const glm::vec4 color{light.color.r, light.color.g, light.color.b, 1.0f};
            const float directionLength = glm::length(light.direction);
            const glm::vec3 direction = directionLength > 1e-4f
                ? light.direction / directionLength
                : glm::vec3{0.0f, -1.0f, 0.0f};

            switch (light.type)
            {
            case LightType::Directional:
                lines.AddDirectionalLight(direction, color);
                break;
            case LightType::Point:
                lines.AddPointLight(light.position, std::max(light.range, 0.1f), color);
                break;
            case LightType::Spot:
                lines.AddSpotLight(
                    light.position,
                    direction,
                    std::max(light.range, 0.1f),
                    light.innerConeAngle,
                    light.outerConeAngle,
                    color);
                break;
            }
        }
    }

    if (m_showProbeGizmos)
    {
        // Draw the actual GI probes so the markers match what the lighting pass samples. The
        // arrow points along the dominant SH direction and is tinted by the probe irradiance.
        for (std::size_t index = 0; index < m_giProbes.size(); ++index)
        {
            const GiProbe& marker = m_giProbes[index];
            if (index >= m_giProbeData.size())
            {
                // Probes beyond the GPU budget still get a plain marker.
                lines.AddProbe(marker);
                continue;
            }

            const GpuGiProbe& probe = m_giProbeData[index];
            const glm::vec3 position{probe.position};
            lines.AddSphere(position, marker.radius, marker.color);
            lines.AddCross(
                position,
                marker.radius * 0.6f,
                glm::vec4{
                    marker.color.r * 0.7f,
                    marker.color.g * 0.7f,
                    marker.color.b * 0.7f,
                    marker.color.a});

            const glm::vec3 direction{
                probe.shX.r + probe.shX.g + probe.shX.b,
                probe.shY.r + probe.shY.g + probe.shY.b,
                probe.shZ.r + probe.shZ.g + probe.shZ.b};
            const float directionLength = glm::length(direction);
            if (directionLength > 1e-4f)
            {
                const glm::vec3 irradiance =
                    glm::clamp(glm::vec3{probe.sh0}, glm::vec3{0.0f}, glm::vec3{1.0f});
                lines.AddArrow(
                    position,
                    direction / directionLength,
                    0.6f,
                    glm::vec4{irradiance, 1.0f});
            }
        }
    }

    if (lines.Empty())
    {
        return;
    }

    const std::uint32_t frameIndex = m_device->GetFrameIndex();
    if (m_debugLineBuffers.size() <= frameIndex)
    {
        m_debugLineBuffers.resize(m_device->GetFramesInFlight());
    }

    const std::uint64_t bytes = lines.VertexData().size() * sizeof(float);
    if (m_debugLineBuffers[frameIndex] == nullptr ||
        m_debugLineBuffers[frameIndex]->GetSize() < bytes)
    {
        const std::uint64_t capacity = std::max<std::uint64_t>(bytes * 2, 1u << 16);
        m_debugLineBuffers[frameIndex] = m_device->CreateBuffer(
            rhi::BufferDesc{capacity, rhi::BufferUsage::Vertex, true});
        MRE_ASSERT(m_debugLineBuffers[frameIndex] != nullptr, "Failed to create the debug line buffer");
    }

    m_debugLineBuffers[frameIndex]->Write(lines.VertexData().data(), bytes);

    const glm::mat4 viewProjection = view.viewProjection;
    commandList.BindPipeline(*m_debugLinePipeline);
    commandList.SetPushConstants(
        &viewProjection,
        sizeof(viewProjection),
        rhi::ShaderStageMask::Vertex);
    commandList.BindVertexBuffer(*m_debugLineBuffers[frameIndex]);
    commandList.Draw(lines.VertexCount());
}

std::uint32_t DeferredRenderer::RecordGBufferSecondaries(
    std::uint32_t frameIndex,
    const rhi::RenderingDesc& desc,
    const std::vector<const DrawItem*>& items)
{
    MRE_PROFILE_ZONE_N("RecordGBufferSecondaries");
    MRE_ASSERT(m_jobSystem != nullptr, "RecordGBufferSecondaries requires a job system");
    MRE_ASSERT(frameIndex < m_gbufferSecondaries.size(), "Frame index out of range");
    MRE_ASSERT(!items.empty(), "RecordGBufferSecondaries requires draw items");

    const std::uint32_t batchCount = std::min(
        m_secondaryBatchCount,
        static_cast<std::uint32_t>(items.size()));
    MRE_ASSERT(batchCount > 0, "RecordGBufferSecondaries requires at least one batch");
    MRE_ASSERT(
        batchCount <= m_gbufferSecondaries[frameIndex].size(),
        "Not enough secondary command lists were allocated");

    std::vector<std::future<void>> futures{};
    futures.reserve(batchCount);

    for (std::uint32_t batch = 0; batch < batchCount; ++batch)
    {
        // Even split: with items.size() >= batchCount every batch owns at least one item.
        const std::size_t begin = static_cast<std::size_t>(batch) * items.size() / batchCount;
        const std::size_t end =
            static_cast<std::size_t>(batch + 1) * items.size() / batchCount;
        MRE_ASSERT(begin < end, "RecordGBufferSecondaries produced an empty batch");

        rhi::CommandList* secondary = m_gbufferSecondaries[frameIndex][batch].get();
        futures.push_back(m_jobSystem->Submit([this, frameIndex, secondary, &desc, &items, begin, end]() {
            RecordGBufferBatch(frameIndex, items, begin, end, *secondary, desc);
        }));
    }

    for (std::future<void>& future : futures)
    {
        future.get();
    }

    return batchCount;
}

void DeferredRenderer::RecordGBufferBatch(
    std::uint32_t frameIndex,
    const std::vector<const DrawItem*>& items,
    std::size_t begin,
    std::size_t end,
    rhi::CommandList& secondary,
    const rhi::RenderingDesc& desc)
{
    MRE_ASSERT(m_gbufferPipeline != nullptr, "RecordGBufferBatch requires the G-buffer pipeline");
    MRE_ASSERT(frameIndex < m_sceneSets.size(), "Frame index out of range");

    secondary.BeginSecondary(desc);
    secondary.BindPipeline(*m_gbufferPipeline);
    secondary.BindDescriptorSet(0, *m_sceneSets[frameIndex]);
    secondary.BindDescriptorSet(1, *m_bindlessSet);

    secondary.BeginGpuZone("GBufferBatch");

    for (std::size_t index = begin; index < end; ++index)
    {
        const DrawItem& item = *items[index];
        secondary.BindVertexBuffer(*item.vertexBuffer);
        secondary.BindIndexBuffer(*item.indexBuffer);
        secondary.SetPushConstants(
            &item.pushConstants,
            sizeof(ObjectPushConstants),
            rhi::ShaderStageMask::VertexFragment);
        secondary.DrawIndexed(item.indexCount, item.firstIndex);
    }

    secondary.EndGpuZone();
    secondary.End();
}

void DeferredRenderer::BuildShadowCullItems()
{
    m_shadowCullItems.clear();

    // The ground plane spans 20x20 world units around the origin.
    constexpr float kGroundRadius = 14.15f;
    for (const DrawItem& item : m_sceneDrawItems)
    {
        m_shadowCullItems.push_back(ShadowCullItem{&item, glm::vec3{0.0f}, kGroundRadius});
    }

    for (const SceneObjectInstance& instance : m_objectInstances)
    {
        for (std::size_t index = 0; index < instance.drawItemCount; ++index)
        {
            m_shadowCullItems.push_back(ShadowCullItem{
                &m_modelDrawItems[instance.firstDrawItem + index],
                instance.bounds.center,
                instance.bounds.radius});
        }
    }
}

std::vector<const DeferredRenderer::DrawItem*> DeferredRenderer::CollectShadowItems(
    const glm::vec3& lightPosition,
    float range) const
{
    std::vector<const DrawItem*> items{};
    for (const ShadowCullItem& cullItem : m_shadowCullItems)
    {
        const float distance = glm::length(cullItem.center - lightPosition);
        if (distance - cullItem.radius <= range)
        {
            items.push_back(cullItem.item);
        }
    }
    return items;
}

void DeferredRenderer::BuildShadowViews()
{
    m_shadowViews.clear();

    if (!m_shadowSettings.enabled)
    {
        return;
    }

    // Directional cascades cover the whole scene; spot and point faces are culled by light range.
    if (FindShadowLight() != nullptr && m_shadowMap != nullptr)
    {
        for (std::uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
        {
            ShadowView view{};
            view.viewIndex = cascade;
            view.texture = m_shadowMap.get();
            view.layer = cascade;
            view.items.reserve(m_shadowCullItems.size());
            for (const ShadowCullItem& cullItem : m_shadowCullItems)
            {
                view.items.push_back(cullItem.item);
            }
            m_shadowViews.push_back(std::move(view));
        }
    }

    if (m_spotShadowMap != nullptr)
    {
        for (std::size_t slot = 0; slot < m_spotShadowLights.size(); ++slot)
        {
            const Light& light = m_spotShadowLights[slot];

            ShadowView view{};
            view.viewIndex = kSpotShadowOffset + static_cast<std::uint32_t>(slot);
            view.texture = m_spotShadowMap.get();
            view.layer = static_cast<std::uint32_t>(slot);
            view.items = CollectShadowItems(light.position, light.range);
            m_shadowViews.push_back(std::move(view));
        }
    }

    if (!m_pointShadowMaps.empty())
    {
        // Every point shadow slot owns its own six-layer map. With dynamic updates a map is only
        // re-rendered when the light moved/changed or the scene geometry changed, so static lights
        // keep their cached shadow map; moving lights still refresh every frame.
        for (std::size_t slot = 0; slot < m_pointShadowLights.size() &&
                                   slot < m_pointShadowMaps.size() &&
                                   slot < m_pointShadowSlotStates.size();
             ++slot)
        {
            const Light& light = m_pointShadowLights[slot];
            PointShadowSlotState& state = m_pointShadowSlotStates[slot];
            const bool needsUpdate =
                !m_shadowSettings.dynamicPointShadows ||
                PointShadowNeedsUpdate(state, light.position, light.range, m_geometryRevision);
            if (!needsUpdate)
            {
                continue;
            }

            std::vector<const DrawItem*> items = CollectShadowItems(light.position, light.range);
            for (std::uint32_t face = 0; face < kPointShadowFaceCount; ++face)
            {
                ShadowView view{};
                view.viewIndex = kPointShadowOffset +
                                 static_cast<std::uint32_t>(slot) * kPointShadowFaceCount + face;
                view.texture = m_pointShadowMaps[slot].get();
                view.layer = face;
                view.items = items;
                m_shadowViews.push_back(std::move(view));
            }

            // The views clear to 1.0 even when nothing is culled, so a cached map is always defined.
            state.position = light.position;
            state.range = light.range;
            state.geometryRevision = m_geometryRevision;
            state.valid = true;
        }
    }
}

void DeferredRenderer::RecordShadowViews(std::uint32_t frameIndex)
{
    MRE_PROFILE_ZONE_N("RecordShadowViews");
    MRE_ASSERT(m_jobSystem != nullptr, "RecordShadowViews requires a job system");
    MRE_ASSERT(frameIndex < m_shadowSecondaries.size(), "Frame index out of range");
    MRE_ASSERT(
        m_shadowViews.size() <= m_shadowSecondaries[frameIndex].size(),
        "Not enough shadow secondary command lists were allocated");

    std::vector<std::future<void>> futures{};
    futures.reserve(m_shadowViews.size());

    for (std::size_t index = 0; index < m_shadowViews.size(); ++index)
    {
        const ShadowView& view = m_shadowViews[index];
        rhi::CommandList* secondary = m_shadowSecondaries[frameIndex][index].get();

        rhi::RenderingDesc desc{};
        desc.depthAttachment.texture = view.texture;
        desc.depthAttachment.clearDepth = 1.0f;
        desc.depthAttachment.clear = true;
        desc.depthAttachment.layer = view.layer;
        desc.area = view.area;

        futures.push_back(m_jobSystem->Submit([this, frameIndex, secondary, &view, desc]() {
            RecordShadowView(frameIndex, view.viewIndex, view.items, *secondary, desc);
        }));
    }

    for (std::future<void>& future : futures)
    {
        future.get();
    }
}

void DeferredRenderer::RecordShadowView(
    std::uint32_t frameIndex,
    std::uint32_t shadowViewIndex,
    const std::vector<const DrawItem*>& items,
    rhi::CommandList& secondary,
    const rhi::RenderingDesc& desc)
{
    MRE_ASSERT(m_shadowPipeline != nullptr, "RecordShadowView requires the shadow pipeline");
    MRE_ASSERT(frameIndex < m_shadowSets.size(), "Frame index out of range");

    secondary.BeginSecondary(desc);
    secondary.BindPipeline(*m_shadowPipeline);
    secondary.BindDescriptorSet(0, *m_shadowSets[frameIndex]);
    secondary.BindDescriptorSet(1, *m_bindlessSet);
    secondary.SetDepthBias(m_shadowSettings.depthBias, m_shadowSettings.depthBias * 1.5f, 0.0f);
    secondary.BeginGpuZone("ShadowView");

    ShadowPushConstants pushConstants{};
    pushConstants.viewIndex = shadowViewIndex;

    const std::uint32_t pushConstantSize = m_shadowPipeline->GetPushConstantSize();
    for (const DrawItem* item : items)
    {
        pushConstants.object = item->pushConstants;
        secondary.BindVertexBuffer(*item->vertexBuffer);
        secondary.BindIndexBuffer(*item->indexBuffer);
        secondary.SetPushConstants(
            &pushConstants,
            pushConstantSize,
            rhi::ShaderStageMask::VertexFragment);
        secondary.DrawIndexed(item->indexCount, item->firstIndex);
    }

    secondary.EndGpuZone();
    secondary.End();
}

bool DeferredRenderer::LoadShaders(rhi::Device& device, const std::filesystem::path& shaderDirectory)
{
    m_gbufferVertexShader = device.CreateShader(
        rhi::ShaderStage::Vertex,
        shaderDirectory / "gbuffer.vert.spv");
    m_gbufferFragmentShader = device.CreateShader(
        rhi::ShaderStage::Fragment,
        shaderDirectory / "gbuffer.frag.spv");
    m_transparentFragmentShader = device.CreateShader(
        rhi::ShaderStage::Fragment,
        shaderDirectory / "transparent.frag.spv");
    m_lightingVertexShader = device.CreateShader(
        rhi::ShaderStage::Vertex,
        shaderDirectory / "lighting.vert.spv");
    m_lightingFragmentShader = device.CreateShader(
        rhi::ShaderStage::Fragment,
        shaderDirectory / "lighting.frag.spv");
    m_clusterCullingShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "cluster_culling.comp.spv");
    m_shadowVertexShader = device.CreateShader(
        rhi::ShaderStage::Vertex,
        shaderDirectory / "shadow.vert.spv");
    m_shadowFragmentShader = device.CreateShader(
        rhi::ShaderStage::Fragment,
        shaderDirectory / "shadow.frag.spv");
    m_giTraceShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "gi_trace.comp.spv");
    m_giTemporalShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "gi_temporal.comp.spv");
    m_giAtrousShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "gi_atrous.comp.spv");
    m_giUpsampleShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "gi_upsample.comp.spv");
    m_rtShadowShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "rt_shadows.comp.spv");
    m_giProbeShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "gi_probes.comp.spv");
    m_rtReflectionShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "rt_reflections.comp.spv");
    m_reflectionTemporalShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "reflection_temporal.comp.spv");
    m_reflectionUpsampleShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "reflection_upsample.comp.spv");
    m_debugLineVertexShader = device.CreateShader(
        rhi::ShaderStage::Vertex,
        shaderDirectory / "debug_line.vert.spv");
    m_debugLineFragmentShader = device.CreateShader(
        rhi::ShaderStage::Fragment,
        shaderDirectory / "debug_line.frag.spv");
    m_cloudShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "clouds.comp.spv");
    m_cloudUpsampleShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "cloud_upsample.comp.spv");
    m_cloudReprojectShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "cloud_reproject.comp.spv");
    m_cloudGodRayShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "cloud_godrays.comp.spv");

    if (m_gbufferVertexShader == nullptr || m_gbufferFragmentShader == nullptr ||
        m_transparentFragmentShader == nullptr ||
        m_lightingVertexShader == nullptr || m_lightingFragmentShader == nullptr ||
        m_clusterCullingShader == nullptr || m_shadowVertexShader == nullptr ||
        m_shadowFragmentShader == nullptr || m_giTraceShader == nullptr ||
        m_giTemporalShader == nullptr || m_giAtrousShader == nullptr ||
        m_giUpsampleShader == nullptr || m_rtShadowShader == nullptr ||
        m_giProbeShader == nullptr || m_rtReflectionShader == nullptr ||
        m_reflectionTemporalShader == nullptr || m_reflectionUpsampleShader == nullptr ||
        m_debugLineVertexShader == nullptr || m_debugLineFragmentShader == nullptr ||
        m_cloudShader == nullptr || m_cloudUpsampleShader == nullptr ||
        m_cloudReprojectShader == nullptr || m_cloudGodRayShader == nullptr)
    {
        MRE_CRITICAL("Failed to load the deferred shaders from '{}'", PathToUtf8(shaderDirectory));
        return false;
    }
    return true;
}

bool DeferredRenderer::CreatePipelines(rhi::Device& device)
{
    const std::array<const rhi::Shader*, 2> gbufferStages{
        m_gbufferVertexShader.get(),
        m_gbufferFragmentShader.get()};

    rhi::PipelineDesc gbufferPipelineDesc{};
    gbufferPipelineDesc.vertexShader = m_gbufferVertexShader.get();
    gbufferPipelineDesc.fragmentShader = m_gbufferFragmentShader.get();
    rhi::ApplyShaderReflection(gbufferPipelineDesc, gbufferStages);
    gbufferPipelineDesc.depthTest = true;
    gbufferPipelineDesc.depthWrite = true;
    gbufferPipelineDesc.blend = false;
    gbufferPipelineDesc.cullMode = rhi::CullMode::None;
    gbufferPipelineDesc.descriptorSetLayouts = {m_sceneSetLayout.get(), m_bindlessSetLayout.get()};
    gbufferPipelineDesc.colorFormats = {
        rhi::Format::R8G8B8A8Unorm,
        rhi::Format::R8G8B8A8Unorm,
    rhi::Format::R16G16B16A16Sfloat,
    rhi::Format::R16G16B16A16Sfloat,
    rhi::Format::R8G8B8A8Unorm,
    rhi::Format::R16G16B16A16Sfloat,
        rhi::Format::R8G8B8A8Unorm,
        rhi::Format::R8G8B8A8Unorm};
    gbufferPipelineDesc.depthFormat = rhi::Format::D32Sfloat;
    m_gbufferPipeline = device.CreatePipeline(gbufferPipelineDesc);

    // Forward pass for alpha blended materials: shares the G-buffer vertex shader, blends over the
    // lit image and tests against the opaque depth without writing it.
    const std::array<const rhi::Shader*, 2> transparentStages{
        m_gbufferVertexShader.get(),
        m_transparentFragmentShader.get()};
    rhi::PipelineDesc transparentPipelineDesc{};
    transparentPipelineDesc.vertexShader = m_gbufferVertexShader.get();
    transparentPipelineDesc.fragmentShader = m_transparentFragmentShader.get();
    rhi::ApplyShaderReflection(transparentPipelineDesc, transparentStages);
    transparentPipelineDesc.depthTest = true;
    transparentPipelineDesc.depthWrite = false;
    transparentPipelineDesc.blend = true;
    transparentPipelineDesc.cullMode = rhi::CullMode::None;
    transparentPipelineDesc.descriptorSetLayouts = {m_sceneSetLayout.get(), m_bindlessSetLayout.get()};
    transparentPipelineDesc.colorFormats = {device.GetSwapchainInfo().format};
    transparentPipelineDesc.depthFormat = rhi::Format::D32Sfloat;
    m_transparentPipeline = device.CreatePipeline(transparentPipelineDesc);

    rhi::PipelineDesc lightingPipelineDesc{};
    lightingPipelineDesc.vertexShader = m_lightingVertexShader.get();
    lightingPipelineDesc.fragmentShader = m_lightingFragmentShader.get();
    lightingPipelineDesc.depthTest = false;
    lightingPipelineDesc.depthWrite = false;
    lightingPipelineDesc.blend = false;
    lightingPipelineDesc.cullMode = rhi::CullMode::None;
    // Set 1 is the bindless material set used for the emissive texture lookup.
    lightingPipelineDesc.descriptorSetLayouts = {m_lightingSetLayout.get(), m_bindlessSetLayout.get()};
    lightingPipelineDesc.colorFormats = {device.GetSwapchainInfo().format};
    // The swapchain scope attaches the G-buffer depth for the transparent pass; pipelines drawn in
    // that scope must declare the matching depth format even when depth testing is disabled.
    lightingPipelineDesc.depthFormat = rhi::Format::D32Sfloat;
    m_lightingPipeline = device.CreatePipeline(lightingPipelineDesc);

    rhi::PipelineDesc clusterPipelineDesc{};
    clusterPipelineDesc.computeShader = m_clusterCullingShader.get();
    clusterPipelineDesc.descriptorSetLayouts = {m_clusterSetLayout.get()};
    m_clusterCullingPipeline = device.CreatePipeline(clusterPipelineDesc);

    const std::array<const rhi::Shader*, 2> shadowStages{
        m_shadowVertexShader.get(),
        m_shadowFragmentShader.get()};

    rhi::PipelineDesc shadowPipelineDesc{};
    shadowPipelineDesc.vertexShader = m_shadowVertexShader.get();
    shadowPipelineDesc.fragmentShader = m_shadowFragmentShader.get();
    rhi::ApplyShaderReflection(shadowPipelineDesc, shadowStages);
    // The shadow pass consumes positions and UVs from the interleaved GltfVertex layout
    // (position at byte 0, uv at byte 24); reflection cannot infer the interleaved offsets.
    shadowPipelineDesc.vertexLayout = rhi::VertexLayout{};
    shadowPipelineDesc.vertexLayout.AddAttribute(0, rhi::Format::R32G32B32Sfloat, 0);
    shadowPipelineDesc.vertexLayout.AddAttribute(2, rhi::Format::R32G32Sfloat, 24);
    shadowPipelineDesc.vertexLayout.SetBindingStride(gbufferPipelineDesc.vertexLayout.GetStride());
    shadowPipelineDesc.depthTest = true;
    shadowPipelineDesc.depthWrite = true;
    shadowPipelineDesc.depthBiasEnable = true;
    shadowPipelineDesc.cullMode = rhi::CullMode::None;
    // The fragment stage alpha-tests masked materials through the bindless textures.
    shadowPipelineDesc.descriptorSetLayouts = {m_shadowSetLayout.get(), m_bindlessSetLayout.get()};
    shadowPipelineDesc.depthFormat = rhi::Format::D32Sfloat;
    m_shadowPipeline = device.CreatePipeline(shadowPipelineDesc);

    const auto createGiPipeline = [&device](
                                      const rhi::Shader& shader,
                                      const rhi::DescriptorSetLayout& layout,
                                      std::uint32_t pushConstantSize,
                                      const rhi::DescriptorSetLayout* secondLayout = nullptr) {
        rhi::PipelineDesc desc{};
        desc.computeShader = &shader;
        desc.descriptorSetLayouts = {&layout};
        if (secondLayout != nullptr)
        {
            desc.descriptorSetLayouts.push_back(secondLayout);
        }
        desc.pushConstantSize = pushConstantSize;
        desc.pushConstantStages = rhi::ShaderStageMask::Compute;
        return device.CreatePipeline(desc);
    };

    // The trace pass also samples the bindless textures and reads the bindless geometry buffers.
    m_giTracePipeline = createGiPipeline(*m_giTraceShader, *m_giTraceSetLayout, 0, m_bindlessSetLayout.get());
    m_giTemporalPipeline = createGiPipeline(*m_giTemporalShader, *m_giTemporalSetLayout, 0);
    m_giAtrousPipeline = createGiPipeline(*m_giAtrousShader, *m_giAtrousSetLayout, sizeof(GiAtrousPushConstants));
    m_giUpsamplePipeline = createGiPipeline(*m_giUpsampleShader, *m_giUpsampleSetLayout, 0);

    if (m_rtShadowSetLayout != nullptr)
    {
        m_rtShadowPipeline = createGiPipeline(*m_rtShadowShader, *m_rtShadowSetLayout, 0);
    }

    if (m_giProbeSetLayout != nullptr)
    {
        m_giProbePipeline =
            createGiPipeline(*m_giProbeShader, *m_giProbeSetLayout, 0, m_bindlessSetLayout.get());
    }

    if (m_reflectionTraceSetLayout != nullptr)
    {
        m_rtReflectionPipeline = createGiPipeline(
            *m_rtReflectionShader,
            *m_reflectionTraceSetLayout,
            0,
            m_bindlessSetLayout.get());
        m_reflectionTemporalPipeline =
            createGiPipeline(*m_reflectionTemporalShader, *m_reflectionTemporalSetLayout, 0);
        m_reflectionUpsamplePipeline =
            createGiPipeline(*m_reflectionUpsampleShader, *m_reflectionUpsampleSetLayout, 0);
    }

    if (m_cloudSetLayout != nullptr)
    {
        rhi::PipelineDesc cloudDesc{};
        cloudDesc.computeShader = m_cloudShader.get();
        cloudDesc.descriptorSetLayouts = {m_cloudSetLayout.get()};
        cloudDesc.pushConstantStages = rhi::ShaderStageMask::Compute;
        m_cloudPipeline = device.CreatePipeline(cloudDesc);

        rhi::PipelineDesc cloudUpsampleDesc{};
        cloudUpsampleDesc.computeShader = m_cloudUpsampleShader.get();
        cloudUpsampleDesc.descriptorSetLayouts = {m_cloudUpsampleSetLayout.get()};
        cloudUpsampleDesc.pushConstantStages = rhi::ShaderStageMask::Compute;
        m_cloudUpsamplePipeline = device.CreatePipeline(cloudUpsampleDesc);

        rhi::PipelineDesc cloudReprojectDesc{};
        cloudReprojectDesc.computeShader = m_cloudReprojectShader.get();
        cloudReprojectDesc.descriptorSetLayouts = {m_cloudReprojectSetLayout.get()};
        cloudReprojectDesc.pushConstantStages = rhi::ShaderStageMask::Compute;
        m_cloudReprojectPipeline = device.CreatePipeline(cloudReprojectDesc);

        rhi::PipelineDesc cloudGodRayDesc{};
        cloudGodRayDesc.computeShader = m_cloudGodRayShader.get();
        cloudGodRayDesc.descriptorSetLayouts = {m_cloudGodRaySetLayout.get()};
        cloudGodRayDesc.pushConstantStages = rhi::ShaderStageMask::Compute;
        m_cloudGodRayPipeline = device.CreatePipeline(cloudGodRayDesc);
    }

    const std::array<const rhi::Shader*, 2> debugLineStages{
        m_debugLineVertexShader.get(),
        m_debugLineFragmentShader.get()};
    rhi::PipelineDesc debugLinePipelineDesc{};
    debugLinePipelineDesc.vertexShader = m_debugLineVertexShader.get();
    debugLinePipelineDesc.fragmentShader = m_debugLineFragmentShader.get();
    rhi::ApplyShaderReflection(debugLinePipelineDesc, debugLineStages);
    debugLinePipelineDesc.topology = rhi::PrimitiveTopology::LineList;
    debugLinePipelineDesc.depthTest = false;
    debugLinePipelineDesc.depthWrite = false;
    debugLinePipelineDesc.blend = true;
    debugLinePipelineDesc.cullMode = rhi::CullMode::None;
    debugLinePipelineDesc.colorFormats = {device.GetSwapchainInfo().format};
    debugLinePipelineDesc.depthFormat = rhi::Format::D32Sfloat;
    m_debugLinePipeline = device.CreatePipeline(debugLinePipelineDesc);

    return m_gbufferPipeline != nullptr && m_transparentPipeline != nullptr &&
           m_lightingPipeline != nullptr && m_cloudPipeline != nullptr &&
           m_cloudUpsamplePipeline != nullptr && m_cloudReprojectPipeline != nullptr &&
           m_cloudGodRayPipeline != nullptr &&
           m_clusterCullingPipeline != nullptr && m_shadowPipeline != nullptr &&
           m_giTracePipeline != nullptr && m_giTemporalPipeline != nullptr &&
           m_giAtrousPipeline != nullptr && m_giUpsamplePipeline != nullptr &&
           m_debugLinePipeline != nullptr &&
           (m_rtShadowSetLayout == nullptr || m_rtShadowPipeline != nullptr) &&
           (m_giProbeSetLayout == nullptr || m_giProbePipeline != nullptr) &&
           (m_reflectionTraceSetLayout == nullptr ||
            (m_rtReflectionPipeline != nullptr && m_reflectionTemporalPipeline != nullptr &&
             m_reflectionUpsamplePipeline != nullptr));
}

bool DeferredRenderer::CreateScene()
{
    MRE_ASSERT(m_bufferManager != nullptr, "DeferredRenderer requires a buffer manager");

    m_planeVertexBuffer = &m_bufferManager->CreateVertexBuffer(kPlaneVertices, sizeof(kPlaneVertices));
    m_planeIndexBuffer = &m_bufferManager->CreateIndexBuffer(kPlaneIndices, sizeof(kPlaneIndices));

    DrawItem ground{};
    ground.vertexBuffer = m_planeVertexBuffer;
    ground.indexBuffer = m_planeIndexBuffer;
    ground.indexCount = static_cast<std::uint32_t>(std::size(kPlaneIndices));
    ground.firstIndex = 0;
    ground.pushConstants.model = glm::mat4{1.0f};
    ground.pushConstants.baseColorFactor = glm::vec4{0.35f, 0.35f, 0.38f, 1.0f};
    ground.pushConstants.params = glm::vec4{0.9f, 0.0f, 0.0f, 0.0f};
    const std::uint32_t whiteIndex = m_whiteTexture->GetBindlessIndex();
    const std::uint32_t defaultSamplerIndex =
        GetSamplerIndex(rhi::AddressMode::Repeat, rhi::AddressMode::Repeat);
    ground.pushConstants.textureIndices = glm::uvec2{whiteIndex, whiteIndex};
    // Material defaults: every texture index must reference a bound bindless slot.
    ground.specular = glm::vec4{1.0f, 1.0f, 1.0f, 1.0f};
    ground.specularTextureIndices = glm::uvec4{whiteIndex, whiteIndex, 0u, 0u};
    ground.specularSamplerIndices = glm::uvec4{defaultSamplerIndex, defaultSamplerIndex, 0u, 0u};
    ground.volume = glm::vec4{0.0f, 1.5f, 0.0f, 0.0f};
    ground.attenuation = glm::vec4{1.0f, 1.0f, 1.0f, 0.0f};
    ground.volumeTextureIndices = glm::uvec4{whiteIndex, whiteIndex, 0u, 0u};
    ground.volumeSamplerIndices = glm::uvec4{defaultSamplerIndex, defaultSamplerIndex, 0u, 0u};
    ground.alpha = glm::vec4{0.0f, 0.5f, 0.0f, 0.0f};
    ground.sheen = glm::vec4{0.0f};
    ground.sheenTextureIndices = glm::uvec4{whiteIndex, whiteIndex, 0u, 0u};
    ground.sheenSamplerIndices = glm::uvec4{defaultSamplerIndex, defaultSamplerIndex, 0u, 0u};
    ground.emissive = glm::vec4{0.0f};
    ground.emissiveTextureIndices = glm::uvec4{whiteIndex, 0u, 0u, 0u};
    ground.emissiveSamplerIndices = glm::uvec4{defaultSamplerIndex, 0u, 0u, 0u};
    ground.vertexBufferIndex = m_nextBindlessBuffer++;
    ground.indexBufferIndex = m_nextBindlessBuffer++;
    m_bindlessSet->BindStorageBuffer(2, *m_planeVertexBuffer, ground.vertexBufferIndex);
    m_bindlessSet->BindStorageBuffer(2, *m_planeIndexBuffer, ground.indexBufferIndex);

    if (m_device->IsRayTracingSupported() && m_planeVertexBuffer != nullptr && m_planeIndexBuffer != nullptr)
    {
        const rhi::BottomLevelAccelerationStructureDesc planeDesc{
            m_planeVertexBuffer,
            static_cast<std::uint32_t>(sizeof(Vertex)),
            static_cast<std::uint32_t>(std::size(kPlaneVertices)),
            m_planeIndexBuffer,
            static_cast<std::uint32_t>(std::size(kPlaneIndices)),
            true};
        m_planeBlas = m_device->CreateBottomLevelAccelerationStructure(planeDesc);
        ground.blas = m_planeBlas.get();
        m_blasDirty = true;
    }

    m_sceneDrawItems.push_back(ground);

    MRE_ASSERT(m_planeVertexBuffer != nullptr, "Failed to create the ground plane vertex buffer");
    return true;
}

void DeferredRenderer::CreateLightingSets(rhi::Device& device)
{
    MRE_ASSERT(m_lightingSetLayout != nullptr, "DeferredRenderer requires a lighting set layout");
    MRE_ASSERT(m_gbufferSampler != nullptr, "DeferredRenderer requires a G-buffer sampler");

    const std::uint32_t framesInFlight = device.GetFramesInFlight();
    m_lightingSets.clear();
    m_lightingSets.reserve(framesInFlight);

    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        std::unique_ptr<rhi::DescriptorSet> lightingSet = device.CreateDescriptorSet(*m_lightingSetLayout);
        lightingSet->BindBuffer(0, *m_uniformBuffers[frame]);
        lightingSet->BindTexture(1, m_gbuffer.GetAlbedo(), *m_gbufferSampler);
        lightingSet->BindTexture(2, m_gbuffer.GetNormal(), *m_gbufferSampler);
        lightingSet->BindTexture(3, m_gbuffer.GetPosition(), *m_gbufferSampler);
        lightingSet->BindBuffer(4, *m_lightBuffers[frame]);
        lightingSet->BindBuffer(5, *m_clusterCountBuffers[frame]);
        lightingSet->BindBuffer(6, *m_clusterIndexBuffers[frame]);
        lightingSet->BindBuffer(7, *m_shadowUniformBuffers[frame]);
        lightingSet->BindTexture(8, *m_shadowMap, 0u);
        lightingSet->BindSampler(9, *m_shadowSampler);
        lightingSet->BindTexture(10, *m_spotShadowMap, 0u);
        // Binding 11 is the descriptor array of per-light point shadow maps.
        for (std::uint32_t slot = 0;
             slot < kMaxPointShadowLights && slot < m_pointShadowMaps.size();
             ++slot)
        {
            lightingSet->BindTexture(11, *m_pointShadowMaps[slot], slot);
        }
        MRE_ASSERT(!m_gi.empty(), "GI resources must exist before creating the lighting sets");
        lightingSet->BindTexture(12, *m_gi[frame].indirect, 0u);
        lightingSet->BindTexture(
            13,
            m_rtSunShadowMask != nullptr ? *m_rtSunShadowMask : *m_whiteTexture,
            0u);
        lightingSet->BindStorageBuffer(
            14,
            m_giProbeBuffer != nullptr ? *m_giProbeBuffer : *m_giProbeDummyBuffer);
        lightingSet->BindTexture(
            15,
            frame < m_gi.size() && m_gi[frame].reflections != nullptr
                ? *m_gi[frame].reflections
                : *m_whiteTexture,
            0u);
        lightingSet->BindTexture(16, m_gbuffer.GetSpecularF0(), 0u);
        lightingSet->BindTexture(17, m_gbuffer.GetTransmission(), 0u);
        lightingSet->BindTexture(18, m_gbuffer.GetSheen(), 0u);
        lightingSet->BindBuffer(19, *m_objectDataBuffers[frame]);
        lightingSet->BindTexture(20, m_gbuffer.GetMotion(), 0u);
        lightingSet->BindTexture(21, *m_brdfLut, 0u);
        lightingSet->BindSampler(22, *m_brdfLutSampler);
        // Clouds: the sky branch samples the full-res scattering/transmittance texture.
        const bool cloudReady = IsCloudActive() && frame < m_cloudFull.size() &&
                                m_cloudFull[frame] != nullptr && m_cloudEmpty != nullptr &&
                                m_cloudSampler != nullptr;
        lightingSet->BindTexture(
            23,
            cloudReady ? *m_cloudFull[frame] : *m_cloudEmpty,
            0u);
        lightingSet->BindSampler(24, *m_cloudSampler);
        const bool godRayReady = cloudReady && frame < m_cloudGodRays.size() &&
                                 m_cloudGodRays[frame] != nullptr;
        lightingSet->BindTexture(
            25,
            godRayReady ? *m_cloudGodRays[frame] : *m_cloudEmpty,
            0u);
        m_lightingSets.push_back(std::move(lightingSet));
    }
}

bool DeferredRenderer::CreateShadowResources(rhi::Device& device)
{
    MRE_PROFILE_ZONE_N("CreateShadowResources");
    MRE_ASSERT(m_shadowSetLayout != nullptr, "DeferredRenderer requires a shadow set layout");

    const rhi::TextureDesc shadowMapDesc{
        kShadowResolution,
        kShadowResolution,
        rhi::Format::D32Sfloat,
        rhi::TextureUsage::DepthAttachment,
        nullptr,
        0,
        kShadowCascadeCount};
    m_shadowMap = device.CreateTexture(shadowMapDesc);
    if (m_shadowMap == nullptr)
    {
        MRE_CRITICAL("Failed to create the shadow map array");
        return false;
    }

    const rhi::TextureDesc spotShadowDesc{
        kSpotShadowResolution,
        kSpotShadowResolution,
        rhi::Format::D32Sfloat,
        rhi::TextureUsage::DepthAttachment,
        nullptr,
        0,
        kMaxSpotShadowLights};
    m_spotShadowMap = device.CreateTexture(spotShadowDesc);
    if (m_spotShadowMap == nullptr)
    {
        MRE_CRITICAL("Failed to create the spot shadow map array");
        return false;
    }

    // One depth array per point shadow slot: the six cube faces are layers, updated independently.
    m_pointShadowMaps.clear();
    m_pointShadowMaps.reserve(kMaxPointShadowLights);
    for (std::uint32_t slot = 0; slot < kMaxPointShadowLights; ++slot)
    {
        const rhi::TextureDesc pointShadowDesc{
            kPointShadowResolution,
            kPointShadowResolution,
            rhi::Format::D32Sfloat,
            rhi::TextureUsage::DepthAttachment,
            nullptr,
            0,
            kPointShadowFaceCount};
        m_pointShadowMaps.push_back(device.CreateTexture(pointShadowDesc));
        if (m_pointShadowMaps.back() == nullptr)
        {
            MRE_CRITICAL("Failed to create point shadow map {}", slot);
            return false;
        }
    }
    for (PointShadowSlotState& state : m_pointShadowSlotStates)
    {
        state = PointShadowSlotState{};
    }

    rhi::SamplerDesc samplerDesc{};
    samplerDesc.minFilter = rhi::Filter::Linear;
    samplerDesc.magFilter = rhi::Filter::Linear;
    samplerDesc.addressModeU = rhi::AddressMode::ClampToEdge;
    samplerDesc.addressModeV = rhi::AddressMode::ClampToEdge;
    samplerDesc.compareEnable = true;
    samplerDesc.compareOp = rhi::CompareOp::LessEqual;
    m_shadowSampler = device.CreateSampler(samplerDesc);
    if (m_shadowSampler == nullptr)
    {
        MRE_CRITICAL("Failed to create the shadow comparison sampler");
        return false;
    }

    const std::uint32_t framesInFlight = device.GetFramesInFlight();
    m_shadowUniformBuffers.clear();
    m_shadowSets.clear();
    m_shadowUniformBuffers.reserve(framesInFlight);
    m_shadowSets.reserve(framesInFlight);

    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        m_shadowUniformBuffers.push_back(
            device.CreateBuffer(rhi::BufferDesc{sizeof(ShadowUniform), rhi::BufferUsage::Uniform, true}));

        std::unique_ptr<rhi::DescriptorSet> shadowSet = device.CreateDescriptorSet(*m_shadowSetLayout);
        shadowSet->BindBuffer(0, *m_shadowUniformBuffers.back());
        m_shadowSets.push_back(std::move(shadowSet));
    }

    MRE_INFO(
        "Shadow maps created ({}x{} x{} cascades, {}x{} x{} spots, {}x{}x{} per point light)",
        kShadowResolution,
        kShadowResolution,
        kShadowCascadeCount,
        kSpotShadowResolution,
        kSpotShadowResolution,
        kMaxSpotShadowLights,
        kPointShadowResolution,
        kPointShadowResolution,
        kPointShadowFaceCount);
    return true;
}

bool DeferredRenderer::CreateClusterResources(rhi::Device& device)
{
    MRE_PROFILE_ZONE_N("CreateClusterResources");
    MRE_ASSERT(device.GetFramesInFlight() > 0, "Device requires frames in flight");
    MRE_ASSERT(m_clusterGrid.count > 0, "Cluster grid must be computed before creating cluster resources");

    m_clusterSets.clear();
    m_lightBuffers.clear();
    m_clusterUniformBuffers.clear();
    m_clusterCountBuffers.clear();
    m_clusterIndexBuffers.clear();

    const std::uint32_t framesInFlight = device.GetFramesInFlight();
    const std::uint64_t lightBufferSize = sizeof(GpuLight) * kMaxLights;
    const std::uint64_t countBufferSize = sizeof(std::uint32_t) * m_clusterGrid.count;
    const std::uint64_t indexBufferSize =
        sizeof(std::uint32_t) * m_clusterGrid.count * kMaxLightsPerCluster;

    m_zeroClusterCounts.assign(m_clusterGrid.count, 0u);

    m_lightBuffers.reserve(framesInFlight);
    m_clusterUniformBuffers.reserve(framesInFlight);
    m_clusterCountBuffers.reserve(framesInFlight);
    m_clusterIndexBuffers.reserve(framesInFlight);
    m_clusterSets.reserve(framesInFlight);

    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        m_lightBuffers.push_back(device.CreateBuffer(rhi::BufferDesc{lightBufferSize, rhi::BufferUsage::Storage, true}));
        m_clusterUniformBuffers.push_back(
            device.CreateBuffer(rhi::BufferDesc{sizeof(ClusterUniform), rhi::BufferUsage::Uniform, true}));
        m_clusterCountBuffers.push_back(
            device.CreateBuffer(rhi::BufferDesc{countBufferSize, rhi::BufferUsage::Storage, true}));
        m_clusterIndexBuffers.push_back(
            device.CreateBuffer(rhi::BufferDesc{indexBufferSize, rhi::BufferUsage::Storage, false}));

        if (m_lightBuffers.back() == nullptr || m_clusterUniformBuffers.back() == nullptr ||
            m_clusterCountBuffers.back() == nullptr || m_clusterIndexBuffers.back() == nullptr)
        {
            MRE_CRITICAL("Failed to create the clustered light buffers");
            return false;
        }

        std::unique_ptr<rhi::DescriptorSet> clusterSet = device.CreateDescriptorSet(*m_clusterSetLayout);
        clusterSet->BindBuffer(0, *m_clusterUniformBuffers.back());
        clusterSet->BindBuffer(1, *m_lightBuffers.back());
        clusterSet->BindBuffer(2, *m_clusterCountBuffers.back());
        clusterSet->BindBuffer(3, *m_clusterIndexBuffers.back());
        m_clusterSets.push_back(std::move(clusterSet));
    }

    MRE_INFO(
        "Cluster grid {}x{}x{} ({} clusters, up to {} lights per cluster)",
        m_clusterGrid.x,
        m_clusterGrid.y,
        m_clusterGrid.z,
        m_clusterGrid.count,
        kMaxLightsPerCluster);
    return true;
}

void DeferredRenderer::SetLights(std::span<const Light> lights)
{
    m_lights.assign(lights.begin(), lights.end());
    if (m_lights.size() > kMaxLights)
    {
        MRE_WARN(
            "Scene registered {} lights, but only the first {} are rendered",
            m_lights.size(),
            kMaxLights);
    }
}

std::uint32_t DeferredRenderer::GetLightCount() const
{
    return static_cast<std::uint32_t>(std::min(m_lights.size(), static_cast<std::size_t>(kMaxLights)));
}

void DeferredRenderer::SetGiProbes(std::span<const GiProbe> probes)
{
    m_giProbes.assign(probes.begin(), probes.end());

    // Probe markers double as the GI probe set; rebuilding resets their accumulated irradiance.
    const std::size_t probeCount = std::min<std::size_t>(m_giProbes.size(), kMaxGiProbes);
    bool changed = probeCount != m_giProbeData.size();
    for (std::size_t index = 0; !changed && index < probeCount; ++index)
    {
        changed = m_giProbeData[index].position.x != m_giProbes[index].position.x ||
                  m_giProbeData[index].position.y != m_giProbes[index].position.y ||
                  m_giProbeData[index].position.z != m_giProbes[index].position.z;
    }

    if (changed)
    {
        m_giProbeData.assign(probeCount, GpuGiProbe{});
        for (std::size_t index = 0; index < probeCount; ++index)
        {
            m_giProbeData[index].position = glm::vec4{m_giProbes[index].position, 1.0f};
        }
        m_giProbeDataDirty = true;
    }
}

void DeferredRenderer::UpdateUniforms(std::uint32_t frameIndex, const FrameView& view)
{
    MRE_ASSERT(frameIndex < m_uniformBuffers.size(), "Frame index out of range");
    MRE_ASSERT(frameIndex < m_lightBuffers.size(), "Frame index out of range");

    const float tanHalfFovY = std::abs(view.projection[1][1]) > 0.0f
        ? 1.0f / std::abs(view.projection[1][1])
        : std::tan(view.fovYRadians * 0.5f);

    m_spotShadowLights.clear();
    m_pointShadowLights.clear();

    m_gpuLights.clear();
    m_gpuLights.reserve(m_lights.size());
    for (const Light& light : m_lights)
    {
        if (m_gpuLights.size() >= kMaxLights)
        {
            break;
        }

        // Assign the limited spot/point shadow slots in submission order. The slot index is stored
        // in GpuLight::params.w (biased by one so zero means "no shadow").
        std::uint32_t shadowSlotPlusOne = 0;
        if (m_shadowSettings.enabled && light.castShadows)
        {
            if (light.type == LightType::Spot && m_spotShadowLights.size() < kMaxSpotShadowLights)
            {
                shadowSlotPlusOne = static_cast<std::uint32_t>(m_spotShadowLights.size()) + 1;
                m_spotShadowLights.push_back(light);
            }
            else if (light.type == LightType::Point && m_pointShadowLights.size() < kMaxPointShadowLights)
            {
                shadowSlotPlusOne = static_cast<std::uint32_t>(m_pointShadowLights.size()) + 1;
                m_pointShadowLights.push_back(light);
            }
        }

        GpuLight gpuLight = ToGpuLight(light);
        gpuLight.params.w = static_cast<float>(shadowSlotPlusOne);
        m_gpuLights.push_back(gpuLight);
    }

    // Ray traced GI evaluates punctual lights through its own capped buffer.
    if (m_giSettings.pointLights)
    {
        m_giLights = SelectGiLights(m_lights, std::min(m_giSettings.maxGiLights, kMaxGiLights));
    }
    else
    {
        m_giLights.clear();
    }

    m_sceneUniform.viewProjection = view.viewProjection;
    m_sceneUniform.previousViewProjection = view.previousViewProjection;
    m_sceneUniform.view = view.view;
    m_sceneUniform.inverseViewProjection = glm::inverse(view.viewProjection);
    m_sceneUniform.cameraPosition = glm::vec4{view.position, 1.0f};
    for (std::size_t band = 0; band < 9; ++band)
    {
        m_sceneUniform.skySh[band] = m_environmentSh.bands[band];
    }
    m_sceneUniform.skyParams = glm::vec4{
        m_environmentSettings.enabled ? 1.0f : 0.0f,
        m_environmentSettings.intensity,
        m_environmentSettings.sunDiscIntensity,
        0.0f};
    m_sceneUniform.skyZenith = glm::vec4{m_environmentSettings.zenithColor, 0.0f};
    m_sceneUniform.skyHorizon = glm::vec4{m_environmentSettings.horizonColor, 0.0f};
    m_sceneUniform.skyGround = glm::vec4{m_environmentSettings.groundColor, 0.0f};
    m_sceneUniform.skyMap = glm::vec4{
        static_cast<float>(m_environmentMapIndex),
        static_cast<float>(m_environmentMapSamplerIndex),
        m_environmentMapBound ? 1.0f : 0.0f,
        0.0f};
    m_sceneUniform.clusterParams = glm::vec4{
        view.nearPlane,
        view.farPlane,
        static_cast<float>(kClusterTileSize),
        static_cast<float>(kClusterZSliceCount)};
    m_sceneUniform.clusterGrid = glm::uvec4{
        m_clusterGrid.x,
        m_clusterGrid.y,
        m_clusterGrid.z,
        static_cast<std::uint32_t>(m_gpuLights.size())};
    m_sceneUniform.screenSize = glm::vec4{
        static_cast<float>(m_gbuffer.GetWidth()),
        static_cast<float>(m_gbuffer.GetHeight()),
        m_clusterDebugView ? 1.0f : 0.0f,
        IsPerPixelGiActive() ? 1.0f : 0.0f};
    // ambientColor.w carries the active GI probe count to the lighting pass (0 disables them).
    m_sceneUniform.ambientColor.w =
        IsProbeGiActive() ? static_cast<float>(m_giProbeData.size()) : 0.0f;
    m_sceneUniform.reflectionParams.x =
        IsReflectionActive() ? m_reflectionSettings.intensity : 0.0f;

    // Primary directional light, used by the transparent forward pass.
    const Light* sunLight = FindShadowLight();
    if (sunLight == nullptr)
    {
        for (const Light& light : m_lights)
        {
            if (light.type == LightType::Directional)
            {
                sunLight = &light;
                break;
            }
        }
    }
    if (sunLight != nullptr)
    {
        const float length = glm::length(sunLight->direction);
        const glm::vec3 toSun =
            length > 1e-4f ? -sunLight->direction / length : glm::vec3{0.0f, -1.0f, 0.0f};
        m_sceneUniform.sunDirection = glm::vec4{toSun, 0.0f};
        m_sceneUniform.sunColor = glm::vec4{sunLight->color * sunLight->intensity, 1.0f};
    }
    else
    {
        m_sceneUniform.sunDirection = glm::vec4{0.0f, -1.0f, 0.0f, 0.0f};
        m_sceneUniform.sunColor = glm::vec4{0.0f, 0.0f, 0.0f, 1.0f};
    }
    m_uniformBuffers[frameIndex]->Write(&m_sceneUniform, sizeof(SceneUniform));

    m_clusterUniform.inverseView = glm::inverse(view.view);
    m_clusterUniform.cameraParams = glm::vec4{view.nearPlane, view.farPlane, tanHalfFovY, view.aspectRatio};
    m_clusterUniform.clusterParams = glm::vec4{
        static_cast<float>(kClusterTileSize),
        static_cast<float>(kClusterZSliceCount),
        0.0f,
        0.0f};
    m_clusterUniform.grid = m_sceneUniform.clusterGrid;
    m_clusterUniform.screen = glm::uvec4{m_gbuffer.GetWidth(), m_gbuffer.GetHeight(), 0u, 0u};
    m_clusterUniformBuffers[frameIndex]->Write(&m_clusterUniform, sizeof(ClusterUniform));

    if (!m_gpuLights.empty())
    {
        m_lightBuffers[frameIndex]->Write(
            m_gpuLights.data(),
            m_gpuLights.size() * sizeof(GpuLight));
    }

    m_clusterCountBuffers[frameIndex]->Write(
        m_zeroClusterCounts.data(),
        m_zeroClusterCounts.size() * sizeof(std::uint32_t));

    UpdateShadowUniforms(frameIndex, view);
    UpdateGiUniform(frameIndex, view);
    UpdateCloudUniform(frameIndex, view);
    UpdateRtShadowUniform(frameIndex);
}

const Light* DeferredRenderer::FindShadowLight() const
{
    for (const Light& light : m_lights)
    {
        if (light.type == LightType::Directional && light.castShadows)
        {
            return &light;
        }
    }
    return nullptr;
}

void DeferredRenderer::UpdateShadowUniforms(std::uint32_t frameIndex, const FrameView& view)
{
    MRE_ASSERT(frameIndex < m_shadowUniformBuffers.size(), "Frame index out of range");

    m_shadowUniform = ShadowUniform{};

    if (m_shadowSettings.enabled)
    {
        const Light* shadowLight = FindShadowLight();
        if (shadowLight != nullptr)
        {
            const std::vector<ShadowCascade> cascades =
                ComputeShadowCascades(view, shadowLight->direction, m_shadowSettings);

            for (std::size_t index = 0; index < cascades.size() && index < kShadowCascadeCount; ++index)
            {
                m_shadowUniform.viewProjections[index] = cascades[index].lightViewProjection;
                if (index + 1 < cascades.size())
                {
                    m_shadowUniform.cascadeSplits[static_cast<glm::length_t>(index)] =
                        cascades[index].splitFar;
                }
            }

            m_shadowUniform.params = glm::vec4{
                m_shadowSettings.depthBias,
                m_shadowSettings.normalBias,
                cascades.empty() ? m_shadowSettings.distance : cascades.back().splitFar,
                1.0f};
        }

        for (std::size_t index = 0; index < m_spotShadowLights.size(); ++index)
        {
            m_shadowUniform.viewProjections[kSpotShadowOffset + index] =
                ComputeSpotShadowMatrix(m_spotShadowLights[index], kShadowNearPlane);
        }

        for (std::size_t index = 0; index < m_pointShadowLights.size(); ++index)
        {
            const std::array<glm::mat4, kPointShadowFaceCount> faces =
                ComputePointShadowMatrices(m_pointShadowLights[index], kShadowNearPlane);
            for (std::uint32_t face = 0; face < kPointShadowFaceCount; ++face)
            {
                m_shadowUniform.viewProjections[kPointShadowOffset + index * kPointShadowFaceCount + face] =
                    faces[face];
            }
        }
    }

    m_shadowUniform.rtParams.x = IsRtShadowActive() ? 1.0f : 0.0f;

    m_shadowUniformBuffers[frameIndex]->Write(&m_shadowUniform, sizeof(ShadowUniform));
}

bool DeferredRenderer::IsRtShadowActive() const
{
    return m_shadowSettings.rtSunShadows && m_shadowSettings.enabled &&
           m_device != nullptr && m_device->IsRayTracingSupported() &&
           m_rtShadowPipeline != nullptr && m_rtSunShadowMask != nullptr &&
           FindShadowLight() != nullptr;
}

bool DeferredRenderer::CreateRtShadowResources(rhi::Device& device)
{
    m_rtShadowUniformBuffers.clear();
    m_rtShadowSets.clear();
    m_rtSunShadowMask.reset();

    if (!device.IsRayTracingSupported() || m_rtShadowPipeline == nullptr)
    {
        return true;
    }

    m_rtSunShadowMask = device.CreateTexture(rhi::TextureDesc{
        m_gbuffer.GetWidth(),
        m_gbuffer.GetHeight(),
        rhi::Format::R16Sfloat,
        rhi::TextureUsage::Storage});
    if (m_rtSunShadowMask == nullptr)
    {
        MRE_CRITICAL("Failed to create the ray traced sun shadow mask");
        return false;
    }

    const std::uint32_t framesInFlight = device.GetFramesInFlight();
    m_rtShadowUniformBuffers.reserve(framesInFlight);
    m_rtShadowSets.reserve(framesInFlight);
    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        m_rtShadowUniformBuffers.push_back(device.CreateBuffer(
            rhi::BufferDesc{sizeof(RtShadowUniform), rhi::BufferUsage::Uniform, true}));

        std::unique_ptr<rhi::DescriptorSet> set = device.CreateDescriptorSet(*m_rtShadowSetLayout);
        set->BindBuffer(0, *m_rtShadowUniformBuffers.back());
        m_rtShadowSets.push_back(std::move(set));
    }

    m_lightingSetsDirty = true;
    MRE_INFO(
        "Ray traced sun shadow resources created ({}x{})",
        m_gbuffer.GetWidth(),
        m_gbuffer.GetHeight());
    return true;
}

void DeferredRenderer::UpdateRtShadowUniform(std::uint32_t frameIndex)
{
    if (m_rtShadowSets.size() <= frameIndex || !IsRtShadowActive())
    {
        return;
    }

    const Light* sun = FindShadowLight();
    if (sun == nullptr)
    {
        return;
    }

    const float length = glm::length(sun->direction);
    const glm::vec3 direction = length > 1e-4f ? -sun->direction / length : glm::vec3{0.0f, -1.0f, 0.0f};

    m_rtShadowUniform.screenSize = glm::vec4{
        static_cast<float>(m_gbuffer.GetWidth()),
        static_cast<float>(m_gbuffer.GetHeight()),
        0.0f,
        0.0f};
    m_rtShadowUniform.sunDirectionIntensity = glm::vec4{direction, sun->intensity};
    m_rtShadowUniform.params = glm::vec4{
        static_cast<float>(std::clamp(m_shadowSettings.rtShadowSamples, 1u, 16u)),
        m_shadowSettings.rtSunAngularRadiusDegrees * 3.14159265358979323846f / 180.0f,
        static_cast<float>(m_giFrameCounter),
        std::max(m_shadowSettings.normalBias, 0.0f)};

    m_rtShadowUniformBuffers[frameIndex]->Write(&m_rtShadowUniform, sizeof(RtShadowUniform));
}

void DeferredRenderer::RecordRtShadowPass(rhi::CommandList& commandList, std::uint32_t frameIndex)
{
    if (!IsRtShadowActive() || m_rtShadowSets.size() <= frameIndex)
    {
        return;
    }

    // Bind after the G-buffer pass so the position/normal textures are shader readable.
    m_rtShadowSets[frameIndex]->BindAccelerationStructure(1, *m_tlas[frameIndex]);
    m_rtShadowSets[frameIndex]->BindTexture(2, m_gbuffer.GetPosition(), 0u);
    m_rtShadowSets[frameIndex]->BindTexture(3, m_gbuffer.GetNormal(), 0u);
    m_rtShadowSets[frameIndex]->BindStorageImage(4, *m_rtSunShadowMask);

    commandList.BeginGpuZone("RtSunShadows");
    commandList.BindPipeline(*m_rtShadowPipeline);
    commandList.BindDescriptorSet(0, *m_rtShadowSets[frameIndex]);
    commandList.Dispatch(
        (m_gbuffer.GetWidth() + 7) / 8,
        (m_gbuffer.GetHeight() + 7) / 8,
        1);
    // The GI trace samples the mask in a later compute pass, so include compute in the barrier.
    commandList.ShaderMemoryBarrier(
        rhi::ShaderStageMask::Compute,
        rhi::ShaderStageMask::FragmentCompute);
    commandList.EndGpuZone();
}

bool DeferredRenderer::IsProbeGiActive() const
{
    return m_giSettings.enabled && m_giSettings.useProbes &&
           m_device != nullptr && m_device->IsRayTracingSupported() &&
           m_giProbePipeline != nullptr && !m_giProbeData.empty();
}

bool DeferredRenderer::IsPerPixelGiActive() const
{
    if (!IsGiActive())
    {
        return false;
    }

    // Probe mode replaces the per-pixel trace so indirect lighting is not counted twice.
    return !IsProbeGiActive();
}

bool DeferredRenderer::CreateGiProbeResources(rhi::Device& device)
{
    m_giProbeUniformBuffers.clear();
    m_giProbeSets.clear();
    m_giProbeDummyBuffer.reset();

    if (!device.IsRayTracingSupported() || m_giProbePipeline == nullptr)
    {
        return true;
    }

    const GpuGiProbe zeroProbe{};
    m_giProbeDummyBuffer = device.CreateBuffer(
        rhi::BufferDesc{sizeof(GpuGiProbe), rhi::BufferUsage::Storage, true},
        &zeroProbe,
        sizeof(GpuGiProbe));
    if (m_giProbeDummyBuffer == nullptr)
    {
        MRE_CRITICAL("Failed to create the dummy GI probe buffer");
        return false;
    }

    const std::uint32_t framesInFlight = device.GetFramesInFlight();
    m_giProbeUniformBuffers.reserve(framesInFlight);
    m_giProbeSets.reserve(framesInFlight);
    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        m_giProbeUniformBuffers.push_back(device.CreateBuffer(
            rhi::BufferDesc{sizeof(GiUniform), rhi::BufferUsage::Uniform, true}));

        std::unique_ptr<rhi::DescriptorSet> set = device.CreateDescriptorSet(*m_giProbeSetLayout);
        set->BindBuffer(0, *m_giProbeUniformBuffers.back());
        m_giProbeSets.push_back(std::move(set));
    }

    m_lightingSetsDirty = true;
    MRE_INFO("GI probe resources created (up to {} probes)", kMaxGiProbes);
    return true;
}

void DeferredRenderer::RecordGiProbes(rhi::CommandList& commandList, std::uint32_t frameIndex)
{
    if (!IsProbeGiActive() || m_giProbeSets.size() <= frameIndex ||
        m_tlas.empty() || m_tlas[frameIndex] == nullptr)
    {
        return;
    }

    const std::uint64_t probeBytes = m_giProbeData.size() * sizeof(GpuGiProbe);
    if (m_giProbeDataDirty)
    {
        if (m_giProbeBuffer == nullptr || m_giProbeBuffer->GetSize() < probeBytes)
        {
            m_giProbeBuffer = m_device->CreateBuffer(
                rhi::BufferDesc{probeBytes, rhi::BufferUsage::Storage, true});
            MRE_ASSERT(m_giProbeBuffer != nullptr, "Failed to create the GI probe buffer");
            m_lightingSetsDirty = true;
        }
        m_giProbeBuffer->Write(m_giProbeData.data(), probeBytes);
        m_giProbeDataDirty = false;
    }

    // Bind after the G-buffer pass so the depth texture is shader readable.
    rhi::DescriptorSet& set = *m_giProbeSets[frameIndex];
    set.BindAccelerationStructure(1, *m_tlas[frameIndex]);
    if (frameIndex < m_geometryDataBuffers.size() && m_geometryDataBuffers[frameIndex] != nullptr)
    {
        set.BindStorageBuffer(2, *m_geometryDataBuffers[frameIndex]);
    }
    set.BindTexture(3, m_gbuffer.GetDepth(), 0u);
    set.BindTexture(4, *m_whiteTexture, 0u);
    set.BindStorageBuffer(5, *m_giProbeBuffer);
    set.BindStorageBuffer(6, *m_giProbeBuffer);
    set.BindTexture(
        9,
        m_rtSunShadowMask != nullptr ? *m_rtSunShadowMask : *m_whiteTexture,
        0u);
    if (frameIndex < m_giLightBuffers.size() && m_giLightBuffers[frameIndex] != nullptr)
    {
        set.BindBuffer(13, *m_giLightBuffers[frameIndex]);
    }

    const std::uint32_t probeCount = static_cast<std::uint32_t>(m_giProbeData.size());
    commandList.BeginGpuZone("GiProbes");
    commandList.BindPipeline(*m_giProbePipeline);
    commandList.BindDescriptorSet(0, set);
    commandList.BindDescriptorSet(1, *m_bindlessSet);
    commandList.Dispatch((probeCount + 63) / 64, 1, 1);
    commandList.ShaderMemoryBarrier(rhi::ShaderStageMask::Compute, rhi::ShaderStageMask::Fragment);
    commandList.EndGpuZone();
}

void DeferredRenderer::BindReflectionDescriptors(std::uint32_t frameIndex)
{
    GiFrameResources& gi = m_gi[frameIndex];
    const std::uint32_t parity = gi.reflectionParity;

    // Trace.
    gi.reflectionTraceSet->BindBuffer(0, *gi.uniformBuffer);
    if (m_tlas[frameIndex] != nullptr)
    {
        gi.reflectionTraceSet->BindAccelerationStructure(1, *m_tlas[frameIndex]);
    }
    if (m_geometryDataBuffers[frameIndex] != nullptr)
    {
        gi.reflectionTraceSet->BindStorageBuffer(2, *m_geometryDataBuffers[frameIndex]);
    }
    gi.reflectionTraceSet->BindTexture(3, m_gbuffer.GetDepth(), 0u);
    gi.reflectionTraceSet->BindTexture(4, m_gbuffer.GetNormal(), 0u);
    gi.reflectionTraceSet->BindTexture(5, *m_blackTexture, 0u);
    gi.reflectionTraceSet->BindStorageImage(6, *gi.reflectionNoisy);
    gi.reflectionTraceSet->BindStorageImage(7, *gi.reflectionHitDistance);
    gi.reflectionTraceSet->BindTexture(8, m_gbuffer.GetAlbedo(), 0u);
    gi.reflectionTraceSet->BindTexture(
        9,
        m_rtSunShadowMask != nullptr ? *m_rtSunShadowMask : *m_whiteTexture,
        0u);
    gi.reflectionTraceSet->BindTexture(10, m_gbuffer.GetSpecularF0(), 0u);
    gi.reflectionTraceSet->BindTexture(11, m_gbuffer.GetTransmission(), 0u);
    gi.reflectionTraceSet->BindTexture(12, m_gbuffer.GetAttenuation(), 0u);
    if (frameIndex < m_giLightBuffers.size() && m_giLightBuffers[frameIndex] != nullptr)
    {
        gi.reflectionTraceSet->BindBuffer(13, *m_giLightBuffers[frameIndex]);
    }

    // Temporal: reads history[parity], writes history[parity ^ 1].
    gi.reflectionTemporalSet->BindBuffer(0, *gi.uniformBuffer);
    gi.reflectionTemporalSet->BindTexture(1, m_gbuffer.GetDepth(), 0u);
    gi.reflectionTemporalSet->BindTexture(2, m_gbuffer.GetNormal(), 0u);
    gi.reflectionTemporalSet->BindTexture(3, m_gbuffer.GetMotion(), 0u);
    gi.reflectionTemporalSet->BindTexture(4, *gi.reflectionNoisy, 0u);
    gi.reflectionTemporalSet->BindTexture(5, *gi.reflectionHistory[parity], 0u);
    gi.reflectionTemporalSet->BindTexture(6, *gi.reflectionMoments[parity], 0u);
    gi.reflectionTemporalSet->BindStorageImage(7, *gi.reflectionHistory[parity ^ 1]);
    gi.reflectionTemporalSet->BindStorageImage(8, *gi.reflectionMoments[parity ^ 1]);

    // A-trous: one set per iteration, ping-ponging between the temporal output and the filtered
    // texture; the schedule always ends on the filtered texture.
    const std::vector<GiAtrousStep> steps =
        BuildGiAtrousSchedule(m_giSettings.atrousIterations, kMaxAtrousIterations);
    for (std::uint32_t stepIndex = 0; stepIndex < steps.size(); ++stepIndex)
    {
        const auto textureFor = [&gi, parity](GiAtrousTexture which) -> rhi::Texture* {
            if (which == GiAtrousTexture::History)
            {
                return gi.reflectionHistory[parity ^ 1].get();
            }
            if (which == GiAtrousTexture::Noisy)
            {
                return gi.reflectionNoisy.get();
            }
            return gi.reflectionFiltered.get();
        };

        rhi::DescriptorSet& set = *gi.reflectionAtrousSets[stepIndex];
        set.BindBuffer(0, *gi.uniformBuffer);
        set.BindTexture(1, m_gbuffer.GetDepth(), 0u);
        set.BindTexture(2, m_gbuffer.GetNormal(), 0u);
        set.BindTexture(3, *gi.reflectionHitDistance, 0u);
        set.BindTexture(4, *textureFor(steps[stepIndex].input), 0u);
        set.BindTexture(5, *gi.reflectionMoments[parity ^ 1], 0u);
        set.BindStorageImage(6, *textureFor(steps[stepIndex].output));
    }

    // Upsample.
    gi.reflectionUpsampleSet->BindBuffer(0, *gi.uniformBuffer);
    gi.reflectionUpsampleSet->BindTexture(1, *gi.reflectionFiltered, 0u);
    gi.reflectionUpsampleSet->BindTexture(2, m_gbuffer.GetDepth(), 0u);
    gi.reflectionUpsampleSet->BindTexture(3, m_gbuffer.GetNormal(), 0u);
    gi.reflectionUpsampleSet->BindStorageImage(4, *gi.reflections);
}

void DeferredRenderer::RecordReflectionPasses(rhi::CommandList& commandList, std::uint32_t frameIndex)
{
    if (!IsReflectionActive() || m_gi.size() <= frameIndex)
    {
        return;
    }

    GiFrameResources& gi = m_gi[frameIndex];
    if (gi.reflectionNoisy == nullptr || m_tlas[frameIndex] == nullptr ||
        m_geometryDataBuffers[frameIndex] == nullptr)
    {
        return;
    }

    const std::uint32_t halfWidth = gi.reflectionNoisy->GetWidth();
    const std::uint32_t halfHeight = gi.reflectionNoisy->GetHeight();
    const std::uint32_t groupsX = (halfWidth + 7) / 8;
    const std::uint32_t groupsY = (halfHeight + 7) / 8;
    const std::uint32_t iterations =
        std::clamp(m_giSettings.atrousIterations, 1u, kMaxAtrousIterations);

    BindReflectionDescriptors(frameIndex);

    commandList.BeginGpuZone("Reflections");

    commandList.BindPipeline(*m_rtReflectionPipeline);
    commandList.BindDescriptorSet(0, *gi.reflectionTraceSet);
    commandList.BindDescriptorSet(1, *m_bindlessSet);
    commandList.Dispatch(groupsX, groupsY, 1);
    commandList.ShaderMemoryBarrier(rhi::ShaderStageMask::Compute, rhi::ShaderStageMask::Compute);

    commandList.BindPipeline(*m_reflectionTemporalPipeline);
    commandList.BindDescriptorSet(0, *gi.reflectionTemporalSet);
    commandList.Dispatch(groupsX, groupsY, 1);
    commandList.ShaderMemoryBarrier(rhi::ShaderStageMask::Compute, rhi::ShaderStageMask::Compute);

    commandList.BindPipeline(*m_giAtrousPipeline);
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
    {
        commandList.BindDescriptorSet(0, *gi.reflectionAtrousSets[iteration]);
        // Signal 2 marks the reflection signal so the shared a-trous kernel uses the reflection
        // trace resolution for its full-resolution G-buffer taps.
        const GiAtrousPushConstants pushConstants{2, iteration};
        commandList.SetPushConstants(&pushConstants, sizeof(pushConstants), rhi::ShaderStageMask::Compute);
        commandList.Dispatch(groupsX, groupsY, 1);
        commandList.ShaderMemoryBarrier(rhi::ShaderStageMask::Compute, rhi::ShaderStageMask::Compute);
    }

    commandList.BindPipeline(*m_reflectionUpsamplePipeline);
    commandList.BindDescriptorSet(0, *gi.reflectionUpsampleSet);
    commandList.Dispatch(
        (m_gbuffer.GetWidth() + 7) / 8,
        (m_gbuffer.GetHeight() + 7) / 8,
        1);
    commandList.ShaderMemoryBarrier(rhi::ShaderStageMask::Compute, rhi::ShaderStageMask::Fragment);

    gi.reflectionParity ^= 1;
    gi.reflectionValid = true;

    commandList.EndGpuZone();
}

std::uint32_t DeferredRenderer::GetSamplerIndex(rhi::AddressMode addressModeU, rhi::AddressMode addressModeV)
{
    MRE_ASSERT(m_device != nullptr, "DeferredRenderer requires a device");
    MRE_ASSERT(m_bindlessSet != nullptr, "DeferredRenderer requires a bindless descriptor set");

    const std::uint32_t key =
        (static_cast<std::uint32_t>(addressModeU) << 2) | static_cast<std::uint32_t>(addressModeV);

    const auto existing = m_samplerCache.find(key);
    if (existing != m_samplerCache.end())
    {
        return existing->second;
    }

    MRE_ASSERT(m_bindlessSamplers.size() < rhi::kMaxBindlessSamplers, "Bindless sampler limit reached");

    rhi::SamplerDesc desc{};
    desc.addressModeU = addressModeU;
    desc.addressModeV = addressModeV;

    std::unique_ptr<rhi::Sampler> sampler = m_device->CreateSampler(desc);
    MRE_ASSERT(sampler != nullptr, "Failed to create a bindless sampler");

    const std::uint32_t index = static_cast<std::uint32_t>(m_bindlessSamplers.size());
    m_bindlessSet->BindSampler(1, *sampler, index);
    m_samplerCache.emplace(key, index);
    m_bindlessSamplers.push_back(std::move(sampler));
    return index;
}

} // namespace mukki
