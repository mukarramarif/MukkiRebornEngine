#include "Renderer/Deferred/DeferredRenderer.h"

#include "Assets/GltfLoader.h"
#include "Assets/HdrLoader.h"
#include "Renderer/Clouds/CloudNoise.h"
#include "Core/JobSystem.h"
#include "Renderer/GlobalIllumination/GiAtrousSchedule.h"
#include "Renderer/Lighting/Subsurface.h"
#include "Renderer/Meshlets/MeshletBuilder.h"
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
#include <cstring>
#include <future>
#include <iterator>
#include <string>
#include <utility>

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
            // Morph delta buffers are read by the vertex shaders, so the bindless storage array
            // must also cover the vertex stage.
            rhi::ShaderStageMask::VertexFragmentComputeMesh,
            rhi::kMaxBindlessBuffers,
            true},
    };
    std::vector<const rhi::Shader*> sceneStages{
        m_gbufferVertexShader.get(),
        m_gbufferFragmentShader.get()};
    if (m_gbufferMeshShader != nullptr)
    {
        sceneStages.push_back(m_gbufferMeshShader.get());
    }
    if (m_meshletCullShader != nullptr)
    {
        // The cull dispatch reads the scene uniform, so set 0 must also cover the compute stage.
        sceneStages.push_back(m_meshletCullShader.get());
    }
    const std::array<const rhi::Shader*, 1> lightingStages{m_lightingFragmentShader.get()};
    const std::array<const rhi::Shader*, 1> clusterStages{m_clusterCullingShader.get()};
    std::vector<const rhi::Shader*> shadowStages{
        m_shadowVertexShader.get(),
        m_shadowFragmentShader.get()};
    if (m_shadowMeshShader != nullptr)
    {
        shadowStages.push_back(m_shadowMeshShader.get());
    }
    if (m_shadowCullShader != nullptr)
    {
        // The shadow cull dispatch reads the shadow uniform, so set 0 must cover compute too.
        shadowStages.push_back(m_shadowCullShader.get());
    }
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
    if (m_shadowCullShader != nullptr)
    {
        const std::array<const rhi::Shader*, 1> shadowCullStages{m_shadowCullShader.get()};
        m_shadowCullSetLayout =
            device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(shadowCullStages, 1));
    }
    if (m_gbufferMeshShader != nullptr)
    {
        const std::array<const rhi::Shader*, 1> meshStages{m_gbufferMeshShader.get()};
        m_meshSetLayout =
            device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(meshStages, 2));
    }
    if (m_meshletCullShader != nullptr)
    {
        const std::array<const rhi::Shader*, 1> cullStages{m_meshletCullShader.get()};
        m_meshletCullSetLayout =
            device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(cullStages, 1));
    }
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
    CreateJointPaletteResources(device);
    // Per draw item morph state and the per-frame current/previous weight regions.
    m_morphDataBuffers.clear();
    m_morphWeightBuffers.clear();
    m_morphDataBuffers.reserve(framesInFlight);
    m_morphWeightBuffers.reserve(framesInFlight);
    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        m_morphDataBuffers.push_back(device.CreateBuffer(rhi::BufferDesc{
            kMaxObjectData * sizeof(GpuMorphData),
            rhi::BufferUsage::Storage,
            true}));
        m_morphWeightBuffers.push_back(device.CreateBuffer(rhi::BufferDesc{
            2ull * kMaxObjectData * kMaxMorphTargets * sizeof(float),
            rhi::BufferUsage::Storage,
            true}));
        MRE_ASSERT(
            m_morphDataBuffers.back() != nullptr && m_morphWeightBuffers.back() != nullptr,
            "Failed to create the morph buffers");
    }
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
        if (frame < m_jointBuffers.size() && m_jointBuffers[frame] != nullptr)
        {
            sceneSet->BindBuffer(2, *m_jointBuffers[frame]);
        }
        if (frame < m_morphDataBuffers.size() && m_morphDataBuffers[frame] != nullptr)
        {
            sceneSet->BindBuffer(3, *m_morphDataBuffers[frame]);
        }
        if (frame < m_morphWeightBuffers.size() && m_morphWeightBuffers[frame] != nullptr)
        {
            sceneSet->BindBuffer(4, *m_morphWeightBuffers[frame]);
        }
        m_sceneSets.push_back(std::move(sceneSet));
    }

    CreateMeshletResources(device);

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

    m_meshletRenderingSupported = m_gbufferMeshShader != nullptr && m_shadowMeshShader != nullptr &&
                                  m_gbufferMeshPipeline != nullptr && m_shadowMeshPipeline != nullptr &&
                                  m_meshSetLayout != nullptr &&
                                  m_meshSets.size() == device.GetFramesInFlight() &&
                                  m_shadowMeshSets.size() == device.GetFramesInFlight();
    m_meshletRenderingEnabled = m_meshletRenderingSupported;
    if (std::getenv("MRE_MESHLETS_OFF") != nullptr)
    {
        m_meshletRenderingEnabled = false;
    }
    if (std::getenv("MRE_MESHLET_DEBUG") != nullptr)
    {
        SetMeshletDebugView(true);
    }
    if (m_meshletRenderingSupported)
    {
        MRE_INFO(
            "Meshlet rendering {} (mesh shader G-buffer and shadow passes, {}, debug view {})",
            m_meshletRenderingEnabled ? "enabled" : "disabled",
            m_meshletCullPipeline != nullptr ? "compute culling" : "in-shader culling",
            m_meshletDebugView ? "on" : "off");
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

    if (!CreateSssResources(device))
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
    m_shadowMeshPipeline.reset();
    m_gbufferMeshPipeline.reset();
    m_meshletCullPipeline.reset();
    m_meshletCullSetLayout.reset();
    m_shadowCullPipeline.reset();
    m_shadowCullSetLayout.reset();
    m_meshSetLayout.reset();
    m_meshDrawBuffers.clear();
    m_meshletIndexBuffers.clear();
    m_meshletCommandBuffers.clear();
    m_meshSets.clear();
    m_meshletCullSets.clear();
    m_shadowMeshletIndexBuffers.clear();
    m_shadowMeshletCommandBuffers.clear();
    m_shadowMeshSets.clear();
    m_shadowCullSets.clear();
    m_jointBuffers.clear();
    m_morphDataBuffers.clear();
    m_morphWeightBuffers.clear();
    m_meshletCount = 0;
    m_meshletTriangleCount = 0;
    m_meshletRenderingSupported = false;
    m_meshletRenderingEnabled = true;
    m_meshletCompactionActive = false;
    m_meshletDebugView = false;
    m_shadowMeshletCompactionActive = false;
    m_clusterCullingPipeline.reset();
    m_lightingPipeline.reset();
    m_gbufferPipeline.reset();
    m_shadowFragmentShader.reset();
    m_shadowMeshShader.reset();
    m_shadowVertexShader.reset();
    m_meshletCullShader.reset();
    m_shadowCullShader.reset();
    m_gbufferMeshShader.reset();
    m_clusterCullingShader.reset();
    m_lightingFragmentShader.reset();
    m_lightingVertexShader.reset();
    m_gbufferFragmentShader.reset();
    m_gbufferVertexShader.reset();
    m_sceneDrawItems.clear();
    m_groundDrawItemValid = false;
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

    if (!CreateSssResources(*m_device))
    {
        MRE_CRITICAL("Failed to recreate the subsurface scattering resources");
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

bool DeferredRenderer::LoadGeneratedModel(std::string name, GltfModel model)
{
    MRE_ASSERT(
        m_initialized,
        "DeferredRenderer::LoadGeneratedModel called before DeferredRenderer::Init");
    MRE_ASSERT(m_device != nullptr, "DeferredRenderer requires a device");

    if (model.primitives.empty())
    {
        MRE_WARN("Generated model '{}' has no primitives", name);
        return false;
    }

    // UnloadModel also cancels a pending asynchronous glTF load so it cannot replace the
    // generated geometry when it finishes.
    UnloadModel();

    const std::optional<std::size_t> modelIndex = UploadModel(
        std::filesystem::path{std::move(name)},
        model,
        false);
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

std::optional<std::size_t> DeferredRenderer::UploadModel(
    const std::filesystem::path& path,
    GltfModel& model,
    bool applyFit)
{
    MRE_PROFILE_ZONE_N("UploadModel");
    MRE_ASSERT(m_device != nullptr, "DeferredRenderer requires a device");

    auto resources = std::make_unique<ModelResources>();
    resources->name = PathToUtf8(path.filename());
    resources->vertexCount = model.vertexCount;
    resources->triangleCount = model.triangleCount;

    glm::mat4 fit{1.0f};
    glm::vec3 fittedMin = model.minBounds;
    glm::vec3 fittedMax = model.maxBounds;
    if (applyFit)
    {
        const glm::vec3 center = (model.minBounds + model.maxBounds) * 0.5f;
        const glm::vec3 extent = model.maxBounds - model.minBounds;
        const float maxExtent = std::max({extent.x, extent.y, extent.z});
        const float scale = maxExtent > 0.0f ? 2.5f / maxExtent : 1.0f;
        const float lift = (center.y - model.minBounds.y) * scale;

        fit = glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, lift, 0.0f}) *
              glm::scale(glm::mat4{1.0f}, glm::vec3{scale}) *
              glm::translate(glm::mat4{1.0f}, -center);

        fittedMin = glm::vec3{fit * glm::vec4{model.minBounds, 1.0f}};
        fittedMax = glm::vec3{fit * glm::vec4{model.maxBounds, 1.0f}};
    }

    resources->fitMatrix = fit;
    resources->fittedBounds.center = (fittedMin + fittedMax) * 0.5f;
    resources->fittedBounds.radius = glm::length(fittedMax - fittedMin) * 0.5f;
    resources->fittedMin = glm::min(fittedMin, fittedMax);
    resources->fittedMax = glm::max(fittedMin, fittedMax);

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
        // Diffuse transmission: the factor lives in the alpha channel, the color is sRGB.
        material.diffuseTransmissionTexture = createTextureFromImage(
            textureImageIndex(source.diffuseTransmissionTexture),
            rhi::Format::R8G8B8A8Unorm);
        material.diffuseTransmissionColorTexture = createTextureFromImage(
            textureImageIndex(source.diffuseTransmissionColorTexture),
            rhi::Format::R8G8B8A8Srgb);
        if (source.diffuseTransmissionFactor > 0.0f && !m_hasSssMaterials)
        {
            m_hasSssMaterials = true;
            MRE_INFO("Diffuse transmission material detected; screen space SSS enabled");
        }

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
        if (material.diffuseTransmissionTexture != nullptr)
        {
            m_bindlessSet->BindTexture(
                0,
                *material.diffuseTransmissionTexture,
                material.diffuseTransmissionTexture->GetBindlessIndex());
        }
        if (material.diffuseTransmissionColorTexture != nullptr)
        {
            m_bindlessSet->BindTexture(
                0,
                *material.diffuseTransmissionColorTexture,
                material.diffuseTransmissionColorTexture->GetBindlessIndex());
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
    const bool meshShaderSupported = m_device->IsMeshShaderSupported();
    resources->vertexBuffers.reserve(model.primitives.size());
    resources->indexBuffers.reserve(model.primitives.size());
    resources->accelerationStructures.reserve(model.primitives.size());
    resources->meshletBuffers.reserve(model.primitives.size());
    resources->meshletPrimitives.reserve(model.primitives.size());
    resources->morphBuffers.reserve(model.primitives.size());
    resources->morphBufferIndices.reserve(model.primitives.size());
    for (const GltfPrimitive& primitive : model.primitives)
    {
        const std::uint64_t vertexBytes = sizeof(GltfVertex) * primitive.vertices.size();
        const std::uint64_t indexBytes = sizeof(std::uint32_t) * primitive.indices.size();

        resources->vertexBuffers.push_back(m_device->CreateBuffer(
            rhi::BufferDesc{vertexBytes, rhi::BufferUsage::Vertex, false, rayTracingSupported, 0, true},
            primitive.vertices.data(),
            vertexBytes));
        resources->indexBuffers.push_back(m_device->CreateBuffer(
            rhi::BufferDesc{indexBytes, rhi::BufferUsage::Index, false, rayTracingSupported, 0, true},
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

        // Pack the meshlets, the vertex index list and the packed micro triangles into one
        // storage buffer that is registered in the bindless buffer array. Skinned primitives skip
        // the mesh shader path: their vertices deform per pose, so they render through the vertex
        // pipeline with GPU skinning. Morphed primitives do the same (deltas are applied in the
        // vertex shader before skinning).
        resources->meshletPrimitives.push_back(ModelResources::MeshletPrimitive{});
        if (meshShaderSupported && primitive.skinIndex < 0 && primitive.morphTargets.empty())
        {
            const MeshletMesh meshlets = BuildMeshlets(primitive.vertices, primitive.indices);
            if (!meshlets.Empty())
            {
                const std::uint32_t meshletBytes =
                    static_cast<std::uint32_t>(meshlets.meshlets.size() * sizeof(GpuMeshlet));
                const std::uint32_t vertexIndexBase = meshletBytes;
                const std::uint32_t triangleBase = vertexIndexBase +
                    static_cast<std::uint32_t>(meshlets.meshletVertices.size() * sizeof(std::uint32_t));
                const std::uint64_t bufferSize = triangleBase +
                    meshlets.meshletTriangles.size() * sizeof(std::uint32_t);

                std::vector<GpuMeshlet> gpuMeshlets(meshlets.meshlets.size());
                for (std::size_t index = 0; index < meshlets.meshlets.size(); ++index)
                {
                    const Meshlet& meshlet = meshlets.meshlets[index];
                    gpuMeshlets[index].centerRadius = glm::vec4{
                        meshlet.bounds.center,
                        meshlet.bounds.radius};
                    gpuMeshlets[index].coneAxisCutoff = glm::vec4{
                        meshlet.bounds.coneAxis,
                        meshlet.bounds.coneCutoff};
                    gpuMeshlets[index].data = glm::uvec4{
                        meshlet.vertexCount,
                        meshlet.triangleCount,
                        meshlet.vertexOffset,
                        meshlet.triangleOffset};
                }

                std::vector<std::byte> bytes(static_cast<std::size_t>(bufferSize));
                std::memcpy(bytes.data(), gpuMeshlets.data(), meshletBytes);
                std::memcpy(
                    bytes.data() + vertexIndexBase,
                    meshlets.meshletVertices.data(),
                    meshlets.meshletVertices.size() * sizeof(std::uint32_t));
                std::memcpy(
                    bytes.data() + triangleBase,
                    meshlets.meshletTriangles.data(),
                    meshlets.meshletTriangles.size() * sizeof(std::uint32_t));

                resources->meshletBuffers.push_back(m_device->CreateBuffer(
                    rhi::BufferDesc{bufferSize, rhi::BufferUsage::Storage, false},
                    bytes.data(),
                    bytes.size()));

                ModelResources::MeshletPrimitive& meshletPrimitive =
                    resources->meshletPrimitives.back();
                meshletPrimitive.meshletCount =
                    static_cast<std::uint32_t>(meshlets.meshlets.size());
                meshletPrimitive.vertexIndexBase = vertexIndexBase;
                meshletPrimitive.triangleBase = triangleBase;
                meshletPrimitive.bufferIndex = m_nextBindlessBuffer++;
                m_bindlessSet->BindStorageBuffer(
                    2,
                    *resources->meshletBuffers.back(),
                    meshletPrimitive.bufferIndex);
                m_meshletCount += meshlets.meshlets.size();
                m_meshletTriangleCount += meshlets.meshletTriangles.size();
            }
        }

        // Morph target deltas: [target * vertexCount + vertex] pairs of (position delta,
        // normal delta) float4s, registered in the bindless storage buffer array so the vertex
        // shaders can index them per draw item.
        resources->morphBuffers.push_back(nullptr);
        resources->morphBufferIndices.push_back(0u);
        if (!primitive.morphTargets.empty())
        {
            if (primitive.morphTargets.size() > kMaxMorphTargets)
            {
                MRE_WARN(
                    "Model '{}' primitive has {} morph targets (max {}); morphing disabled for it",
                    resources->name,
                    primitive.morphTargets.size(),
                    kMaxMorphTargets);
            }
            else
            {
                const std::size_t vertexCount = primitive.vertices.size();
                std::vector<glm::vec4> deltas(
                    primitive.morphTargets.size() * vertexCount * 2u,
                    glm::vec4{0.0f});
                for (std::size_t target = 0; target < primitive.morphTargets.size(); ++target)
                {
                    const GltfMorphTarget& morph = primitive.morphTargets[target];
                    for (std::size_t vertex = 0; vertex < vertexCount; ++vertex)
                    {
                        glm::vec3 positionDelta{0.0f};
                        glm::vec3 normalDelta{0.0f};
                        if (vertex < morph.positionDeltas.size())
                        {
                            positionDelta = morph.positionDeltas[vertex];
                        }
                        if (vertex < morph.normalDeltas.size())
                        {
                            normalDelta = morph.normalDeltas[vertex];
                        }
                        deltas[(target * vertexCount + vertex) * 2u + 0u] =
                            glm::vec4{positionDelta, 0.0f};
                        deltas[(target * vertexCount + vertex) * 2u + 1u] =
                            glm::vec4{normalDelta, 0.0f};
                    }
                }

                resources->morphBuffers.back() = m_device->CreateBuffer(
                    rhi::BufferDesc{
                        deltas.size() * sizeof(glm::vec4),
                        rhi::BufferUsage::Storage,
                        false},
                    deltas.data(),
                    deltas.size() * sizeof(glm::vec4));
                resources->morphBufferIndices.back() = m_nextBindlessBuffer++;
                m_bindlessSet->BindStorageBuffer(
                    2,
                    *resources->morphBuffers.back(),
                    resources->morphBufferIndices.back());
            }
        }
    }
    m_blasDirty = true;

    for (std::size_t nodeIndex = 0; nodeIndex < model.nodes.size(); ++nodeIndex)
    {
        const GltfNode& node = model.nodes[nodeIndex];
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
            item.nodeIndex = static_cast<int>(nodeIndex);
            item.skinIndex = primitive.skinIndex;
            item.baseNodeWorld = fit * node.worldTransform;
            item.pushConstants.model = item.baseNodeWorld;
            if (primitiveIndex < resources->accelerationStructures.size())
            {
                item.blas = resources->accelerationStructures[primitiveIndex].get();
            }
            item.vertexBufferIndex = resources->vertexBufferIndices[primitiveIndex];
            item.indexBufferIndex = resources->indexBufferIndices[primitiveIndex];
            const ModelResources::MeshletPrimitive& meshletPrimitive =
                resources->meshletPrimitives[primitiveIndex];
            item.meshletCount = meshletPrimitive.meshletCount;
            item.meshletOffset = 0;
            item.meshletVertexIndexBase = meshletPrimitive.vertexIndexBase;
            item.meshletTriangleBase = meshletPrimitive.triangleBase;
            item.meshletBufferIndex = meshletPrimitive.bufferIndex;
            item.morphBufferIndex = resources->morphBufferIndices[primitiveIndex];
            item.morphTargetCount =
                primitive.morphTargets.size() <= kMaxMorphTargets
                    ? static_cast<std::uint32_t>(primitive.morphTargets.size())
                    : 0u;
            item.morphVertexCount = static_cast<std::uint32_t>(primitive.vertices.size());

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
            item.subsurface = glm::vec4{0.0f};
            item.subsurfaceTextureIndices = glm::uvec4{whiteIndex, whiteIndex, 0u, 0u};
            item.subsurfaceSamplerIndices =
                glm::uvec4{defaultSamplerIndex, defaultSamplerIndex, 0u, 0u};

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
                item.subsurface = glm::vec4{
                    material.diffuseTransmissionColorFactor.r,
                    material.diffuseTransmissionColorFactor.g,
                    material.diffuseTransmissionColorFactor.b,
                    material.diffuseTransmissionFactor};

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

                if (gpuMaterial.diffuseTransmissionTexture != nullptr)
                {
                    item.subsurfaceTextureIndices.x =
                        gpuMaterial.diffuseTransmissionTexture->GetBindlessIndex();
                }
                if (gpuMaterial.diffuseTransmissionColorTexture != nullptr)
                {
                    item.subsurfaceTextureIndices.y =
                        gpuMaterial.diffuseTransmissionColorTexture->GetBindlessIndex();
                }
                if (material.diffuseTransmissionTexture >= 0 &&
                    material.diffuseTransmissionTexture < static_cast<int>(model.textures.size()))
                {
                    const GltfTexture& texture =
                        model.textures[static_cast<std::size_t>(material.diffuseTransmissionTexture)];
                    item.subsurfaceSamplerIndices.x =
                        GetSamplerIndex(ToRhiAddressMode(texture.wrapS), ToRhiAddressMode(texture.wrapT));
                }
                if (material.diffuseTransmissionColorTexture >= 0 &&
                    material.diffuseTransmissionColorTexture < static_cast<int>(model.textures.size()))
                {
                    const GltfTexture& texture =
                        model.textures[static_cast<std::size_t>(material.diffuseTransmissionColorTexture)];
                    item.subsurfaceSamplerIndices.y =
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

    resources->animationData = ExtractAnimationData(model);

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

    std::uint64_t meshletCount = 0;
    for (const ModelResources::MeshletPrimitive& primitive : m_models[modelIndex]->meshletPrimitives)
    {
        meshletCount += primitive.meshletCount;
    }

    MRE_INFO(
        "Model '{}' uploaded ({} draw items, {} textures, {} meshlets)",
        m_modelName,
        m_models[modelIndex]->drawItems.size(),
        textureCount,
        meshletCount);
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

    if (model.animationData != nullptr)
    {
        sceneObject.animation = std::make_unique<SceneObjectInstance::AnimationInstance>();
        sceneObject.animation->pose = RestPose(*model.animationData);
        sceneObject.animation->previousPose = sceneObject.animation->pose;
        sceneObject.animation->poseValid = true;
    }

    for (const DrawItem& source : model.drawItems)
    {
        DrawItem instance = source;
        const glm::mat4 baseTransform = transform * source.baseNodeWorld;
        instance.pushConstants.model = scaleMatrix * baseTransform;
        m_modelDrawItems.push_back(instance);
        m_modelBaseTransforms.push_back(baseTransform);
    }

    sceneObject.drawItemCount = m_modelDrawItems.size() - sceneObject.firstDrawItem;
    m_objectInstances.push_back(std::move(sceneObject));
    RebuildJointPaletteLayout();
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
    const AnimationData* animationData = GetAnimationData(sceneObject);
    for (std::size_t index = 0; index < sceneObject.drawItemCount; ++index)
    {
        const DrawItem& source = model.drawItems[index];
        const std::size_t drawIndex = sceneObject.firstDrawItem + index;
        glm::mat4 baseTransform = transform * source.baseNodeWorld;
        if (animationData != nullptr && sceneObject.animation != nullptr)
        {
            const PoseData& pose = sceneObject.animation->pose;
            if (source.skinIndex >= 0)
            {
                // Skinned meshes ignore the mesh node transform (glTF spec).
                baseTransform = transform;
            }
            else if (source.nodeIndex >= 0 &&
                     source.nodeIndex < static_cast<int>(pose.nodeWorld.size()))
            {
                baseTransform =
                    transform * model.fitMatrix * pose.nodeWorld[static_cast<std::size_t>(source.nodeIndex)];
            }
        }
        m_modelDrawItems[drawIndex].pushConstants.model = scaleMatrix * baseTransform;
        m_modelBaseTransforms[drawIndex] = baseTransform;
    }
    ++m_geometryRevision;
}

const AnimationData* DeferredRenderer::GetAnimationData(const SceneObjectInstance& instance) const
{
    if (instance.modelIndex >= m_models.size())
    {
        return nullptr;
    }
    return m_models[instance.modelIndex]->animationData.get();
}

void DeferredRenderer::RebuildJointPaletteLayout()
{
    std::uint32_t usage = 0;
    bool capacityExceeded = false;
    for (SceneObjectInstance& instance : m_objectInstances)
    {
        if (instance.animation == nullptr)
        {
            continue;
        }
        const AnimationData* data = GetAnimationData(instance);
        if (data == nullptr)
        {
            instance.animation.reset();
            continue;
        }

        std::uint32_t instanceUsage = 0;
        for (const GltfSkin& skin : data->skins)
        {
            instanceUsage += static_cast<std::uint32_t>(skin.joints.size());
        }
        if (usage + instanceUsage > kMaxJointMatrices)
        {
            capacityExceeded = true;
            instance.animation->jointPaletteOffset = 0;
            instance.animation->jointMatrixCount = 0;
        }
        else
        {
            instance.animation->jointPaletteOffset = usage;
            instance.animation->jointMatrixCount = instanceUsage;
            usage += instanceUsage;
        }
    }
    if (capacityExceeded)
    {
        MRE_WARN(
            "Joint palette capacity exceeded ({} matrices); skinning disabled for the overflow",
            kMaxJointMatrices);
    }

    for (const SceneObjectInstance& instance : m_objectInstances)
    {
        if (instance.animation == nullptr)
        {
            continue;
        }
        const AnimationData* data = GetAnimationData(instance);
        if (data == nullptr)
        {
            continue;
        }

        std::vector<std::uint32_t> skinOffsets(data->skins.size(), 0u);
        std::uint32_t offset = 0;
        for (std::size_t skin = 0; skin < data->skins.size(); ++skin)
        {
            skinOffsets[skin] = offset;
            offset += static_cast<std::uint32_t>(data->skins[skin].joints.size());
        }

        for (std::size_t index = 0; index < instance.drawItemCount; ++index)
        {
            DrawItem& draw = m_modelDrawItems[instance.firstDrawItem + index];
            draw.jointOffset = 0;
            draw.jointCount = 0;
            draw.previousJointOffset = 0;
            if (draw.skinIndex < 0 || instance.animation->jointMatrixCount == 0)
            {
                continue;
            }
            const std::size_t skin = static_cast<std::size_t>(draw.skinIndex);
            if (skin >= data->skins.size())
            {
                continue;
            }
            draw.jointOffset = instance.animation->jointPaletteOffset + skinOffsets[skin];
            draw.jointCount = static_cast<std::uint32_t>(data->skins[skin].joints.size());
            draw.previousJointOffset = kMaxJointMatrices + draw.jointOffset;
        }
    }
}

void DeferredRenderer::UploadJointPalettes(std::uint32_t frameIndex)
{
    if (frameIndex >= m_jointBuffers.size() || m_jointBuffers[frameIndex] == nullptr)
    {
        return;
    }

    rhi::Buffer& buffer = *m_jointBuffers[frameIndex];
    for (const SceneObjectInstance& instance : m_objectInstances)
    {
        if (instance.animation == nullptr || instance.animation->jointMatrixCount == 0)
        {
            continue;
        }
        const SceneObjectInstance::AnimationInstance& animation = *instance.animation;
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(animation.jointMatrixCount) * sizeof(glm::mat4);
        if (animation.pose.jointMatrices.size() >= animation.jointMatrixCount)
        {
            buffer.Write(
                animation.pose.jointMatrices.data(),
                bytes,
                static_cast<std::uint64_t>(animation.jointPaletteOffset) * sizeof(glm::mat4));
        }
        if (animation.previousPose.jointMatrices.size() >= animation.jointMatrixCount)
        {
            buffer.Write(
                animation.previousPose.jointMatrices.data(),
                bytes,
                static_cast<std::uint64_t>(kMaxJointMatrices + animation.jointPaletteOffset) *
                    sizeof(glm::mat4));
        }
    }
}

void DeferredRenderer::UploadMorphData(std::uint32_t frameIndex)
{
    if (frameIndex >= m_morphDataBuffers.size() || frameIndex >= m_morphWeightBuffers.size() ||
        m_morphDataBuffers[frameIndex] == nullptr || m_morphWeightBuffers[frameIndex] == nullptr)
    {
        return;
    }

    const std::size_t itemCount = m_sceneDrawItems.size() + m_modelDrawItems.size();
    if (itemCount > kMaxObjectData)
    {
        return;
    }

    std::vector<GpuMorphData> morphData(itemCount);
    std::vector<float> weights(2ull * itemCount * kMaxMorphTargets, 0.0f);

    // Scene draw items (the ground plane) never morph; model draw items start after them.
    std::size_t index = m_sceneDrawItems.size();
    for (std::size_t item = 0; item < m_modelDrawItems.size(); ++item, ++index)
    {
        const DrawItem& draw = m_modelDrawItems[item];
        if (draw.morphTargetCount == 0)
        {
            continue;
        }
        morphData[index].data = glm::uvec4{
            draw.morphBufferIndex,
            draw.morphTargetCount,
            draw.morphVertexCount,
            0u};
        morphData[index].offsets = glm::uvec4{
            static_cast<std::uint32_t>(index * kMaxMorphTargets),
            static_cast<std::uint32_t>((itemCount + index) * kMaxMorphTargets),
            0u,
            0u};

        // The owning instance drives the weights (defaults or the evaluated clip).
        const SceneObjectInstance* owner = nullptr;
        for (const SceneObjectInstance& instance : m_objectInstances)
        {
            if (item >= instance.firstDrawItem &&
                item < instance.firstDrawItem + instance.drawItemCount)
            {
                owner = &instance;
                break;
            }
        }
        if (owner == nullptr || owner->animation == nullptr)
        {
            continue;
        }

        const PoseData& pose = owner->animation->pose;
        const PoseData& previousPose = owner->animation->previousPose;
        const auto writeWeights = [&](const PoseData& source, std::size_t weightOffset) {
            if (draw.nodeIndex < 0 || draw.nodeIndex >= static_cast<int>(source.nodeWeights.size()))
            {
                return;
            }
            const std::vector<float>& nodeWeights =
                source.nodeWeights[static_cast<std::size_t>(draw.nodeIndex)];
            for (std::uint32_t target = 0;
                 target < draw.morphTargetCount && target < nodeWeights.size();
                 ++target)
            {
                weights[weightOffset + target] = nodeWeights[target];
            }
        };
        writeWeights(pose, morphData[index].offsets.x);
        writeWeights(previousPose, morphData[index].offsets.y);
    }

    m_morphDataBuffers[frameIndex]->Write(
        morphData.data(),
        morphData.size() * sizeof(GpuMorphData));
    m_morphWeightBuffers[frameIndex]->Write(weights.data(), weights.size() * sizeof(float));
}

void DeferredRenderer::UpdateAnimatedDrawTransforms()
{
    const glm::mat4 scaleMatrix = glm::scale(glm::mat4{1.0f}, glm::vec3{m_modelScale});
    for (const SceneObjectInstance& instance : m_objectInstances)
    {
        if (instance.animation == nullptr)
        {
            continue;
        }
        const AnimationData* data = GetAnimationData(instance);
        if (data == nullptr)
        {
            continue;
        }

        const PoseData& pose = instance.animation->pose;
        const ModelResources& model = *m_models[instance.modelIndex];
        for (std::size_t index = 0; index < instance.drawItemCount; ++index)
        {
            const std::size_t drawIndex = instance.firstDrawItem + index;
            DrawItem& draw = m_modelDrawItems[drawIndex];
            glm::mat4 baseTransform = instance.transform * draw.baseNodeWorld;
            if (draw.skinIndex >= 0)
            {
                // Skinned meshes ignore the mesh node transform (glTF spec).
                baseTransform = instance.transform;
            }
            else if (draw.nodeIndex >= 0 &&
                     draw.nodeIndex < static_cast<int>(pose.nodeWorld.size()))
            {
                baseTransform = instance.transform * model.fitMatrix *
                                pose.nodeWorld[static_cast<std::size_t>(draw.nodeIndex)];
            }
            draw.pushConstants.model = scaleMatrix * baseTransform;
            m_modelBaseTransforms[drawIndex] = baseTransform;
        }
    }
}

void DeferredRenderer::UpdateAnimations(float deltaSeconds)
{
    if (!m_initialized)
    {
        return;
    }

    bool anyChanged = false;
    for (SceneObjectInstance& instance : m_objectInstances)
    {
        SceneObjectInstance::AnimationInstance* animation = instance.animation.get();
        if (animation == nullptr)
        {
            continue;
        }
        const AnimationData* data = GetAnimationData(instance);
        if (data == nullptr)
        {
            continue;
        }

        bool changed = false;
        const bool hasClip = animation->clipIndex >= 0 &&
                             animation->clipIndex < static_cast<int>(data->animations.size());
        if (hasClip && animation->playing)
        {
            const GltfAnimation& clip =
                data->animations[static_cast<std::size_t>(animation->clipIndex)];
            animation->time += deltaSeconds * animation->speed;
            animation->time = WrapClipTime(clip.duration, animation->time, animation->loop);
            changed = true;
        }

        if (changed || !animation->poseValid)
        {
            PoseData pose = RestPose(*data);
            if (hasClip)
            {
                // The rest pose covers clips with no usable channels (EvaluatePose returns false).
                static_cast<void>(EvaluatePose(
                    *data,
                    data->animations[static_cast<std::size_t>(animation->clipIndex)],
                    animation->time,
                    pose));
            }
            animation->previousPose = std::move(animation->pose);
            animation->pose = std::move(pose);
            animation->poseValid = true;
            changed = true;
        }

        anyChanged = anyChanged || changed;
    }

    if (anyChanged)
    {
        UpdateAnimatedDrawTransforms();
        // Cached point shadow maps key off the geometry revision; a changed pose moves skinned
        // geometry, so refresh them for this frame.
        ++m_geometryRevision;
    }
}

bool DeferredRenderer::HasAnimation(std::size_t objectIndex) const
{
    if (objectIndex >= m_objectInstances.size())
    {
        return false;
    }
    return GetAnimationData(m_objectInstances[objectIndex]) != nullptr;
}

std::vector<std::string> DeferredRenderer::GetAnimationClips(std::size_t objectIndex) const
{
    std::vector<std::string> clips{};
    if (objectIndex >= m_objectInstances.size())
    {
        return clips;
    }
    const AnimationData* data = GetAnimationData(m_objectInstances[objectIndex]);
    if (data == nullptr)
    {
        return clips;
    }
    clips.reserve(data->animations.size());
    for (const GltfAnimation& clip : data->animations)
    {
        clips.push_back(clip.name);
    }
    return clips;
}

const std::string& DeferredRenderer::GetAnimationClipName(std::size_t objectIndex) const
{
    static const std::string kEmpty{};
    if (objectIndex >= m_objectInstances.size() ||
        m_objectInstances[objectIndex].animation == nullptr)
    {
        return kEmpty;
    }
    return m_objectInstances[objectIndex].animation->clipName;
}

bool DeferredRenderer::IsAnimationPlaying(std::size_t objectIndex) const
{
    if (objectIndex >= m_objectInstances.size() ||
        m_objectInstances[objectIndex].animation == nullptr)
    {
        return false;
    }
    return m_objectInstances[objectIndex].animation->playing;
}

float DeferredRenderer::GetAnimationTime(std::size_t objectIndex) const
{
    if (objectIndex >= m_objectInstances.size() ||
        m_objectInstances[objectIndex].animation == nullptr)
    {
        return 0.0f;
    }
    return m_objectInstances[objectIndex].animation->time;
}

float DeferredRenderer::GetAnimationDuration(std::size_t objectIndex) const
{
    if (objectIndex >= m_objectInstances.size() ||
        m_objectInstances[objectIndex].animation == nullptr)
    {
        return 0.0f;
    }
    const SceneObjectInstance& instance = m_objectInstances[objectIndex];
    const AnimationData* data = GetAnimationData(instance);
    const SceneObjectInstance::AnimationInstance& animation = *instance.animation;
    if (data == nullptr || animation.clipIndex < 0 ||
        animation.clipIndex >= static_cast<int>(data->animations.size()))
    {
        return 0.0f;
    }
    return data->animations[static_cast<std::size_t>(animation.clipIndex)].duration;
}

float DeferredRenderer::GetAnimationSpeed(std::size_t objectIndex) const
{
    if (objectIndex >= m_objectInstances.size() ||
        m_objectInstances[objectIndex].animation == nullptr)
    {
        return 1.0f;
    }
    return m_objectInstances[objectIndex].animation->speed;
}

void DeferredRenderer::SetAnimationClip(std::size_t objectIndex, std::string_view clipName)
{
    if (objectIndex >= m_objectInstances.size())
    {
        return;
    }
    SceneObjectInstance& instance = m_objectInstances[objectIndex];
    SceneObjectInstance::AnimationInstance* animation = instance.animation.get();
    const AnimationData* data = GetAnimationData(instance);
    if (animation == nullptr || data == nullptr)
    {
        return;
    }

    int clipIndex = -1;
    for (std::size_t index = 0; index < data->animations.size(); ++index)
    {
        if (data->animations[index].name == clipName)
        {
            clipIndex = static_cast<int>(index);
            break;
        }
    }

    animation->clipIndex = clipIndex;
    animation->clipName = clipIndex >= 0 ? data->animations[static_cast<std::size_t>(clipIndex)].name
                                         : std::string{};
    animation->time = 0.0f;
    animation->playing = clipIndex >= 0;
    animation->poseValid = false;
    UpdateAnimations(0.0f);
}

void DeferredRenderer::SetAnimationPlaying(std::size_t objectIndex, bool playing)
{
    if (objectIndex >= m_objectInstances.size() ||
        m_objectInstances[objectIndex].animation == nullptr)
    {
        return;
    }
    m_objectInstances[objectIndex].animation->playing = playing;
}

void DeferredRenderer::SetAnimationTime(std::size_t objectIndex, float seconds)
{
    if (objectIndex >= m_objectInstances.size() ||
        m_objectInstances[objectIndex].animation == nullptr)
    {
        return;
    }
    SceneObjectInstance& instance = m_objectInstances[objectIndex];
    SceneObjectInstance::AnimationInstance& animation = *instance.animation;
    const AnimationData* data = GetAnimationData(instance);
    if (data == nullptr || animation.clipIndex < 0 ||
        animation.clipIndex >= static_cast<int>(data->animations.size()))
    {
        return;
    }
    const float duration =
        data->animations[static_cast<std::size_t>(animation.clipIndex)].duration;
    animation.time = WrapClipTime(duration, seconds, animation.loop);
    animation.poseValid = false;
    UpdateAnimations(0.0f);
}

void DeferredRenderer::SetAnimationSpeed(std::size_t objectIndex, float speed)
{
    if (objectIndex >= m_objectInstances.size() ||
        m_objectInstances[objectIndex].animation == nullptr)
    {
        return;
    }
    m_objectInstances[objectIndex].animation->speed = std::max(speed, 0.0f);
}

void DeferredRenderer::SetAnimationLoop(std::size_t objectIndex, bool loop)
{
    if (objectIndex >= m_objectInstances.size() ||
        m_objectInstances[objectIndex].animation == nullptr)
    {
        return;
    }
    m_objectInstances[objectIndex].animation->loop = loop;
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

bool DeferredRenderer::GetSceneObjectFittedBounds(
    std::size_t objectIndex,
    glm::vec3& outMin,
    glm::vec3& outMax) const
{
    if (objectIndex >= m_objectInstances.size())
    {
        return false;
    }

    const SceneObjectInstance& sceneObject = m_objectInstances[objectIndex];
    MRE_ASSERT(sceneObject.modelIndex < m_models.size(), "Scene object model index out of range");
    const ModelResources& model = *m_models[sceneObject.modelIndex];
    const glm::mat4 world =
        glm::scale(glm::mat4{1.0f}, glm::vec3{m_modelScale}) * sceneObject.transform;

    glm::vec3 boundsMin{std::numeric_limits<float>::max()};
    glm::vec3 boundsMax{std::numeric_limits<float>::lowest()};
    for (int corner = 0; corner < 8; ++corner)
    {
        const glm::vec3 local{
            (corner & 1) != 0 ? model.fittedMax.x : model.fittedMin.x,
            (corner & 2) != 0 ? model.fittedMax.y : model.fittedMin.y,
            (corner & 4) != 0 ? model.fittedMax.z : model.fittedMin.z};
        const glm::vec3 worldCorner = glm::vec3{world * glm::vec4{local, 1.0f}};
        boundsMin = glm::min(boundsMin, worldCorner);
        boundsMax = glm::max(boundsMax, worldCorner);
    }

    outMin = boundsMin;
    outMax = boundsMax;
    // A degenerate (for example planar) model still needs a usable collider bound.
    const glm::vec3 minimumExtent{0.01f};
    const glm::vec3 center = (outMin + outMax) * 0.5f;
    const glm::vec3 halfExtents = glm::max((outMax - outMin) * 0.5f, minimumExtent);
    outMin = center - halfExtents;
    outMax = center + halfExtents;
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
    m_hasSssMaterials = false;
    m_meshletCount = 0;
    m_meshletTriangleCount = 0;
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

void DeferredRenderer::SetMeshletRenderingEnabled(bool enabled)
{
    m_meshletRenderingEnabled = enabled && m_meshletRenderingSupported;
    if (!m_meshletRenderingEnabled)
    {
        m_meshletDebugView = false;
    }
}

void DeferredRenderer::SetMeshletDebugView(bool enabled)
{
    m_meshletDebugView = enabled && m_meshletRenderingSupported && m_meshletRenderingEnabled;
}

void DeferredRenderer::SetModelScale(float scale)
{
    MRE_ASSERT(scale > 0.0f, "Model scale must be greater than zero");
    m_modelScale = std::clamp(scale, 0.01f, 100.0f);
    ApplyModelScale();
    ++m_geometryRevision;
}

void DeferredRenderer::SetGroundPlaneExtent(float halfExtent)
{
    MRE_ASSERT(halfExtent > 0.0f, "Ground plane extent must be greater than zero");
    m_groundPlaneExtent = std::clamp(halfExtent, 1.0f, 100000.0f);
    if (!m_groundDrawItemValid)
    {
        return;
    }

    m_groundDrawItem.pushConstants.model =
        glm::scale(glm::mat4{1.0f}, glm::vec3{m_groundPlaneExtent / 10.0f});
    if (m_groundVisible && !m_sceneDrawItems.empty())
    {
        m_sceneDrawItems.front().pushConstants.model = m_groundDrawItem.pushConstants.model;
    }
    ++m_geometryRevision;
}

void DeferredRenderer::SetGroundVisible(bool visible)
{
    if (visible == m_groundVisible)
    {
        return;
    }
    m_groundVisible = visible;
    if (!m_groundDrawItemValid)
    {
        return;
    }

    if (visible)
    {
        m_sceneDrawItems.insert(m_sceneDrawItems.begin(), m_groundDrawItem);
    }
    else if (!m_sceneDrawItems.empty())
    {
        m_sceneDrawItems.erase(m_sceneDrawItems.begin());
    }
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
    std::vector<GpuMeshDraw> meshDraws{};
    meshDraws.resize(itemCount);

    std::size_t index = 0;
    const auto updateItem = [this, &objectData, &meshDraws, &index](DrawItem& item) {
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
        objectData[index].subsurface = item.subsurface;
        objectData[index].subsurfaceTextureIndices = item.subsurfaceTextureIndices;
        objectData[index].subsurfaceSamplerIndices = item.subsurfaceSamplerIndices;
        objectData[index].skinning = glm::uvec4{
            item.jointOffset,
            item.previousJointOffset,
            item.jointCount,
            item.jointCount > 0 ? 1u : 0u};
        item.pushConstants.drawIndex = static_cast<std::uint32_t>(index);
        meshDraws[index].meshletOffset = item.meshletOffset;
        meshDraws[index].meshletCount = item.meshletCount;
        meshDraws[index].vertexIndexBase = item.meshletVertexIndexBase;
        meshDraws[index].triangleBase = item.meshletTriangleBase;
        meshDraws[index].vertexBufferIndex = item.vertexBufferIndex;
        meshDraws[index].meshletBufferIndex = item.meshletBufferIndex;
        meshDraws[index].compactedOffset = item.compactedOffset;
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

    // Assign the per-frame compacted meshlet slices and rearm the indirect mesh-task commands. The
    // cull shader accumulates groupCountX per draw; draws that do not fit the compaction capacity
    // fall back to direct dispatch and cull in the mesh shader.
    m_meshletCompactionActive =
        m_meshletCullPipeline != nullptr && frameIndex < m_meshletIndexBuffers.size() &&
        frameIndex < m_meshletCommandBuffers.size() && frameIndex < m_meshletCullSets.size() &&
        m_meshletIndexBuffers[frameIndex] != nullptr &&
        m_meshletCommandBuffers[frameIndex] != nullptr &&
        m_meshletCullSets[frameIndex] != nullptr;
    std::uint32_t compactedOffset = 0;
    if (m_meshletCompactionActive)
    {
        for (GpuMeshDraw& draw : meshDraws)
        {
            if (draw.meshletCount == 0)
            {
                draw.compactedOffset = kMeshletCullDisabled;
                continue;
            }
            draw.compactedOffset = compactedOffset;
            compactedOffset += draw.meshletCount;
            if (compactedOffset > kMeshletCompactionCapacity)
            {
                m_meshletCompactionActive = false;
                break;
            }
        }
    }

    if (m_meshletCompactionActive && !meshDraws.empty())
    {
        // groupCountY/Z stay 1; only groupCountX is accumulated by the cull dispatch.
        std::vector<std::uint32_t> commands(meshDraws.size() * 3u, 1u);
        for (std::size_t draw = 0; draw < meshDraws.size(); ++draw)
        {
            commands[draw * 3u] = 0u;
        }
        m_meshletCommandBuffers[frameIndex]->Write(
            commands.data(),
            commands.size() * sizeof(std::uint32_t));
    }
    else
    {
        for (GpuMeshDraw& draw : meshDraws)
        {
            draw.compactedOffset = kMeshletCullDisabled;
        }
    }

    index = 0;
    for (DrawItem& item : m_sceneDrawItems)
    {
        item.compactedOffset = meshDraws[index].compactedOffset;
        ++index;
    }
    for (DrawItem& item : m_modelDrawItems)
    {
        item.compactedOffset = meshDraws[index].compactedOffset;
        ++index;
    }

    if (!objectData.empty())
    {
        m_objectDataBuffers[frameIndex]->Write(objectData.data(), objectData.size() * sizeof(GpuObjectData));
    }

    if (!meshDraws.empty() && frameIndex < m_meshDrawBuffers.size() &&
        m_meshDrawBuffers[frameIndex] != nullptr)
    {
        m_meshDrawBuffers[frameIndex]->Write(meshDraws.data(), meshDraws.size() * sizeof(GpuMeshDraw));
    }

    UploadJointPalettes(frameIndex);
    UploadMorphData(frameIndex);
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
        geometry.subsurface = item.subsurface;
        geometry.subsurfaceTextureIndices = item.subsurfaceTextureIndices;
        geometry.subsurfaceSamplerIndices = item.subsurfaceSamplerIndices;
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

    const bool meshRenderingActive = m_meshletRenderingEnabled && m_meshletRenderingSupported &&
                                     m_gbufferMeshPipeline != nullptr &&
                                     frameIndex < m_meshSets.size() &&
                                     frameIndex < m_meshDrawBuffers.size();
    const bool meshCompactionActive =
        meshRenderingActive && m_meshletCompactionActive && m_meshletCullPipeline != nullptr &&
        frameIndex < m_meshletCommandBuffers.size() &&
        m_meshletCommandBuffers[frameIndex] != nullptr;
    // Alpha blended materials skip the opaque G-buffer and are drawn back-to-front after lighting.
    // Opaque items with meshlets use the mesh shader pipeline when it is available.
    std::vector<const DrawItem*> opaqueItems{};
    std::vector<const DrawItem*> transparentItems{};
    opaqueItems.reserve(m_sceneDrawItems.size() + m_modelDrawItems.size());
    transparentItems.reserve(m_modelDrawItems.size());
    const auto isMeshItem = [meshRenderingActive](const DrawItem& item) {
        return meshRenderingActive && item.meshletCount > 0;
    };
    const auto classifyItem = [&opaqueItems, &transparentItems](const DrawItem& item) {
        if (item.alphaBlend)
        {
            transparentItems.push_back(&item);
        }
        else
        {
            opaqueItems.push_back(&item);
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
    BuildShadowMeshletCull(frameIndex);

    const bool meshShadowActive =
        meshRenderingActive && m_shadowMeshPipeline != nullptr &&
        frameIndex < m_shadowMeshSets.size() && !m_shadowViews.empty();
    const bool shadowCompactionActive = meshShadowActive && m_shadowMeshletCompactionActive;
    if (shadowCompactionActive)
    {
        RecordShadowMeshletCullPass(commandList, frameIndex);
    }

    if (!m_shadowViews.empty())
    {
        commandList.BeginGpuZone("ShadowPass");

        const bool useShadowSecondaries =
            m_jobSystem != nullptr && frameIndex < m_shadowSecondaries.size() &&
            m_shadowViews.size() <= m_shadowSecondaries[frameIndex].size();

        if (useShadowSecondaries)
        {
            RecordShadowViews(frameIndex, meshShadowActive, shadowCompactionActive);

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

                ShadowPushConstants pushConstants{};
                pushConstants.viewIndex = shadowView.viewIndex;

                if (meshShadowActive)
                {
                    commandList.BindPipeline(*m_shadowMeshPipeline);
                    commandList.BindDescriptorSet(0, *m_shadowSets[frameIndex]);
                    commandList.BindDescriptorSet(1, *m_bindlessSet);
                    commandList.BindDescriptorSet(2, *m_shadowMeshSets[frameIndex]);
                    commandList.SetDepthBias(
                        m_shadowSettings.depthBias,
                        m_shadowSettings.depthBias * 1.5f,
                        0.0f);

                    const std::uint32_t meshPushConstantSize =
                        m_shadowMeshPipeline->GetPushConstantSize();
                    for (const ShadowMeshDraw& meshDraw : shadowView.meshDraws)
                    {
                        const DrawItem& item = *meshDraw.item;
                        pushConstants.object = item.pushConstants;
                        pushConstants.compactedOffset = shadowCompactionActive
                                                            ? meshDraw.compactedOffset
                                                            : kMeshletCullDisabled;
                        commandList.SetPushConstants(
                            &pushConstants,
                            meshPushConstantSize,
                            rhi::ShaderStageMask::FragmentMesh);
                        if (shadowCompactionActive &&
                            meshDraw.compactedOffset != kMeshletCullDisabled)
                        {
                            commandList.DrawMeshTasksIndirect(
                                *m_shadowMeshletCommandBuffers[frameIndex],
                                static_cast<std::uint64_t>(meshDraw.commandIndex) *
                                    sizeof(GpuMeshTaskCommand),
                                1u,
                                sizeof(GpuMeshTaskCommand));
                        }
                        else
                        {
                            commandList.DrawMeshTasks(item.meshletCount);
                        }
                    }
                }

                commandList.BindPipeline(*m_shadowPipeline);
                commandList.BindDescriptorSet(0, *m_shadowSets[frameIndex]);
                commandList.BindDescriptorSet(1, *m_bindlessSet);
                commandList.SetDepthBias(
                    m_shadowSettings.depthBias,
                    m_shadowSettings.depthBias * 1.5f,
                    0.0f);

                const std::uint32_t pushConstantSize = m_shadowPipeline->GetPushConstantSize();
                for (const DrawItem* item : shadowView.items)
                {
                    if (meshShadowActive && item->meshletCount > 0)
                    {
                        continue;
                    }
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

    const bool useSecondaries =
        m_jobSystem != nullptr && m_secondaryBatchCount > 1 && opaqueItems.size() > 1 &&
        frameIndex < m_gbufferSecondaries.size();

    if (meshCompactionActive)
    {
        RecordMeshletCullPass(commandList, frameIndex, opaqueItems);
    }

    commandList.BeginGpuZone("GBuffer");

    if (useSecondaries)
    {
        const std::uint32_t batchCount = RecordGBufferSecondaries(
            frameIndex,
            gbufferDesc,
            opaqueItems,
            meshRenderingActive,
            meshCompactionActive);

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
        if (meshRenderingActive)
        {
            bool meshPipelineBound = false;
            for (const DrawItem* item : opaqueItems)
            {
                if (!isMeshItem(*item))
                {
                    continue;
                }
                if (!meshPipelineBound)
                {
                    commandList.BindPipeline(*m_gbufferMeshPipeline);
                    commandList.BindDescriptorSet(0, *m_sceneSets[frameIndex]);
                    commandList.BindDescriptorSet(1, *m_bindlessSet);
                    commandList.BindDescriptorSet(2, *m_meshSets[frameIndex]);
                    meshPipelineBound = true;
                }
                commandList.SetPushConstants(
                    &item->pushConstants,
                    sizeof(ObjectPushConstants),
                    rhi::ShaderStageMask::FragmentMesh);
                if (meshCompactionActive && item->compactedOffset != kMeshletCullDisabled)
                {
                    commandList.DrawMeshTasksIndirect(
                        *m_meshletCommandBuffers[frameIndex],
                        static_cast<std::uint64_t>(item->pushConstants.drawIndex) *
                            sizeof(GpuMeshTaskCommand),
                        1u,
                        sizeof(GpuMeshTaskCommand));
                }
                else
                {
                    commandList.DrawMeshTasks(item->meshletCount);
                }
            }
        }

        bool vertexPipelineBound = false;
        for (const DrawItem* item : opaqueItems)
        {
            if (isMeshItem(*item))
            {
                continue;
            }
            if (!vertexPipelineBound)
            {
                commandList.BindPipeline(*m_gbufferPipeline);
                commandList.BindDescriptorSet(0, *m_sceneSets[frameIndex]);
                commandList.BindDescriptorSet(1, *m_bindlessSet);
                vertexPipelineBound = true;
            }
            commandList.BindVertexBuffer(*item->vertexBuffer);
            commandList.BindIndexBuffer(*item->indexBuffer);
            commandList.SetPushConstants(
                &item->pushConstants,
                sizeof(ObjectPushConstants),
                rhi::ShaderStageMask::VertexFragment);
            commandList.DrawIndexed(item->indexCount, item->firstIndex);
        }
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
    // Lighting and the transparent pass render offscreen in HDR with two attachments: the lit
    // image and the subsurface diffuse source. The screen space SSS passes diffuse the source and
    // the composite tone maps everything into the swapchain.
    rhi::RenderingDesc lightingDesc{};
    lightingDesc.colorAttachments = {
        rhi::ColorAttachment{m_lightingColor.get(), rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f}, true},
        rhi::ColorAttachment{m_sssSource.get(), rhi::ClearColor{0.0f, 0.0f, 0.0f, 0.0f}, true}};
    lightingDesc.depthAttachment = rhi::DepthAttachment{&m_gbuffer.GetDepth(), 1.0f, false};
    commandList.BeginRendering(lightingDesc);
    commandList.BindPipeline(*m_lightingPipeline);
    commandList.BindDescriptorSet(0, *m_lightingSets[frameIndex]);
    commandList.BindDescriptorSet(1, *m_bindlessSet);
    commandList.Draw(3);

    if (!transparentItems.empty() && !m_meshletDebugView)
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
    commandList.EndRendering();
    commandList.EndGpuZone();

    // The composite always needs its descriptor set, even when no material uses subsurface
    // scattering (it then passes the lighting target through).
    CreateSssDescriptorSets();
    if (IsSssActive())
    {
        RecordSssPasses(commandList, view);
    }

    commandList.BeginGpuZone("Composite");
    commandList.BeginSwapchainRendering(
        rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f},
        rhi::DepthAttachment{&m_gbuffer.GetDepth(), 1.0f, false},
        true);
    if (m_compositePipeline != nullptr && m_compositeSet != nullptr)
    {
        CompositePushConstants composite{};
        const bool passthrough = m_clusterDebugView || m_meshletDebugView;
        const bool useSharpSource = !IsSssActive();
        const float strength = passthrough ? 0.0f
                                           : (useSharpSource ? 1.0f
                                                             : std::max(m_sssSettings.strength, 0.0f));
        composite.params = glm::vec4{
            strength,
            static_cast<float>(m_sssSettings.debugView),
            useSharpSource ? 1.0f : 0.0f,
            passthrough ? 1.0f : 0.0f};
        commandList.BindPipeline(*m_compositePipeline);
        commandList.BindDescriptorSet(0, *m_compositeSet);
        commandList.SetPushConstants(
            &composite,
            sizeof(composite),
            rhi::ShaderStageMask::Fragment);
        commandList.Draw(3);
    }

    RecordDebugLines(commandList, view);
    commandList.EndSwapchainRendering();
    commandList.EndGpuZone();
}

void DeferredRenderer::RecordDebugLines(rhi::CommandList& commandList, const FrameView& view)
{
    if (m_debugLinePipeline == nullptr ||
        (!m_showLightGizmos && !m_showProbeGizmos && !m_showPhysicsGizmos))
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
            // Probes that failed the depth check (embedded in geometry) are flagged red.
            const bool depthValid = probe.depth.z >= 0.5f;
            const glm::vec4 sphereColor =
                depthValid ? marker.color : glm::vec4{1.0f, 0.15f, 0.15f, 1.0f};
            lines.AddSphere(position, marker.radius, sphereColor);
            lines.AddCross(
                position,
                marker.radius * 0.6f,
                glm::vec4{
                    sphereColor.r * 0.7f,
                    sphereColor.g * 0.7f,
                    sphereColor.b * 0.7f,
                    sphereColor.a});

            const glm::vec3 direction{
                probe.shX.r + probe.shX.g + probe.shX.b,
                probe.shY.r + probe.shY.g + probe.shY.b,
                probe.shZ.r + probe.shZ.g + probe.shZ.b};
            const float directionLength = glm::length(direction);
            if (depthValid && directionLength > 1e-4f)
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

    if (m_showPhysicsGizmos)
    {
        lines.Append(m_physicsDebugLines);
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
    const std::vector<const DrawItem*>& items,
    bool meshRenderingActive,
    bool meshCompactionActive)
{
    MRE_PROFILE_ZONE_N("RecordGBufferSecondaries");
    MRE_ASSERT(m_jobSystem != nullptr, "RecordGBufferSecondaries requires a job system");
    MRE_ASSERT(frameIndex < m_gbufferSecondaries.size(), "Frame index out of range");
    MRE_ASSERT(!items.empty(), "RecordGBufferSecondaries requires draw items");
    MRE_ASSERT(
        !meshRenderingActive ||
            (m_gbufferMeshPipeline != nullptr && frameIndex < m_meshSets.size()),
        "Meshlet secondaries require the mesh pipeline and descriptor set");
    MRE_ASSERT(
        !meshCompactionActive ||
            (m_meshletCullPipeline != nullptr && frameIndex < m_meshletCommandBuffers.size() &&
             m_meshletCommandBuffers[frameIndex] != nullptr),
        "Meshlet secondaries require the cull command buffer");

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
        futures.push_back(m_jobSystem->Submit(
            [this, frameIndex, secondary, &desc, &items, begin, end, meshRenderingActive, meshCompactionActive]() {
                RecordGBufferBatch(
                    frameIndex,
                    items,
                    begin,
                    end,
                    meshRenderingActive,
                    meshCompactionActive,
                    *secondary,
                    desc);
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
    bool meshRenderingActive,
    bool meshCompactionActive,
    rhi::CommandList& secondary,
    const rhi::RenderingDesc& desc)
{
    MRE_ASSERT(m_gbufferPipeline != nullptr, "RecordGBufferBatch requires the G-buffer pipeline");
    MRE_ASSERT(frameIndex < m_sceneSets.size(), "Frame index out of range");
    MRE_ASSERT(
        !meshRenderingActive ||
            (m_gbufferMeshPipeline != nullptr && frameIndex < m_meshSets.size()),
        "Meshlet batches require the mesh pipeline and descriptor set");
    MRE_ASSERT(
        !meshCompactionActive ||
            (frameIndex < m_meshletCommandBuffers.size() &&
             m_meshletCommandBuffers[frameIndex] != nullptr),
        "Meshlet batches require the cull command buffer");

    secondary.BeginSecondary(desc);
    secondary.BeginGpuZone("GBufferBatch");

    // Mesh items first so the batch binds each pipeline at most once.
    if (meshRenderingActive)
    {
        bool meshPipelineBound = false;
        for (std::size_t index = begin; index < end; ++index)
        {
            const DrawItem& item = *items[index];
            if (item.meshletCount == 0)
            {
                continue;
            }
            if (!meshPipelineBound)
            {
                secondary.BindPipeline(*m_gbufferMeshPipeline);
                secondary.BindDescriptorSet(0, *m_sceneSets[frameIndex]);
                secondary.BindDescriptorSet(1, *m_bindlessSet);
                secondary.BindDescriptorSet(2, *m_meshSets[frameIndex]);
                meshPipelineBound = true;
            }
            secondary.SetPushConstants(
                &item.pushConstants,
                sizeof(ObjectPushConstants),
                rhi::ShaderStageMask::FragmentMesh);
            if (meshCompactionActive && item.compactedOffset != kMeshletCullDisabled)
            {
                secondary.DrawMeshTasksIndirect(
                    *m_meshletCommandBuffers[frameIndex],
                    static_cast<std::uint64_t>(item.pushConstants.drawIndex) *
                        sizeof(GpuMeshTaskCommand),
                    1u,
                    sizeof(GpuMeshTaskCommand));
            }
            else
            {
                secondary.DrawMeshTasks(item.meshletCount);
            }
        }
    }

    bool vertexPipelineBound = false;
    for (std::size_t index = begin; index < end; ++index)
    {
        const DrawItem& item = *items[index];
        if (meshRenderingActive && item.meshletCount > 0)
        {
            continue;
        }
        if (!vertexPipelineBound)
        {
            secondary.BindPipeline(*m_gbufferPipeline);
            secondary.BindDescriptorSet(0, *m_sceneSets[frameIndex]);
            secondary.BindDescriptorSet(1, *m_bindlessSet);
            vertexPipelineBound = true;
        }
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

void DeferredRenderer::RecordMeshletCullPass(
    rhi::CommandList& commandList,
    std::uint32_t frameIndex,
    const std::vector<const DrawItem*>& items)
{
    MRE_PROFILE_ZONE_N("RecordMeshletCullPass");
    MRE_ASSERT(m_meshletCullPipeline != nullptr, "RecordMeshletCullPass requires the cull pipeline");
    MRE_ASSERT(m_meshletCompactionActive, "RecordMeshletCullPass requires active compaction");
    MRE_ASSERT(frameIndex < m_sceneSets.size(), "Frame index out of range");
    MRE_ASSERT(frameIndex < m_meshletCullSets.size(), "Frame index out of range");

    commandList.BeginGpuZone("MeshletCull");
    commandList.BindPipeline(*m_meshletCullPipeline);
    commandList.BindDescriptorSet(0, *m_sceneSets[frameIndex]);
    commandList.BindDescriptorSet(1, *m_meshletCullSets[frameIndex]);
    commandList.BindDescriptorSet(2, *m_bindlessSet);

    MeshletCullPushConstants cull{};
    for (const DrawItem* item : items)
    {
        if (item->meshletCount == 0 || item->compactedOffset == kMeshletCullDisabled)
        {
            continue;
        }

        cull.model = item->pushConstants.model;
        cull.drawIndex = item->pushConstants.drawIndex;
        cull.meshletCount = item->meshletCount;
        commandList.SetPushConstants(&cull, sizeof(cull), rhi::ShaderStageMask::Compute);
        commandList.Dispatch(
            (item->meshletCount + kMeshletCullGroupSize - 1u) / kMeshletCullGroupSize,
            1u,
            1u);
    }

    commandList.EndGpuZone();
    commandList.MeshTaskIndirectBarrier();
}

void DeferredRenderer::BuildShadowCullItems()
{
    m_shadowCullItems.clear();

    // The ground plane spans 20x20 world units at its default extent, scaled by SetGroundPlaneExtent.
    const float groundRadius = 14.15f * (m_groundPlaneExtent / 10.0f);
    for (const DrawItem& item : m_sceneDrawItems)
    {
        m_shadowCullItems.push_back(ShadowCullItem{&item, glm::vec3{0.0f}, groundRadius});
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

void DeferredRenderer::BuildShadowMeshletCull(std::uint32_t frameIndex)
{
    MRE_PROFILE_ZONE_N("BuildShadowMeshletCull");
    m_shadowMeshletCompactionActive = false;
    for (ShadowView& view : m_shadowViews)
    {
        view.meshDraws.clear();
    }
    if (m_shadowViews.empty() || !m_meshletRenderingEnabled || !m_meshletRenderingSupported ||
        m_shadowMeshPipeline == nullptr)
    {
        return;
    }

    // Gather the mesh items of every view. Command indices and compacted offsets are assigned in
    // view/item order so the cull pass and the per-view draw recording agree.
    std::uint32_t commandCount = 0;
    std::uint32_t meshletCount = 0;
    for (ShadowView& view : m_shadowViews)
    {
        view.meshDraws.reserve(view.items.size());
        for (const DrawItem* item : view.items)
        {
            if (item->meshletCount == 0)
            {
                continue;
            }
            view.meshDraws.push_back(ShadowMeshDraw{item, commandCount, meshletCount});
            ++commandCount;
            meshletCount += item->meshletCount;
        }
    }

    const bool resourcesAvailable =
        m_shadowCullPipeline != nullptr && frameIndex < m_shadowCullSets.size() &&
        frameIndex < m_shadowMeshletIndexBuffers.size() &&
        frameIndex < m_shadowMeshletCommandBuffers.size() &&
        m_shadowCullSets[frameIndex] != nullptr &&
        m_shadowMeshletIndexBuffers[frameIndex] != nullptr &&
        m_shadowMeshletCommandBuffers[frameIndex] != nullptr;
    const bool fits = commandCount <= kMaxShadowViews * kMaxObjectData &&
                      meshletCount <= kShadowMeshletCompactionCapacity;
    m_shadowMeshletCompactionActive = resourcesAvailable && fits;
    if (!m_shadowMeshletCompactionActive)
    {
        for (ShadowView& view : m_shadowViews)
        {
            for (ShadowMeshDraw& draw : view.meshDraws)
            {
                draw.commandIndex = kMeshletCullDisabled;
                draw.compactedOffset = kMeshletCullDisabled;
            }
        }
        return;
    }

    // Rearm the per-view indirect commands: groupCountY/Z stay 1 and groupCountX is accumulated
    // by the cull dispatches.
    std::vector<std::uint32_t> commands(commandCount * 3u, 1u);
    for (std::uint32_t command = 0; command < commandCount; ++command)
    {
        commands[command * 3u] = 0u;
    }
    if (!commands.empty())
    {
        m_shadowMeshletCommandBuffers[frameIndex]->Write(
            commands.data(),
            commands.size() * sizeof(std::uint32_t));
    }
}

void DeferredRenderer::RecordShadowMeshletCullPass(
    rhi::CommandList& commandList,
    std::uint32_t frameIndex)
{
    MRE_PROFILE_ZONE_N("RecordShadowMeshletCullPass");
    MRE_ASSERT(m_shadowCullPipeline != nullptr, "RecordShadowMeshletCullPass requires the cull pipeline");
    MRE_ASSERT(
        m_shadowMeshletCompactionActive,
        "RecordShadowMeshletCullPass requires active compaction");
    MRE_ASSERT(frameIndex < m_shadowSets.size(), "Frame index out of range");
    MRE_ASSERT(frameIndex < m_shadowCullSets.size(), "Frame index out of range");

    commandList.BeginGpuZone("ShadowMeshletCull");
    commandList.BindPipeline(*m_shadowCullPipeline);
    commandList.BindDescriptorSet(0, *m_shadowSets[frameIndex]);
    commandList.BindDescriptorSet(1, *m_shadowCullSets[frameIndex]);
    commandList.BindDescriptorSet(2, *m_bindlessSet);

    ShadowCullPushConstants cull{};
    for (const ShadowView& view : m_shadowViews)
    {
        for (const ShadowMeshDraw& meshDraw : view.meshDraws)
        {
            const DrawItem& item = *meshDraw.item;
            cull.model = item.pushConstants.model;
            cull.viewIndex = view.viewIndex;
            cull.drawIndex = item.pushConstants.drawIndex;
            cull.compactedOffset = meshDraw.compactedOffset;
            cull.commandIndex = meshDraw.commandIndex;
            cull.meshletCount = item.meshletCount;
            commandList.SetPushConstants(&cull, sizeof(cull), rhi::ShaderStageMask::Compute);
            commandList.Dispatch(
                (item.meshletCount + kMeshletCullGroupSize - 1u) / kMeshletCullGroupSize,
                1u,
                1u);
        }
    }

    commandList.EndGpuZone();
    commandList.MeshTaskIndirectBarrier();
}

void DeferredRenderer::RecordShadowViews(
    std::uint32_t frameIndex,
    bool meshShadowActive,
    bool meshCompactionActive)
{
    MRE_PROFILE_ZONE_N("RecordShadowViews");
    MRE_ASSERT(m_jobSystem != nullptr, "RecordShadowViews requires a job system");
    MRE_ASSERT(frameIndex < m_shadowSecondaries.size(), "Frame index out of range");
    MRE_ASSERT(
        m_shadowViews.size() <= m_shadowSecondaries[frameIndex].size(),
        "Not enough shadow secondary command lists were allocated");
    MRE_ASSERT(
        !meshShadowActive ||
            (m_shadowMeshPipeline != nullptr && frameIndex < m_shadowMeshSets.size()),
        "Meshlet shadow views require the mesh pipeline and descriptor set");
    MRE_ASSERT(
        !meshCompactionActive ||
            (frameIndex < m_shadowMeshletCommandBuffers.size() &&
             m_shadowMeshletCommandBuffers[frameIndex] != nullptr),
        "Meshlet shadow views require the cull command buffer");

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

        futures.push_back(m_jobSystem->Submit(
            [this, frameIndex, secondary, &view, desc, meshShadowActive, meshCompactionActive]() {
                RecordShadowView(
                    frameIndex,
                    view,
                    meshShadowActive,
                    meshCompactionActive,
                    *secondary,
                    desc);
            }));
    }

    for (std::future<void>& future : futures)
    {
        future.get();
    }
}

void DeferredRenderer::RecordShadowView(
    std::uint32_t frameIndex,
    const ShadowView& view,
    bool meshShadowActive,
    bool meshCompactionActive,
    rhi::CommandList& secondary,
    const rhi::RenderingDesc& desc)
{
    MRE_ASSERT(m_shadowPipeline != nullptr, "RecordShadowView requires the shadow pipeline");
    MRE_ASSERT(frameIndex < m_shadowSets.size(), "Frame index out of range");
    MRE_ASSERT(
        !meshShadowActive ||
            (m_shadowMeshPipeline != nullptr && frameIndex < m_shadowMeshSets.size()),
        "Meshlet shadow views require the mesh pipeline and descriptor set");
    MRE_ASSERT(
        !meshCompactionActive ||
            (frameIndex < m_shadowMeshletCommandBuffers.size() &&
             m_shadowMeshletCommandBuffers[frameIndex] != nullptr),
        "Meshlet shadow views require the cull command buffer");

    secondary.BeginSecondary(desc);
    secondary.BeginGpuZone("ShadowView");

    ShadowPushConstants pushConstants{};
    pushConstants.viewIndex = view.viewIndex;

    // Mesh items first so the view binds each pipeline at most once.
    if (meshShadowActive)
    {
        secondary.BindPipeline(*m_shadowMeshPipeline);
        secondary.BindDescriptorSet(0, *m_shadowSets[frameIndex]);
        secondary.BindDescriptorSet(1, *m_bindlessSet);
        secondary.BindDescriptorSet(2, *m_shadowMeshSets[frameIndex]);
        secondary.SetDepthBias(m_shadowSettings.depthBias, m_shadowSettings.depthBias * 1.5f, 0.0f);

        const std::uint32_t meshPushConstantSize =
            m_shadowMeshPipeline->GetPushConstantSize();
        for (const ShadowMeshDraw& meshDraw : view.meshDraws)
        {
            const DrawItem& item = *meshDraw.item;
            pushConstants.object = item.pushConstants;
            pushConstants.compactedOffset = meshCompactionActive ? meshDraw.compactedOffset
                                                                : kMeshletCullDisabled;
            secondary.SetPushConstants(
                &pushConstants,
                meshPushConstantSize,
                rhi::ShaderStageMask::FragmentMesh);
            if (meshCompactionActive && meshDraw.compactedOffset != kMeshletCullDisabled)
            {
                secondary.DrawMeshTasksIndirect(
                    *m_shadowMeshletCommandBuffers[frameIndex],
                    static_cast<std::uint64_t>(meshDraw.commandIndex) *
                        sizeof(GpuMeshTaskCommand),
                    1u,
                    sizeof(GpuMeshTaskCommand));
            }
            else
            {
                secondary.DrawMeshTasks(item.meshletCount);
            }
        }
    }

    secondary.BindPipeline(*m_shadowPipeline);
    secondary.BindDescriptorSet(0, *m_shadowSets[frameIndex]);
    secondary.BindDescriptorSet(1, *m_bindlessSet);
    secondary.SetDepthBias(m_shadowSettings.depthBias, m_shadowSettings.depthBias * 1.5f, 0.0f);

    const std::uint32_t pushConstantSize = m_shadowPipeline->GetPushConstantSize();
    for (const DrawItem* item : view.items)
    {
        if (meshShadowActive && item->meshletCount > 0)
        {
            continue;
        }
        pushConstants.object = item->pushConstants;
        pushConstants.compactedOffset = kMeshletCullDisabled;
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
    m_sssBlurShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "sss_blur.comp.spv");
    m_compositeShader = device.CreateShader(
        rhi::ShaderStage::Fragment,
        shaderDirectory / "composite.frag.spv");

    // Meshlet rendering is optional: devices without mesh shaders keep the vertex pipeline.
    if (device.IsMeshShaderSupported())
    {
        m_gbufferMeshShader = device.CreateShader(
            rhi::ShaderStage::Mesh,
            shaderDirectory / "gbuffer.mesh.spv");
        m_shadowMeshShader = device.CreateShader(
            rhi::ShaderStage::Mesh,
            shaderDirectory / "shadow.mesh.spv");
        m_shadowCullShader = device.CreateShader(
            rhi::ShaderStage::Compute,
            shaderDirectory / "shadow_cull.comp.spv");
        m_meshletCullShader = device.CreateShader(
            rhi::ShaderStage::Compute,
            shaderDirectory / "meshlet_cull.comp.spv");
        if (m_gbufferMeshShader == nullptr || m_shadowMeshShader == nullptr)
        {
            MRE_WARN("Failed to load the mesh shaders; meshlet rendering stays disabled");
        }
        if (m_meshletCullShader == nullptr)
        {
            MRE_WARN("Failed to load the meshlet cull shader; meshlets cull in the mesh shader");
        }
        if (m_shadowCullShader == nullptr)
        {
            MRE_WARN("Failed to load the shadow cull shader; shadow meshlets cull in the shader");
        }
    }

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
        m_cloudReprojectShader == nullptr || m_cloudGodRayShader == nullptr ||
        m_sssBlurShader == nullptr || m_compositeShader == nullptr)
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

    // Meshlet variant of the opaque G-buffer pass; the fragment stage and attachments match.
    if (m_gbufferMeshShader != nullptr && m_meshSetLayout != nullptr)
    {
        const std::array<const rhi::Shader*, 2> gbufferMeshStages{
            m_gbufferMeshShader.get(),
            m_gbufferFragmentShader.get()};
        rhi::PipelineDesc gbufferMeshPipelineDesc{};
        gbufferMeshPipelineDesc.meshShader = m_gbufferMeshShader.get();
        gbufferMeshPipelineDesc.fragmentShader = m_gbufferFragmentShader.get();
        rhi::ApplyShaderReflection(gbufferMeshPipelineDesc, gbufferMeshStages);
        gbufferMeshPipelineDesc.depthTest = true;
        gbufferMeshPipelineDesc.depthWrite = true;
        gbufferMeshPipelineDesc.blend = false;
        gbufferMeshPipelineDesc.cullMode = rhi::CullMode::None;
        gbufferMeshPipelineDesc.descriptorSetLayouts = {
            m_sceneSetLayout.get(),
            m_bindlessSetLayout.get(),
            m_meshSetLayout.get()};
        gbufferMeshPipelineDesc.colorFormats = gbufferPipelineDesc.colorFormats;
        gbufferMeshPipelineDesc.depthFormat = rhi::Format::D32Sfloat;
        m_gbufferMeshPipeline = device.CreatePipeline(gbufferMeshPipelineDesc);
    }

    // Compute cull feeding the G-buffer meshlet draws: set 0 is the scene uniform set, set 1 the
    // mesh draw/compaction/indirect buffers and set 2 the bindless meshlet data.
    if (m_meshletCullShader != nullptr && m_meshletCullSetLayout != nullptr)
    {
        const std::array<const rhi::Shader*, 1> cullStages{m_meshletCullShader.get()};
        rhi::PipelineDesc meshletCullPipelineDesc{};
        meshletCullPipelineDesc.computeShader = m_meshletCullShader.get();
        rhi::ApplyShaderReflection(meshletCullPipelineDesc, cullStages);
        meshletCullPipelineDesc.descriptorSetLayouts = {
            m_sceneSetLayout.get(),
            m_meshletCullSetLayout.get(),
            m_bindlessSetLayout.get()};
        m_meshletCullPipeline = device.CreatePipeline(meshletCullPipelineDesc);
        if (m_meshletCullPipeline == nullptr)
        {
            MRE_WARN("Failed to create the meshlet cull pipeline; meshlets cull in the mesh shader");
        }
    }

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
    // The transparent pass renders into the offscreen HDR lighting scope, which has two
    // attachments; the pipeline must declare the matching attachment count (the subsurface target
    // keeps its cleared value).
    transparentPipelineDesc.colorFormats = {
        rhi::Format::R16G16B16A16Sfloat,
        rhi::Format::R16G16B16A16Sfloat};
    transparentPipelineDesc.depthFormat = rhi::Format::D32Sfloat;
    m_transparentPipeline = device.CreatePipeline(transparentPipelineDesc);

    rhi::PipelineDesc lightingPipelineDesc{};
    lightingPipelineDesc.vertexShader = m_lightingVertexShader.get();
    lightingPipelineDesc.fragmentShader = m_lightingFragmentShader.get();
    lightingPipelineDesc.depthTest = false;
    lightingPipelineDesc.depthWrite = false;
    lightingPipelineDesc.blend = false;
    lightingPipelineDesc.cullMode = rhi::CullMode::None;
    // Set 1 is the bindless material set used for the emissive/subsurface texture lookups.
    lightingPipelineDesc.descriptorSetLayouts = {m_lightingSetLayout.get(), m_bindlessSetLayout.get()};
    // Attachment 0: HDR lit image, attachment 1: subsurface diffuse source (blurred before the
    // composite adds it and applies tone mapping).
    lightingPipelineDesc.colorFormats = {
        rhi::Format::R16G16B16A16Sfloat,
        rhi::Format::R16G16B16A16Sfloat};
    lightingPipelineDesc.depthFormat = rhi::Format::D32Sfloat;
    m_lightingPipeline = device.CreatePipeline(lightingPipelineDesc);

    if (m_sssBlurShader != nullptr)
    {
        const std::array<const rhi::Shader*, 1> sssBlurStages{m_sssBlurShader.get()};
        m_sssBlurSetLayout =
            device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(sssBlurStages, 0));

        rhi::PipelineDesc sssBlurPipelineDesc{};
        sssBlurPipelineDesc.computeShader = m_sssBlurShader.get();
        sssBlurPipelineDesc.descriptorSetLayouts = {m_sssBlurSetLayout.get()};
        sssBlurPipelineDesc.pushConstantSize = sizeof(SssBlurPushConstants);
        sssBlurPipelineDesc.pushConstantStages = rhi::ShaderStageMask::Compute;
        m_sssBlurPipeline = device.CreatePipeline(sssBlurPipelineDesc);
    }

    if (m_compositeShader != nullptr)
    {
        const std::array<const rhi::Shader*, 1> compositeStages{m_compositeShader.get()};
        m_compositeSetLayout =
            device.CreateDescriptorSetLayout(rhi::MergeShaderReflectionBindings(compositeStages, 0));

        rhi::PipelineDesc compositePipelineDesc{};
        compositePipelineDesc.vertexShader = m_lightingVertexShader.get();
        compositePipelineDesc.fragmentShader = m_compositeShader.get();
        compositePipelineDesc.descriptorSetLayouts = {m_compositeSetLayout.get()};
        compositePipelineDesc.pushConstantSize = sizeof(CompositePushConstants);
        compositePipelineDesc.pushConstantStages = rhi::ShaderStageMask::Fragment;
        compositePipelineDesc.cullMode = rhi::CullMode::None;
        compositePipelineDesc.colorFormats = {device.GetSwapchainInfo().format};
        // Drawn in the swapchain scope that also hosts the debug line pass (depth test disabled).
        compositePipelineDesc.depthFormat = rhi::Format::D32Sfloat;
        m_compositePipeline = device.CreatePipeline(compositePipelineDesc);
    }

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
    // The shadow pass consumes a subset of the interleaved GltfVertex layout (position at byte 0,
    // uv at byte 24, joints at byte 32, weights at byte 48); reflection cannot infer the
    // interleaved offsets. The skinning attributes are consumed for pose evaluation so their
    // locations must be declared even though only positions drive the depth output.
    shadowPipelineDesc.vertexLayout = rhi::VertexLayout{};
    shadowPipelineDesc.vertexLayout.AddAttribute(0, rhi::Format::R32G32B32Sfloat, 0);
    shadowPipelineDesc.vertexLayout.AddAttribute(2, rhi::Format::R32G32Sfloat, 24);
    shadowPipelineDesc.vertexLayout.AddAttribute(3, rhi::Format::R32G32B32A32Uint, 32);
    shadowPipelineDesc.vertexLayout.AddAttribute(4, rhi::Format::R32G32B32A32Sfloat, 48);
    shadowPipelineDesc.vertexLayout.SetBindingStride(gbufferPipelineDesc.vertexLayout.GetStride());
    shadowPipelineDesc.depthTest = true;
    shadowPipelineDesc.depthWrite = true;
    shadowPipelineDesc.depthBiasEnable = true;
    shadowPipelineDesc.cullMode = rhi::CullMode::None;
    // The fragment stage alpha-tests masked materials through the bindless textures.
    shadowPipelineDesc.descriptorSetLayouts = {m_shadowSetLayout.get(), m_bindlessSetLayout.get()};
    shadowPipelineDesc.depthFormat = rhi::Format::D32Sfloat;
    m_shadowPipeline = device.CreatePipeline(shadowPipelineDesc);

    if (m_shadowMeshShader != nullptr && m_meshSetLayout != nullptr)
    {
        const std::array<const rhi::Shader*, 2> shadowMeshStages{
            m_shadowMeshShader.get(),
            m_shadowFragmentShader.get()};
        rhi::PipelineDesc shadowMeshPipelineDesc{};
        shadowMeshPipelineDesc.meshShader = m_shadowMeshShader.get();
        shadowMeshPipelineDesc.fragmentShader = m_shadowFragmentShader.get();
        rhi::ApplyShaderReflection(shadowMeshPipelineDesc, shadowMeshStages);
        shadowMeshPipelineDesc.depthTest = true;
        shadowMeshPipelineDesc.depthWrite = true;
        shadowMeshPipelineDesc.depthBiasEnable = true;
        shadowMeshPipelineDesc.cullMode = rhi::CullMode::None;
        shadowMeshPipelineDesc.descriptorSetLayouts = {
            m_shadowSetLayout.get(),
            m_bindlessSetLayout.get(),
            m_meshSetLayout.get()};
        shadowMeshPipelineDesc.depthFormat = rhi::Format::D32Sfloat;
        m_shadowMeshPipeline = device.CreatePipeline(shadowMeshPipelineDesc);
    }

    // Compute cull for the shadow views: set 0 is the shadow uniform set, set 1 the mesh
    // draw/compaction/indirect buffers and set 2 the bindless meshlet data.
    if (m_shadowCullShader != nullptr && m_shadowCullSetLayout != nullptr)
    {
        const std::array<const rhi::Shader*, 1> shadowCullStages{m_shadowCullShader.get()};
        rhi::PipelineDesc shadowCullPipelineDesc{};
        shadowCullPipelineDesc.computeShader = m_shadowCullShader.get();
        rhi::ApplyShaderReflection(shadowCullPipelineDesc, shadowCullStages);
        shadowCullPipelineDesc.descriptorSetLayouts = {
            m_shadowSetLayout.get(),
            m_shadowCullSetLayout.get(),
            m_bindlessSetLayout.get()};
        m_shadowCullPipeline = device.CreatePipeline(shadowCullPipelineDesc);
        if (m_shadowCullPipeline == nullptr)
        {
            MRE_WARN("Failed to create the shadow cull pipeline; shadow meshlets cull in the shader");
        }
    }

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
    ground.pushConstants.model = glm::scale(glm::mat4{1.0f}, glm::vec3{m_groundPlaneExtent / 10.0f});
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

    m_groundDrawItem = ground;
    m_groundDrawItemValid = true;
    if (m_groundVisible)
    {
        m_sceneDrawItems.push_back(ground);
    }

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
        if (frame < m_objectDataBuffers.size() && m_objectDataBuffers[frame] != nullptr)
        {
            shadowSet->BindBuffer(1, *m_objectDataBuffers[frame]);
        }
        if (frame < m_jointBuffers.size() && m_jointBuffers[frame] != nullptr)
        {
            shadowSet->BindBuffer(2, *m_jointBuffers[frame]);
        }
        if (frame < m_morphDataBuffers.size() && m_morphDataBuffers[frame] != nullptr)
        {
            shadowSet->BindBuffer(3, *m_morphDataBuffers[frame]);
        }
        if (frame < m_morphWeightBuffers.size() && m_morphWeightBuffers[frame] != nullptr)
        {
            shadowSet->BindBuffer(4, *m_morphWeightBuffers[frame]);
        }
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

void DeferredRenderer::CreateJointPaletteResources(rhi::Device& device)
{
    const std::uint32_t framesInFlight = device.GetFramesInFlight();
    m_jointBuffers.clear();
    m_jointBuffers.reserve(framesInFlight);
    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        // The current and previous palettes share the buffer; the previous region starts at
        // kMaxJointMatrices.
        m_jointBuffers.push_back(device.CreateBuffer(rhi::BufferDesc{
            2ull * kMaxJointMatrices * sizeof(glm::mat4),
            rhi::BufferUsage::Storage,
            true}));
        if (m_jointBuffers.back() == nullptr)
        {
            MRE_CRITICAL("Failed to create the joint palette buffer");
            return;
        }
    }
}

void DeferredRenderer::CreateMeshletResources(rhi::Device& device)
{
    if (m_meshSetLayout == nullptr)
    {
        return;
    }

    const std::uint32_t framesInFlight = device.GetFramesInFlight();
    m_meshDrawBuffers.clear();
    m_meshletIndexBuffers.clear();
    m_meshletCommandBuffers.clear();
    m_meshSets.clear();
    m_meshletCullSets.clear();
    m_shadowMeshletIndexBuffers.clear();
    m_shadowMeshletCommandBuffers.clear();
    m_shadowMeshSets.clear();
    m_shadowCullSets.clear();
    m_meshDrawBuffers.reserve(framesInFlight);
    m_meshletIndexBuffers.reserve(framesInFlight);
    m_meshletCommandBuffers.reserve(framesInFlight);
    m_meshSets.reserve(framesInFlight);
    m_meshletCullSets.reserve(framesInFlight);
    m_shadowMeshletIndexBuffers.reserve(framesInFlight);
    m_shadowMeshletCommandBuffers.reserve(framesInFlight);
    m_shadowMeshSets.reserve(framesInFlight);
    m_shadowCullSets.reserve(framesInFlight);
    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        m_meshDrawBuffers.push_back(device.CreateBuffer(
            rhi::BufferDesc{kMaxObjectData * sizeof(GpuMeshDraw), rhi::BufferUsage::Storage, true}));
        if (m_meshDrawBuffers.back() == nullptr)
        {
            MRE_CRITICAL("Failed to create the mesh draw data buffer");
            return;
        }

        // The compute cull writes the visible meshlet indices of every draw into this frame buffer.
        m_meshletIndexBuffers.push_back(device.CreateBuffer(rhi::BufferDesc{
            kMeshletCompactionCapacity * sizeof(std::uint32_t),
            rhi::BufferUsage::Storage}));
        if (m_meshletIndexBuffers.back() == nullptr)
        {
            MRE_CRITICAL("Failed to create the meshlet compaction buffer");
            return;
        }

        std::unique_ptr<rhi::DescriptorSet> meshSet = device.CreateDescriptorSet(*m_meshSetLayout);
        meshSet->BindBuffer(0, *m_meshDrawBuffers.back());
        meshSet->BindBuffer(1, *m_meshletIndexBuffers.back());
        m_meshSets.push_back(std::move(meshSet));

        if (m_meshletCullSetLayout != nullptr)
        {
            // Host visible so the per-frame command reset can rearm the indirect group counts.
            m_meshletCommandBuffers.push_back(device.CreateBuffer(rhi::BufferDesc{
                kMaxObjectData * sizeof(GpuMeshTaskCommand),
                rhi::BufferUsage::Storage,
                true,
                false,
                0,
                false,
                true}));
            if (m_meshletCommandBuffers.back() == nullptr)
            {
                MRE_CRITICAL("Failed to create the meshlet cull command buffer");
                return;
            }

            std::unique_ptr<rhi::DescriptorSet> cullSet =
                device.CreateDescriptorSet(*m_meshletCullSetLayout);
            cullSet->BindBuffer(0, *m_meshDrawBuffers.back());
            cullSet->BindBuffer(1, *m_meshletIndexBuffers.back());
            cullSet->BindBuffer(2, *m_meshletCommandBuffers.back());
            m_meshletCullSets.push_back(std::move(cullSet));
        }

        if (m_shadowCullSetLayout != nullptr)
        {
            // Shadow views compact into their own per-frame list and indirect commands.
            m_shadowMeshletIndexBuffers.push_back(device.CreateBuffer(rhi::BufferDesc{
                kShadowMeshletCompactionCapacity * sizeof(std::uint32_t),
                rhi::BufferUsage::Storage}));
            if (m_shadowMeshletIndexBuffers.back() == nullptr)
            {
                MRE_CRITICAL("Failed to create the shadow meshlet compaction buffer");
                return;
            }

            m_shadowMeshletCommandBuffers.push_back(device.CreateBuffer(rhi::BufferDesc{
                kMaxShadowViews * kMaxObjectData * sizeof(GpuMeshTaskCommand),
                rhi::BufferUsage::Storage,
                true,
                false,
                0,
                false,
                true}));
            if (m_shadowMeshletCommandBuffers.back() == nullptr)
            {
                MRE_CRITICAL("Failed to create the shadow meshlet cull command buffer");
                return;
            }

            std::unique_ptr<rhi::DescriptorSet> shadowCullSet =
                device.CreateDescriptorSet(*m_shadowCullSetLayout);
            shadowCullSet->BindBuffer(0, *m_meshDrawBuffers.back());
            shadowCullSet->BindBuffer(1, *m_shadowMeshletIndexBuffers.back());
            shadowCullSet->BindBuffer(2, *m_shadowMeshletCommandBuffers.back());
            m_shadowCullSets.push_back(std::move(shadowCullSet));
        }

        // The shadow mesh shader remaps through the shadow compaction list; bind the G-buffer list
        // as a valid placeholder when shadow culling is unavailable (the sentinel path never reads
        // it).
        const rhi::Buffer* shadowIndexBuffer =
            m_shadowMeshletIndexBuffers.size() == static_cast<std::size_t>(frame) + 1u
                ? m_shadowMeshletIndexBuffers.back().get()
                : m_meshletIndexBuffers.back().get();
        std::unique_ptr<rhi::DescriptorSet> shadowMeshSet =
            device.CreateDescriptorSet(*m_meshSetLayout);
        shadowMeshSet->BindBuffer(0, *m_meshDrawBuffers.back());
        shadowMeshSet->BindBuffer(1, *shadowIndexBuffer);
        m_shadowMeshSets.push_back(std::move(shadowMeshSet));
    }
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
        // z: debug view selection (0 = lit, 1 = cluster light count heatmap, 2 = GI only,
        // 3 = per-meshlet colors).
        m_meshletDebugView ? 3.0f
                           : (m_giDebugView ? 2.0f : (m_clusterDebugView ? 1.0f : 0.0f)),
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

bool DeferredRenderer::ReadGiProbes(std::vector<GpuGiProbe>& out)
{
    out.clear();
    if (m_giProbeBuffer == nullptr || m_giProbeData.empty() || m_device == nullptr)
    {
        return false;
    }

    m_device->WaitIdle();
    out.resize(m_giProbeData.size());
    if (!m_giProbeBuffer->Read(out.data(), out.size() * sizeof(GpuGiProbe)))
    {
        return false;
    }

    // Keep the CPU copy's SH and depth current for the debug gizmos; the positions stay untouched
    // and the data is not re-uploaded (the GPU owns the accumulated coefficients).
    for (std::size_t index = 0; index < m_giProbeData.size(); ++index)
    {
        m_giProbeData[index].sh0 = out[index].sh0;
        m_giProbeData[index].shX = out[index].shX;
        m_giProbeData[index].shY = out[index].shY;
        m_giProbeData[index].shZ = out[index].shZ;
        m_giProbeData[index].depth = out[index].depth;
    }
    return true;
}

bool DeferredRenderer::CreateSssResources(rhi::Device& device)
{
    m_lightingColor.reset();
    m_sssSource.reset();
    m_sssBlurTargets[0].reset();
    m_sssBlurTargets[1].reset();
    m_sssLut.reset();
    m_sssSampler.reset();
    m_sssBlurSets.clear();
    m_compositeSet.reset();

    const std::uint32_t width = m_gbuffer.GetWidth();
    const std::uint32_t height = m_gbuffer.GetHeight();
    if (width == 0 || height == 0)
    {
        return false;
    }

    m_sssHalfWidth = std::max(1u, width / 2u);
    m_sssHalfHeight = std::max(1u, height / 2u);

    m_lightingColor = device.CreateTexture(rhi::TextureDesc{
        width,
        height,
        rhi::Format::R16G16B16A16Sfloat,
        rhi::TextureUsage::ColorAttachment});
    m_sssSource = device.CreateTexture(rhi::TextureDesc{
        width,
        height,
        rhi::Format::R16G16B16A16Sfloat,
        rhi::TextureUsage::ColorAttachment});
    m_sssBlurTargets[0] = device.CreateTexture(rhi::TextureDesc{
        m_sssHalfWidth,
        m_sssHalfHeight,
        rhi::Format::R16G16B16A16Sfloat,
        rhi::TextureUsage::Storage});
    m_sssBlurTargets[1] = device.CreateTexture(rhi::TextureDesc{
        m_sssHalfWidth,
        m_sssHalfHeight,
        rhi::Format::R16G16B16A16Sfloat,
        rhi::TextureUsage::Storage});

    std::vector<float> lutPixels{};
    BuildSubsurfaceLut(lutPixels);
    m_sssLut = device.CreateTexture(rhi::TextureDesc{
        kSubsurfaceLutWidth,
        kSubsurfaceLutHeight,
        rhi::Format::R32G32B32A32Sfloat,
        rhi::TextureUsage::Sampled,
        lutPixels.data(),
        lutPixels.size() * sizeof(float)});
    m_sssSampler = device.CreateSampler(rhi::SamplerDesc{});

    if (m_lightingColor == nullptr || m_sssSource == nullptr ||
        m_sssBlurTargets[0] == nullptr || m_sssBlurTargets[1] == nullptr ||
        m_sssLut == nullptr || m_sssSampler == nullptr)
    {
        MRE_CRITICAL("Failed to create the subsurface scattering resources");
        return false;
    }
    return true;
}

// Descriptor sets are created lazily after the first lighting pass: the color targets are in the
// undefined layout at creation and only become shader readable when the pass ends.
void DeferredRenderer::CreateSssDescriptorSets()
{
    if (m_device == nullptr || m_sssBlurPipeline == nullptr || m_sssBlurSetLayout == nullptr ||
        m_compositeSetLayout == nullptr || m_lightingColor == nullptr || m_sssSource == nullptr ||
        m_sssBlurTargets[0] == nullptr || m_sssBlurTargets[1] == nullptr || m_sssLut == nullptr ||
        m_sssSampler == nullptr)
    {
        return;
    }

    if (m_sssBlurSets.empty())
    {
        constexpr std::uint32_t kMaxPasses = 6; // 3 iterations * (horizontal + vertical)
        m_sssBlurSets.reserve(kMaxPasses);
        for (std::uint32_t pass = 0; pass < kMaxPasses; ++pass)
        {
            const bool horizontal = (pass % 2u) == 0u;
            rhi::Texture* source = m_sssSource.get();
            if (pass > 0)
            {
                source = horizontal ? m_sssBlurTargets[1].get() : m_sssBlurTargets[0].get();
            }
            rhi::Texture* dest =
                horizontal ? m_sssBlurTargets[0].get() : m_sssBlurTargets[1].get();

            std::unique_ptr<rhi::DescriptorSet> set =
                m_device->CreateDescriptorSet(*m_sssBlurSetLayout);
            set->BindTexture(0, *source, 0u);
            set->BindSampler(1, *m_sssSampler);
            set->BindTexture(2, m_gbuffer.GetPosition(), 0u);
            set->BindTexture(3, *m_sssLut, 0u);
            set->BindSampler(4, *m_sssSampler);
            set->BindStorageImage(5, *dest);
            m_sssBlurSets.push_back(std::move(set));
        }
    }

    if (m_compositeSet == nullptr)
    {
        m_compositeSet = m_device->CreateDescriptorSet(*m_compositeSetLayout);
        m_compositeSet->BindTexture(0, *m_lightingColor, 0u);
        m_compositeSet->BindTexture(1, *m_sssSource, 0u);
        m_compositeSet->BindTexture(2, *m_sssBlurTargets[1], 0u);
        m_compositeSet->BindSampler(3, *m_sssSampler);
    }
}

void DeferredRenderer::RecordSssPasses(rhi::CommandList& commandList, const FrameView& view)
{
    if (m_sssBlurPipeline == nullptr || !m_hasSssMaterials)
    {
        return;
    }

    if (m_sssBlurSets.empty())
    {
        return;
    }

    const std::uint32_t iterations =
        std::clamp(m_sssSettings.iterations, 1u, 3u);
    const float radius =
        std::clamp(m_sssSettings.radiusPixels, 1.0f, kSubsurfaceLutRadiusMaxPx);
    const float logMin = std::log(kSubsurfaceLutRadiusMinPx);
    const float logMax = std::log(kSubsurfaceLutRadiusMaxPx);
    const float logRange = std::max(logMax - logMin, 1.0e-4f);

    commandList.BeginGpuZone("SssBlur");
    commandList.BindPipeline(*m_sssBlurPipeline);
    for (std::uint32_t pass = 0; pass < 2u * iterations; ++pass)
    {
        if (pass > 0)
        {
            commandList.ShaderMemoryBarrier(
                rhi::ShaderStageMask::Compute,
                rhi::ShaderStageMask::Compute);
        }

        const bool horizontal = (pass % 2u) == 0u;
        const std::uint32_t sourceWidth =
            pass == 0 ? m_gbuffer.GetWidth() : m_sssHalfWidth;
        const std::uint32_t sourceHeight =
            pass == 0 ? m_gbuffer.GetHeight() : m_sssHalfHeight;
        const float spacing =
            std::max(radius / 4.0f, 0.5f) * static_cast<float>(1u + pass / 2u);

        SssBlurPushConstants push{};
        push.params = glm::vec4{
            radius,
            std::max(m_sssSettings.depthTolerance, 0.01f),
            spacing,
            0.0f};
        push.fullSize = glm::vec4{
            static_cast<float>(m_gbuffer.GetWidth()),
            static_cast<float>(m_gbuffer.GetHeight()),
            0.0f,
            0.0f};
        push.sourceSize = glm::vec4{
            static_cast<float>(sourceWidth),
            static_cast<float>(sourceHeight),
            0.0f,
            0.0f};
        push.destSize = glm::vec4{
            static_cast<float>(m_sssHalfWidth),
            static_cast<float>(m_sssHalfHeight),
            0.0f,
            0.0f};
        push.direction = horizontal ? glm::vec4{1.0f, 0.0f, 0.0f, 0.0f}
                                    : glm::vec4{0.0f, 1.0f, 0.0f, 0.0f};
        push.cameraPosition = glm::vec4{view.position, 1.0f};
        push.projection = glm::vec4{
            std::tan(std::max(view.fovYRadians, 1.0e-3f) * 0.5f),
            static_cast<float>(m_gbuffer.GetHeight()),
            0.0f,
            0.0f};
        push.lutParams = glm::vec4{1.0f / logRange, -logMin / logRange, 0.0f, 0.0f};

        commandList.BindDescriptorSet(0, *m_sssBlurSets[pass]);
        commandList.SetPushConstants(&push, sizeof(push), rhi::ShaderStageMask::Compute);
        commandList.Dispatch(
            (m_sssHalfWidth + 7u) / 8u,
            (m_sssHalfHeight + 7u) / 8u,
            1);
    }
    commandList.EndGpuZone();
}

bool DeferredRenderer::IsSssActive() const
{
    return m_sssSettings.enabled && m_hasSssMaterials && m_sssBlurPipeline != nullptr;
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
