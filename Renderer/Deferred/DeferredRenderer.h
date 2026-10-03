#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Assets/AsyncGltfLoader.h"
#include "Renderer/BufferManager.h"
#include "Renderer/Debug/DebugLines.h"
#include "Renderer/Deferred/GBuffer.h"
#include "Renderer/FrameView.h"
#include "Renderer/GlobalIllumination/GiTypes.h"
#include "Renderer/Lighting/ClusterGrid.h"
#include "Renderer/Lighting/Light.h"
#include "Renderer/Lighting/ShadowCascade.h"
#include "Renderer/RHI/CommandList.h"
#include "Renderer/RHI/Resources.h"

namespace mukki
{

class JobSystem;
struct SceneDescription;

struct SceneUniform
{
    glm::mat4 viewProjection{1.0f};
    glm::mat4 previousViewProjection{1.0f};
    glm::mat4 view{1.0f};
    glm::vec4 cameraPosition{0.0f};
    glm::vec4 ambientColor{0.0f};
    glm::vec4 clusterParams{0.0f}; // x: near, y: far, z: tile size, w: z slice count
    glm::uvec4 clusterGrid{0u};    // x, y, z grid dimensions, w: light count
    glm::vec4 screenSize{0.0f};    // x: width, y: height
    glm::vec4 reflectionParams{0.0f}; // x: ray traced reflection intensity (0 disables)
    glm::vec4 sunDirection{0.0f, -1.0f, 0.0f, 0.0f}; // xyz: direction toward the sun
    glm::vec4 sunColor{0.0f};      // rgb: color * intensity, used by the transparent pass
};

struct ClusterUniform
{
    glm::mat4 inverseView{1.0f};
    glm::vec4 cameraParams{0.0f};  // x: near, y: far, z: tan(fovY / 2), w: aspect ratio
    glm::vec4 clusterParams{0.0f}; // x: tile size, y: z slice count
    glm::uvec4 grid{0u};           // x, y, z grid dimensions, w: light count
    glm::uvec4 screen{0u};         // x: width, y: height
};

struct ObjectPushConstants
{
    glm::mat4 model{1.0f};
    glm::vec4 baseColorFactor{1.0f};
    glm::vec4 params{0.0f};
    glm::uvec2 textureIndices{0u};
    glm::uvec2 samplerIndices{0u};
    std::uint32_t drawIndex = 0;
};

// Per draw item data indexed by ObjectPushConstants::drawIndex.
struct GpuObjectData
{
    glm::mat4 previousModel{1.0f};
    glm::vec4 specular{0.0f};           // x: specularFactor, yzw: specularColorFactor
    glm::uvec4 specularTextureIndices{0u}; // x: specularTexture, y: specularColorTexture
    glm::uvec4 specularSamplerIndices{0u}; // x: specular sampler, y: specular color sampler
    glm::vec4 volume{0.0f};             // x: transmission, y: ior, z: thickness, w: dispersion
    glm::vec4 attenuation{0.0f};        // rgb: attenuation color, a: attenuation distance (0 = none)
    glm::uvec4 volumeTextureIndices{0u}; // x: transmission texture, y: thickness texture
    glm::uvec4 volumeSamplerIndices{0u}; // x: transmission sampler, y: thickness sampler
    glm::vec4 alpha{0.0f};              // x: blend, y: cutoff, z/w: unused
    glm::vec4 sheen{0.0f};              // rgb: sheenColorFactor, a: sheenRoughnessFactor
    glm::uvec4 sheenTextureIndices{0u}; // x: sheenColorTexture, y: sheenRoughnessTexture
    glm::uvec4 sheenSamplerIndices{0u}; // x: sheen color sampler, y: sheen roughness sampler
};

class DeferredRenderer
{
public:
    struct Desc
    {
        std::filesystem::path shaderDirectory{};
        JobSystem* jobSystem = nullptr;
    };

    DeferredRenderer() = default;
    ~DeferredRenderer() = default;

    DeferredRenderer(const DeferredRenderer&) = delete;
    DeferredRenderer& operator=(const DeferredRenderer&) = delete;

    bool Init(rhi::Device& device, const Desc& desc);
    void Shutdown();
    void Resize(std::uint32_t width, std::uint32_t height);
    void Render(rhi::CommandList& commandList, const FrameView& view);

    // Replaces the lights used by the next Render call. Up to kMaxLights are uploaded; additional
    // lights are ignored.
    void SetLights(std::span<const Light> lights);
    [[nodiscard]] std::uint32_t GetLightCount() const;

    // Uniform scale applied to the loaded glTF model about the world origin. The ground plane is
    // not affected.
    void SetModelScale(float scale);
    [[nodiscard]] float GetModelScale() const { return m_modelScale; }

    // Updates the transform of a scene object instantiated by LoadScene. Object indices follow the
    // order of SceneDescription::objects.
    void UpdateSceneObjectTransform(std::size_t objectIndex, const glm::mat4& transform);
    [[nodiscard]] std::size_t GetSceneObjectCount() const { return m_objectInstances.size(); }

    // World space bounds of the instantiated scene objects. Returns false when no object is loaded.
    [[nodiscard]] bool GetSceneBounds(glm::vec3& outMin, glm::vec3& outMax) const;

    // Replaces the lit image with a heatmap of the clustered light counts for debugging.
    void SetClusterDebugView(bool enabled) { m_clusterDebugView = enabled; }
    [[nodiscard]] bool IsClusterDebugViewEnabled() const { return m_clusterDebugView; }

    // World space debug gizmos: wireframe markers for lights and world space GI probes.
    void SetGiProbes(std::span<const GiProbe> probes);
    void SetGizmoView(bool showLights, bool showProbes)
    {
        m_showLightGizmos = showLights;
        m_showProbeGizmos = showProbes;
    }
    [[nodiscard]] bool AreLightGizmosEnabled() const { return m_showLightGizmos; }
    [[nodiscard]] bool AreProbeGizmosEnabled() const { return m_showProbeGizmos; }

    // Directional cascaded shadow map settings. Shadows come from the first directional light that
    // has Light::castShadows set.
    void SetShadowSettings(const ShadowSettings& settings) { m_shadowSettings = settings; }
    [[nodiscard]] const ShadowSettings& GetShadowSettings() const { return m_shadowSettings; }

    // Ray traced diffuse and glossy global illumination settings.
    void SetGiSettings(const GiSettings& settings);
    [[nodiscard]] const GiSettings& GetGiSettings() const { return m_giSettings; }
    [[nodiscard]] bool IsGiActive() const;

    // Ray traced glossy reflection settings. Reflections reuse the GI trace resolution.
    void SetReflectionSettings(const ReflectionSettings& settings);
    [[nodiscard]] const ReflectionSettings& GetReflectionSettings() const
    {
        return m_reflectionSettings;
    }

    bool LoadModel(const std::filesystem::path& path);
    void RequestModelLoad(const std::filesystem::path& path);
    bool PollModelLoad();
    void UnloadModel();

    // Loads every glTF object and light from a scene description. Models referenced more than once
    // are loaded and uploaded only once. Replaces the current models and lights; the ground plane
    // is kept.
    bool LoadScene(const SceneDescription& scene);
    void ClearScene();

    [[nodiscard]] bool IsInitialized() const { return m_initialized; }
    [[nodiscard]] bool IsModelLoading() const { return m_asyncLoader.IsLoading(); }
    [[nodiscard]] bool HasModel() const { return !m_modelDrawItems.empty(); }
    [[nodiscard]] const std::string& GetModelName() const { return m_modelName; }
    [[nodiscard]] std::uint64_t GetModelVertexCount() const { return m_modelVertexCount; }
    [[nodiscard]] std::uint64_t GetModelTriangleCount() const { return m_modelTriangleCount; }

    [[nodiscard]] rhi::Texture& GetAlbedoTexture() { return m_gbuffer.GetAlbedo(); }
    [[nodiscard]] rhi::Texture& GetNormalTexture() { return m_gbuffer.GetNormal(); }
    [[nodiscard]] rhi::Texture& GetPositionTexture() { return m_gbuffer.GetPosition(); }
    [[nodiscard]] rhi::Texture& GetMotionTexture() { return m_gbuffer.GetMotion(); }

private:
    struct DrawItem
    {
        rhi::Buffer* vertexBuffer = nullptr;
        rhi::Buffer* indexBuffer = nullptr;
        std::uint32_t indexCount = 0;
        std::uint32_t firstIndex = 0;
        ObjectPushConstants pushConstants{};
        const rhi::AccelerationStructure* blas = nullptr;
        std::uint32_t vertexBufferIndex = 0;
        std::uint32_t indexBufferIndex = 0;
        // KHR_materials_specular material data (see GpuObjectData).
        glm::vec4 specular{0.0f};
        glm::uvec4 specularTextureIndices{0u};
        glm::uvec4 specularSamplerIndices{0u};
        // Glass / transparency material data (see GpuObjectData).
        glm::vec4 volume{0.0f};
        glm::vec4 attenuation{0.0f};
        glm::uvec4 volumeTextureIndices{0u};
        glm::uvec4 volumeSamplerIndices{0u};
        glm::vec4 alpha{0.0f};
        bool alphaBlend = false;
        glm::vec4 sheen{0.0f};
        glm::uvec4 sheenTextureIndices{0u};
        glm::uvec4 sheenSamplerIndices{0u};
    };

    // Per draw item data read by ray traced effects (global illumination color bleeding).
    struct GpuGeometryData
    {
        glm::vec4 baseColorFactor{1.0f}; // rgb: albedo, a: unused
        glm::uvec4 bufferIndices{0u};    // x: vertex buffer, y: index buffer, z: vertex stride
        glm::uvec4 textureIndices{0u};   // x: base color texture, y: metallic-roughness texture
        glm::uvec4 samplerIndices{0u};   // x: base color sampler, y: metallic-roughness sampler
        glm::vec4 params{0.0f};          // x: roughness, y: metalness, z: alpha mode (0/1/2), w: alpha cutoff
        glm::vec4 volume{0.0f};          // x: transmission, y: ior, z: thickness, w: dispersion
        glm::vec4 attenuation{0.0f};     // rgb: attenuation color, a: attenuation distance (0 = none)
        glm::uvec4 volumeTextureIndices{0u}; // x: transmission texture, y: thickness texture
        glm::uvec4 volumeSamplerIndices{0u}; // x: transmission sampler, y: thickness sampler
        glm::vec4 sheen{0.0f};               // rgb: sheenColorFactor, a: sheenRoughnessFactor
        glm::uvec4 sheenTextureIndices{0u};  // x: sheenColorTexture, y: sheenRoughnessTexture
        glm::uvec4 sheenSamplerIndices{0u};
    };

    inline static constexpr std::uint32_t kMaxObjectData = 2048;

    struct Material
    {
        std::unique_ptr<rhi::Texture> baseColorTexture{};
        std::unique_ptr<rhi::Texture> metallicRoughnessTexture{};
        std::unique_ptr<rhi::Texture> specularTexture{};
        std::unique_ptr<rhi::Texture> specularColorTexture{};
        std::unique_ptr<rhi::Texture> transmissionTexture{};
        std::unique_ptr<rhi::Texture> thicknessTexture{};
        std::unique_ptr<rhi::Texture> sheenColorTexture{};
        std::unique_ptr<rhi::Texture> sheenRoughnessTexture{};
    };

    struct SceneObjectBounds
    {
        glm::vec3 center{0.0f};
        float radius = 0.0f;
    };

    struct ModelResources
    {
        std::string name{};
        std::vector<Material> materials{};
        std::vector<std::unique_ptr<rhi::Buffer>> vertexBuffers{};
        std::vector<std::unique_ptr<rhi::Buffer>> indexBuffers{};
        std::vector<std::uint32_t> vertexBufferIndices{};
        std::vector<std::uint32_t> indexBufferIndices{};
        std::vector<std::unique_ptr<rhi::AccelerationStructure>> accelerationStructures{};
        std::vector<DrawItem> drawItems{};
        SceneObjectBounds fittedBounds{};
        std::uint64_t vertexCount = 0;
        std::uint64_t triangleCount = 0;
    };

    struct SceneObjectInstance
    {
        std::size_t modelIndex = 0;
        std::size_t firstDrawItem = 0;
        std::size_t drawItemCount = 0;
        glm::mat4 transform{1.0f};
        SceneObjectBounds bounds{};
    };

    struct ShadowPushConstants
    {
        ObjectPushConstants object{};
        std::uint32_t viewIndex = 0;
    };

    struct ShadowUniform
    {
        glm::mat4 viewProjections[kMaxShadowViews]{}; // cascades, spot maps, point cube faces
        glm::vec4 cascadeSplits{0.0f}; // x, y, z: split distances between cascades
        glm::vec4 params{0.0f};        // x: depthBias, y: normalBias, z: distance, w: enabled
        glm::vec4 rtParams{0.0f};      // x: ray traced sun shadows enabled, y/z/w: unused
    };

    // Matches Renderer/Shaders/rt_shadows.comp.slang.
    struct RtShadowUniform
    {
        glm::vec4 screenSize{0.0f};            // x: width, y: height
        glm::vec4 sunDirectionIntensity{0.0f}; // xyz: toward the sun, w: intensity
        glm::vec4 params{0.0f}; // x: samples, y: angular radius (radians), z: frame counter, w: normal bias
    };

    // One shadow map render view: a cascade layer, a spot layer or a point cube face (rendered
    // into a tile of the point shadow atlas).
    struct ShadowView
    {
        std::uint32_t viewIndex = 0;
        rhi::Texture* texture = nullptr;
        std::uint32_t layer = 0;
        rhi::RenderArea area{};
        std::vector<const DrawItem*> items{};
    };

    struct ShadowCullItem
    {
        const DrawItem* item = nullptr;
        glm::vec3 center{0.0f};
        float radius = 0.0f;
    };

    struct GiAtrousPushConstants
    {
        std::uint32_t signal = 0;
        std::uint32_t iteration = 0;
    };

    static constexpr std::uint32_t kMaxAtrousIterations = 8;

    struct GiFrameResources
    {
        std::unique_ptr<rhi::Texture> diffuseNoisy{};
        std::unique_ptr<rhi::Texture> specularNoisy{};
        std::unique_ptr<rhi::Texture> hitDistance{};
        std::unique_ptr<rhi::Texture> diffuseHistory[2]{};
        std::unique_ptr<rhi::Texture> specularHistory[2]{};
        std::unique_ptr<rhi::Texture> diffuseMoments[2]{};
        std::unique_ptr<rhi::Texture> specularMoments[2]{};
        std::unique_ptr<rhi::Texture> diffuseFiltered{};
        std::unique_ptr<rhi::Texture> specularFiltered{};
        std::unique_ptr<rhi::Texture> indirect{};
        std::unique_ptr<rhi::Buffer> uniformBuffer{};
        std::unique_ptr<rhi::DescriptorSet> traceSet{};
        std::unique_ptr<rhi::DescriptorSet> temporalSet{};
        // One descriptor set per iteration and signal: a set must never be updated after binding.
        std::array<std::unique_ptr<rhi::DescriptorSet>, kMaxAtrousIterations * 2> atrousSets{};
        std::unique_ptr<rhi::DescriptorSet> upsampleSet{};
        std::uint32_t historyParity = 0;
        bool historyValid = false;

        // Ray traced reflections share the GI trace resolution and the GI a-trous pipeline.
        std::unique_ptr<rhi::Texture> reflectionNoisy{};
        std::unique_ptr<rhi::Texture> reflectionHitDistance{};
        std::unique_ptr<rhi::Texture> reflectionHistory[2]{};
        std::unique_ptr<rhi::Texture> reflectionMoments[2]{};
        std::unique_ptr<rhi::Texture> reflectionFiltered{};
        std::unique_ptr<rhi::Texture> reflections{};
        std::unique_ptr<rhi::DescriptorSet> reflectionTraceSet{};
        std::unique_ptr<rhi::DescriptorSet> reflectionTemporalSet{};
        std::array<std::unique_ptr<rhi::DescriptorSet>, kMaxAtrousIterations> reflectionAtrousSets{};
        std::unique_ptr<rhi::DescriptorSet> reflectionUpsampleSet{};
        std::uint32_t reflectionParity = 0;
        bool reflectionValid = false;
    };

    bool LoadShaders(rhi::Device& device, const std::filesystem::path& shaderDirectory);
    bool CreatePipelines(rhi::Device& device);
    bool CreateScene();
    bool CreateClusterResources(rhi::Device& device);
    bool CreateShadowResources(rhi::Device& device);
    void CreateLightingSets(rhi::Device& device);
    void UpdateUniforms(std::uint32_t frameIndex, const FrameView& view);
    void UpdateShadowUniforms(std::uint32_t frameIndex, const FrameView& view);
    [[nodiscard]] const Light* FindShadowLight() const;
    void BuildShadowCullItems();
    void BuildShadowViews();
    [[nodiscard]] std::vector<const DrawItem*> CollectShadowItems(
        const glm::vec3& lightPosition,
        float range) const;
    void RecordShadowViews(std::uint32_t frameIndex);
    void RecordShadowView(
        std::uint32_t frameIndex,
        std::uint32_t shadowViewIndex,
        const std::vector<const DrawItem*>& items,
        rhi::CommandList& secondary,
        const rhi::RenderingDesc& desc);
    void ApplyModelScale();
    void UpdateObjectData(std::uint32_t frameIndex);
    void BuildAccelerationStructures(rhi::CommandList& commandList, std::uint32_t frameIndex);
    bool CreateGiResources(rhi::Device& device);
    void UpdateGiUniform(std::uint32_t frameIndex, const FrameView& view);
    void BindGiDescriptors(std::uint32_t frameIndex);
    void RecordGiPasses(rhi::CommandList& commandList, std::uint32_t frameIndex);
    bool CreateRtShadowResources(rhi::Device& device);
    void UpdateRtShadowUniform(std::uint32_t frameIndex);
    void RecordRtShadowPass(rhi::CommandList& commandList, std::uint32_t frameIndex);
    [[nodiscard]] bool IsRtShadowActive() const;
    bool CreateGiProbeResources(rhi::Device& device);
    void RecordGiProbes(rhi::CommandList& commandList, std::uint32_t frameIndex);
    [[nodiscard]] bool IsProbeGiActive() const;
    [[nodiscard]] bool IsPerPixelGiActive() const;
    bool CreateReflectionResources(rhi::Device& device);
    void UpdateReflectionUniform(std::uint32_t frameIndex);
    void BindReflectionDescriptors(std::uint32_t frameIndex);
    void RecordReflectionPasses(rhi::CommandList& commandList, std::uint32_t frameIndex);
    [[nodiscard]] bool IsReflectionActive() const;
    void RecordDebugLines(rhi::CommandList& commandList, const FrameView& view);
    [[nodiscard]] std::optional<std::size_t> UploadModel(const std::filesystem::path& path, GltfModel& model);
    [[nodiscard]] std::optional<std::size_t> LoadModelResources(const std::filesystem::path& path);
    void AddModelInstance(std::size_t modelIndex, const glm::mat4& transform);
    void ClearModelInstances();
    void ClearModels();
    [[nodiscard]] std::uint32_t GetSamplerIndex(rhi::AddressMode addressModeU, rhi::AddressMode addressModeV);
    [[nodiscard]] std::uint32_t RecordGBufferSecondaries(
        std::uint32_t frameIndex,
        const rhi::RenderingDesc& desc,
        const std::vector<const DrawItem*>& items);
    void RecordGBufferBatch(
        std::uint32_t frameIndex,
        const std::vector<const DrawItem*>& items,
        std::size_t begin,
        std::size_t end,
        rhi::CommandList& secondary,
        const rhi::RenderingDesc& desc);

    rhi::Device* m_device = nullptr;
    JobSystem* m_jobSystem = nullptr;
    AsyncGltfLoader m_asyncLoader{};
    GBuffer m_gbuffer{};
    std::unique_ptr<BufferManager> m_bufferManager{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_uniformBuffers{};
    rhi::Buffer* m_planeVertexBuffer = nullptr;
    rhi::Buffer* m_planeIndexBuffer = nullptr;
    std::unique_ptr<rhi::Sampler> m_gbufferSampler{};
    std::vector<std::unique_ptr<rhi::Sampler>> m_bindlessSamplers{};
    std::unordered_map<std::uint32_t, std::uint32_t> m_samplerCache{};
    std::unique_ptr<rhi::Texture> m_whiteTexture{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_sceneSetLayout{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_bindlessSetLayout{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_lightingSetLayout{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_clusterSetLayout{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_shadowSetLayout{};
    std::unique_ptr<rhi::Texture> m_shadowMap{};
    std::unique_ptr<rhi::Texture> m_spotShadowMap{};
    std::unique_ptr<rhi::Texture> m_pointShadowMap{};
    std::unique_ptr<rhi::Sampler> m_shadowSampler{};
    std::vector<std::unique_ptr<rhi::DescriptorSet>> m_shadowSets{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_shadowUniformBuffers{};
    std::vector<std::vector<std::unique_ptr<rhi::CommandList>>> m_shadowSecondaries{};
    std::vector<ShadowView> m_shadowViews{};
    std::vector<ShadowCullItem> m_shadowCullItems{};
    std::vector<Light> m_spotShadowLights{};
    std::vector<Light> m_pointShadowLights{};
    std::unique_ptr<rhi::DescriptorSet> m_bindlessSet{};
    std::vector<std::unique_ptr<rhi::DescriptorSet>> m_sceneSets{};
    std::vector<std::unique_ptr<rhi::DescriptorSet>> m_lightingSets{};
    std::vector<std::unique_ptr<rhi::DescriptorSet>> m_clusterSets{};
    std::unique_ptr<rhi::Shader> m_gbufferVertexShader{};
    std::unique_ptr<rhi::Shader> m_gbufferFragmentShader{};
    std::unique_ptr<rhi::Shader> m_transparentFragmentShader{};
    std::unique_ptr<rhi::Shader> m_lightingVertexShader{};
    std::unique_ptr<rhi::Shader> m_lightingFragmentShader{};
    std::unique_ptr<rhi::Shader> m_clusterCullingShader{};
    std::unique_ptr<rhi::Shader> m_shadowVertexShader{};
    std::unique_ptr<rhi::Shader> m_shadowFragmentShader{};
    std::unique_ptr<rhi::Pipeline> m_gbufferPipeline{};
    std::unique_ptr<rhi::Pipeline> m_transparentPipeline{};
    std::unique_ptr<rhi::Pipeline> m_lightingPipeline{};
    std::unique_ptr<rhi::Pipeline> m_clusterCullingPipeline{};
    std::unique_ptr<rhi::Pipeline> m_shadowPipeline{};
    std::vector<DrawItem> m_sceneDrawItems{};
    std::vector<DrawItem> m_modelDrawItems{};
    std::vector<glm::mat4> m_modelBaseTransforms{};
    std::vector<std::unique_ptr<ModelResources>> m_models{};
    std::unique_ptr<rhi::AccelerationStructure> m_planeBlas{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_objectDataBuffers{};
    std::vector<glm::mat4> m_previousModels{};
    std::vector<GiFrameResources> m_gi{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_giTraceSetLayout{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_giTemporalSetLayout{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_giAtrousSetLayout{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_giUpsampleSetLayout{};
    std::unique_ptr<rhi::Shader> m_giTraceShader{};
    std::unique_ptr<rhi::Shader> m_giTemporalShader{};
    std::unique_ptr<rhi::Shader> m_giAtrousShader{};
    std::unique_ptr<rhi::Shader> m_giUpsampleShader{};
    std::unique_ptr<rhi::Pipeline> m_giTracePipeline{};
    std::unique_ptr<rhi::Pipeline> m_giTemporalPipeline{};
    std::unique_ptr<rhi::Pipeline> m_giAtrousPipeline{};
    std::unique_ptr<rhi::Pipeline> m_giUpsamplePipeline{};
    GiUniform m_giUniform{};
    GiSettings m_giSettings{};
    ReflectionSettings m_reflectionSettings{};
    std::uint32_t m_giFrameCounter = 0;
    std::unique_ptr<rhi::Texture> m_blackTexture{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_reflectionTraceSetLayout{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_reflectionTemporalSetLayout{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_reflectionUpsampleSetLayout{};
    std::unique_ptr<rhi::Shader> m_rtReflectionShader{};
    std::unique_ptr<rhi::Shader> m_reflectionTemporalShader{};
    std::unique_ptr<rhi::Shader> m_reflectionUpsampleShader{};
    std::unique_ptr<rhi::Pipeline> m_rtReflectionPipeline{};
    std::unique_ptr<rhi::Pipeline> m_reflectionTemporalPipeline{};
    std::unique_ptr<rhi::Pipeline> m_reflectionUpsamplePipeline{};
    std::unique_ptr<rhi::Shader> m_rtShadowShader{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_rtShadowSetLayout{};
    std::unique_ptr<rhi::Pipeline> m_rtShadowPipeline{};
    std::unique_ptr<rhi::Texture> m_rtSunShadowMask{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_rtShadowUniformBuffers{};
    std::vector<std::unique_ptr<rhi::DescriptorSet>> m_rtShadowSets{};
    RtShadowUniform m_rtShadowUniform{};
    std::unique_ptr<rhi::Shader> m_giProbeShader{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_giProbeSetLayout{};
    std::unique_ptr<rhi::Pipeline> m_giProbePipeline{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_giProbeUniformBuffers{};
    std::vector<std::unique_ptr<rhi::DescriptorSet>> m_giProbeSets{};
    std::unique_ptr<rhi::Buffer> m_giProbeBuffer{};
    std::unique_ptr<rhi::Buffer> m_giProbeDummyBuffer{};
    std::vector<GpuGiProbe> m_giProbeData{};
    bool m_giProbeDataDirty = false;
    bool m_probeGiActiveLastFrame = false;
    GiUniform m_giProbeUniform{};
    std::unique_ptr<rhi::Shader> m_debugLineVertexShader{};
    std::unique_ptr<rhi::Shader> m_debugLineFragmentShader{};
    std::unique_ptr<rhi::Pipeline> m_debugLinePipeline{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_debugLineBuffers{};
    std::size_t m_debugLineBufferCapacity = 0;
    std::vector<GiProbe> m_giProbes{};
    bool m_showLightGizmos = true;
    bool m_showProbeGizmos = true;
    glm::mat4 m_inverseViewProjection{1.0f};
    std::vector<std::unique_ptr<rhi::AccelerationStructure>> m_tlas{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_geometryDataBuffers{};
    std::vector<GpuGeometryData> m_geometryData{};
    std::vector<std::size_t> m_tlasInstanceCounts{};
    std::uint32_t m_nextBindlessBuffer = 0;
    bool m_blasDirty = false;
    std::unordered_map<std::string, std::size_t> m_modelCache{};
    std::vector<SceneObjectInstance> m_objectInstances{};
    std::vector<std::vector<std::unique_ptr<rhi::CommandList>>> m_gbufferSecondaries{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_lightBuffers{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_clusterUniformBuffers{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_clusterCountBuffers{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_clusterIndexBuffers{};
    std::vector<Light> m_lights{};
    std::vector<GpuLight> m_gpuLights{};
    std::vector<std::uint32_t> m_zeroClusterCounts{};
    SceneUniform m_sceneUniform{};
    ClusterUniform m_clusterUniform{};
    ShadowUniform m_shadowUniform{};
    ShadowSettings m_shadowSettings{};
    ClusterGrid m_clusterGrid{};
    std::string m_modelName{};
    std::uint64_t m_modelVertexCount = 0;
    std::uint64_t m_modelTriangleCount = 0;
    float m_modelScale = 1.0f;
    std::uint32_t m_secondaryBatchCount = 1;
    bool m_lightingSetsDirty = true;
    bool m_clusterDebugView = false;
    bool m_initialized = false;
};

} // namespace mukki
