#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
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
#include "Renderer/Deferred/GBuffer.h"
#include "Renderer/Lighting/ClusterGrid.h"
#include "Renderer/Lighting/Light.h"
#include "Renderer/RHI/Resources.h"

namespace mukki
{

class JobSystem;

struct SceneUniform
{
    glm::mat4 viewProjection{1.0f};
    glm::mat4 view{1.0f};
    glm::vec4 cameraPosition{0.0f};
    glm::vec4 ambientColor{0.0f};
    glm::vec4 clusterParams{0.0f}; // x: near, y: far, z: tile size, w: z slice count
    glm::uvec4 clusterGrid{0u};    // x, y, z grid dimensions, w: light count
    glm::vec4 screenSize{0.0f};    // x: width, y: height
};

struct ClusterUniform
{
    glm::mat4 inverseView{1.0f};
    glm::vec4 cameraParams{0.0f};  // x: near, y: far, z: tan(fovY / 2), w: aspect ratio
    glm::vec4 clusterParams{0.0f}; // x: tile size, y: z slice count
    glm::uvec4 grid{0u};           // x, y, z grid dimensions, w: light count
    glm::uvec4 screen{0u};         // x: width, y: height
};

struct FrameView
{
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::mat4 viewProjection{1.0f};
    glm::vec3 position{0.0f};
    float fovYRadians = 1.0471975512f;
    float aspectRatio = 16.0f / 9.0f;
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;
};

struct ObjectPushConstants
{
    glm::mat4 model{1.0f};
    glm::vec4 baseColorFactor{1.0f};
    glm::vec4 params{0.0f};
    glm::uvec2 textureIndices{0u};
    glm::uvec2 samplerIndices{0u};
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

    // Replaces the lit image with a heatmap of the clustered light counts for debugging.
    void SetClusterDebugView(bool enabled) { m_clusterDebugView = enabled; }
    [[nodiscard]] bool IsClusterDebugViewEnabled() const { return m_clusterDebugView; }

    bool LoadModel(const std::filesystem::path& path);
    void RequestModelLoad(const std::filesystem::path& path);
    bool PollModelLoad();
    void UnloadModel();

    [[nodiscard]] bool IsInitialized() const { return m_initialized; }
    [[nodiscard]] bool IsModelLoading() const { return m_asyncLoader.IsLoading(); }
    [[nodiscard]] bool HasModel() const { return !m_modelDrawItems.empty(); }
    [[nodiscard]] const std::string& GetModelName() const { return m_modelName; }
    [[nodiscard]] std::uint64_t GetModelVertexCount() const { return m_modelVertexCount; }
    [[nodiscard]] std::uint64_t GetModelTriangleCount() const { return m_modelTriangleCount; }

    [[nodiscard]] rhi::Texture& GetAlbedoTexture() { return m_gbuffer.GetAlbedo(); }
    [[nodiscard]] rhi::Texture& GetNormalTexture() { return m_gbuffer.GetNormal(); }
    [[nodiscard]] rhi::Texture& GetPositionTexture() { return m_gbuffer.GetPosition(); }

private:
    struct DrawItem
    {
        rhi::Buffer* vertexBuffer = nullptr;
        rhi::Buffer* indexBuffer = nullptr;
        std::uint32_t indexCount = 0;
        std::uint32_t firstIndex = 0;
        ObjectPushConstants pushConstants{};
    };

    struct Material
    {
        std::unique_ptr<rhi::Texture> baseColorTexture{};
        std::unique_ptr<rhi::Texture> metallicRoughnessTexture{};
    };

    bool CreatePipelines(rhi::Device& device, const std::filesystem::path& shaderDirectory);
    bool CreateScene();
    bool CreateClusterResources(rhi::Device& device);
    void CreateLightingSets(rhi::Device& device);
    void UpdateUniforms(std::uint32_t frameIndex, const FrameView& view);
    void ApplyModelScale();
    bool UploadModel(const std::filesystem::path& path, GltfModel& model);
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
    std::unique_ptr<rhi::DescriptorSet> m_bindlessSet{};
    std::vector<std::unique_ptr<rhi::DescriptorSet>> m_sceneSets{};
    std::vector<std::unique_ptr<rhi::DescriptorSet>> m_lightingSets{};
    std::vector<std::unique_ptr<rhi::DescriptorSet>> m_clusterSets{};
    std::vector<Material> m_materials{};
    std::unique_ptr<rhi::Shader> m_gbufferVertexShader{};
    std::unique_ptr<rhi::Shader> m_gbufferFragmentShader{};
    std::unique_ptr<rhi::Shader> m_lightingVertexShader{};
    std::unique_ptr<rhi::Shader> m_lightingFragmentShader{};
    std::unique_ptr<rhi::Shader> m_clusterCullingShader{};
    std::unique_ptr<rhi::Pipeline> m_gbufferPipeline{};
    std::unique_ptr<rhi::Pipeline> m_lightingPipeline{};
    std::unique_ptr<rhi::Pipeline> m_clusterCullingPipeline{};
    std::vector<DrawItem> m_sceneDrawItems{};
    std::vector<DrawItem> m_modelDrawItems{};
    std::vector<glm::mat4> m_modelBaseTransforms{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_modelVertexBuffers{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_modelIndexBuffers{};
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
