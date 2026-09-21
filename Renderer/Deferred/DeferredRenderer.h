#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Renderer/BufferManager.h"
#include "Renderer/Deferred/GBuffer.h"
#include "Renderer/RHI/Resources.h"

namespace mukki
{

struct SceneUniform
{
    glm::mat4 viewProjection{1.0f};
    glm::vec4 cameraPosition{0.0f};
    glm::vec4 lightDirection{0.0f};
    glm::vec4 lightColor{1.0f};
    glm::vec4 ambientColor{0.0f};
};

struct ObjectPushConstants
{
    glm::mat4 model{1.0f};
    glm::vec4 albedo{1.0f};
    glm::vec4 params{0.0f};
};

class DeferredRenderer
{
public:
    struct Desc
    {
        std::filesystem::path shaderDirectory{};
    };

    DeferredRenderer() = default;
    ~DeferredRenderer() = default;

    DeferredRenderer(const DeferredRenderer&) = delete;
    DeferredRenderer& operator=(const DeferredRenderer&) = delete;

    bool Init(rhi::Device& device, const Desc& desc);
    void Shutdown();
    void Resize(std::uint32_t width, std::uint32_t height);
    void Render(rhi::CommandList& commandList, const glm::mat4& viewProjection, const glm::vec3& cameraPosition);

    [[nodiscard]] bool IsInitialized() const { return m_initialized; }
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

    bool CreatePipelines(rhi::Device& device, const std::filesystem::path& shaderDirectory);
    bool CreateScene();
    void CreateLightingSets(rhi::Device& device);
    void UpdateUniforms(std::uint32_t frameIndex, const glm::mat4& viewProjection, const glm::vec3& cameraPosition);

    rhi::Device* m_device = nullptr;
    GBuffer m_gbuffer{};
    std::unique_ptr<BufferManager> m_bufferManager{};
    std::vector<std::unique_ptr<rhi::Buffer>> m_uniformBuffers{};
    rhi::Buffer* m_cubeVertexBuffer = nullptr;
    rhi::Buffer* m_cubeIndexBuffer = nullptr;
    rhi::Buffer* m_planeVertexBuffer = nullptr;
    rhi::Buffer* m_planeIndexBuffer = nullptr;
    std::unique_ptr<rhi::Sampler> m_gbufferSampler{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_sceneSetLayout{};
    std::unique_ptr<rhi::DescriptorSetLayout> m_lightingSetLayout{};
    std::vector<std::unique_ptr<rhi::DescriptorSet>> m_sceneSets{};
    std::vector<std::unique_ptr<rhi::DescriptorSet>> m_lightingSets{};
    std::unique_ptr<rhi::Shader> m_gbufferVertexShader{};
    std::unique_ptr<rhi::Shader> m_gbufferFragmentShader{};
    std::unique_ptr<rhi::Shader> m_lightingVertexShader{};
    std::unique_ptr<rhi::Shader> m_lightingFragmentShader{};
    std::unique_ptr<rhi::Pipeline> m_gbufferPipeline{};
    std::unique_ptr<rhi::Pipeline> m_lightingPipeline{};
    std::vector<DrawItem> m_drawItems{};
    SceneUniform m_sceneUniform{};
    bool m_lightingSetsDirty = true;
    bool m_initialized = false;
};

} // namespace mukki
