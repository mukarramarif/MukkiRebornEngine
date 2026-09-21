#include "Renderer/Deferred/DeferredRenderer.h"

#include "Renderer/RHI/Device.h"
#include "Utilities/Assert.h"
#include "Utilities/Log.h"

#include <algorithm>
#include <cstddef>

#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>

namespace mukki
{
namespace
{

struct Vertex
{
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f};
};

constexpr Vertex kCubeVertices[] = {
    {{0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},
    {{-0.5f, -0.5f, 0.5f}, {-1.0f, 0.0f, 0.0f}},
    {{-0.5f, 0.5f, 0.5f}, {-1.0f, 0.0f, 0.0f}},
    {{-0.5f, 0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}},
    {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}},
    {{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, -0.5f, 0.5f}, {0.0f, -1.0f, 0.0f}},
    {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}},
    {{0.5f, -0.5f, 0.5f}, {0.0f, -1.0f, 0.0f}},
    {{-0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
    {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
};

constexpr std::uint32_t kCubeIndices[] = {
    0,  1,  2,  0,  2,  3,
    4,  5,  6,  4,  6,  7,
    8,  9,  10, 8,  10, 11,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 21, 22, 20, 22, 23,
};

constexpr Vertex kPlaneVertices[] = {
    {{-10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}},
    {{10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}},
    {{10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}},
    {{-10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}},
};

constexpr std::uint32_t kPlaneIndices[] = {0, 1, 2, 0, 2, 3};

} // namespace

bool DeferredRenderer::Init(rhi::Device& device, const Desc& desc)
{
    MRE_ASSERT(!m_initialized, "DeferredRenderer::Init called twice");
    MRE_ASSERT(!desc.shaderDirectory.empty(), "DeferredRenderer requires a shader directory");

    m_device = &device;

    const rhi::Extent2D extent = device.GetSwapchainInfo().extent;
    if (!m_gbuffer.Init(device, std::max(1u, extent.width), std::max(1u, extent.height)))
    {
        return false;
    }

    m_bufferManager = std::make_unique<BufferManager>(device);
    m_gbufferSampler = device.CreateSampler(rhi::SamplerDesc{});

    const std::vector<rhi::DescriptorBinding> sceneBindings{
        rhi::DescriptorBinding{0, rhi::DescriptorType::UniformBuffer, rhi::ShaderStageMask::VertexFragment},
    };
    const std::vector<rhi::DescriptorBinding> lightingBindings{
        rhi::DescriptorBinding{0, rhi::DescriptorType::UniformBuffer, rhi::ShaderStageMask::VertexFragment},
        rhi::DescriptorBinding{1, rhi::DescriptorType::CombinedImageSampler, rhi::ShaderStageMask::Fragment},
        rhi::DescriptorBinding{2, rhi::DescriptorType::CombinedImageSampler, rhi::ShaderStageMask::Fragment},
        rhi::DescriptorBinding{3, rhi::DescriptorType::CombinedImageSampler, rhi::ShaderStageMask::Fragment},
    };
    m_sceneSetLayout = device.CreateDescriptorSetLayout(sceneBindings);
    m_lightingSetLayout = device.CreateDescriptorSetLayout(lightingBindings);

    const std::uint32_t framesInFlight = device.GetFramesInFlight();
    m_uniformBuffers.reserve(framesInFlight);
    m_sceneSets.reserve(framesInFlight);
    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        m_uniformBuffers.push_back(device.CreateBuffer(
            rhi::BufferDesc{sizeof(SceneUniform), rhi::BufferUsage::Uniform, true},
            &m_sceneUniform,
            sizeof(SceneUniform)));

        std::unique_ptr<rhi::DescriptorSet> sceneSet = device.CreateDescriptorSet(*m_sceneSetLayout);
        sceneSet->BindBuffer(0, *m_uniformBuffers.back());
        m_sceneSets.push_back(std::move(sceneSet));
    }

    if (!CreateScene() || !CreatePipelines(device, desc.shaderDirectory))
    {
        Shutdown();
        return false;
    }

    m_sceneUniform.lightDirection = glm::vec4(glm::normalize(glm::vec3(-0.4f, -1.0f, -0.3f)), 0.0f);
    m_sceneUniform.lightColor = glm::vec4(1.0f, 0.97f, 0.92f, 3.0f);
    m_sceneUniform.ambientColor = glm::vec4(0.05f, 0.06f, 0.09f, 1.0f);

    m_initialized = true;
    MRE_INFO("Deferred renderer initialized ({}x{})", m_gbuffer.GetWidth(), m_gbuffer.GetHeight());
    return true;
}

void DeferredRenderer::Shutdown()
{
    m_lightingSets.clear();
    m_sceneSets.clear();
    m_lightingPipeline.reset();
    m_gbufferPipeline.reset();
    m_lightingFragmentShader.reset();
    m_lightingVertexShader.reset();
    m_gbufferFragmentShader.reset();
    m_gbufferVertexShader.reset();
    m_drawItems.clear();
    m_lightingSetLayout.reset();
    m_sceneSetLayout.reset();
    m_gbufferSampler.reset();
    m_planeIndexBuffer = nullptr;
    m_planeVertexBuffer = nullptr;
    m_cubeIndexBuffer = nullptr;
    m_cubeVertexBuffer = nullptr;
    m_uniformBuffers.clear();
    m_bufferManager.reset();
    m_gbuffer.Shutdown();
    m_device = nullptr;
    m_lightingSetsDirty = true;
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
    m_lightingSetsDirty = true;

    if (!m_gbuffer.Resize(*m_device, width, height))
    {
        MRE_CRITICAL("Failed to resize the G-buffer");
    }
}

void DeferredRenderer::Render(
    rhi::CommandList& commandList,
    const glm::mat4& viewProjection,
    const glm::vec3& cameraPosition)
{
    MRE_ASSERT(m_initialized, "DeferredRenderer::Render called before DeferredRenderer::Init");
    MRE_ASSERT(m_device != nullptr, "DeferredRenderer requires a device");

    const std::uint32_t frameIndex = m_device->GetFrameIndex();
    MRE_ASSERT(frameIndex < m_sceneSets.size(), "Frame index out of range");

    UpdateUniforms(frameIndex, viewProjection, cameraPosition);

    rhi::RenderingDesc gbufferDesc{};
    gbufferDesc.colorAttachments = {
        rhi::ColorAttachment{&m_gbuffer.GetAlbedo(), rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f}, true},
        rhi::ColorAttachment{&m_gbuffer.GetNormal(), rhi::ClearColor{0.5f, 0.5f, 1.0f, 1.0f}, true},
        rhi::ColorAttachment{&m_gbuffer.GetPosition(), rhi::ClearColor{0.0f, 0.0f, 0.0f, 0.0f}, true},
    };
    gbufferDesc.depthAttachment = rhi::DepthAttachment{&m_gbuffer.GetDepth(), 1.0f, true};

    commandList.BeginRendering(gbufferDesc);
    commandList.BindPipeline(*m_gbufferPipeline);
    commandList.BindDescriptorSet(0, *m_sceneSets[frameIndex]);
    for (const DrawItem& item : m_drawItems)
    {
        commandList.BindVertexBuffer(*item.vertexBuffer);
        commandList.BindIndexBuffer(*item.indexBuffer);
        commandList.SetPushConstants(
            &item.pushConstants,
            sizeof(ObjectPushConstants),
            rhi::ShaderStageMask::VertexFragment);
        commandList.DrawIndexed(item.indexCount, item.firstIndex);
    }
    commandList.EndRendering();

    if (m_lightingSetsDirty)
    {
        CreateLightingSets(*m_device);
        m_lightingSetsDirty = false;
    }

    commandList.BeginSwapchainRendering(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f}, true);
    commandList.BindPipeline(*m_lightingPipeline);
    commandList.BindDescriptorSet(0, *m_lightingSets[frameIndex]);
    commandList.Draw(3);
    commandList.EndSwapchainRendering();
}

bool DeferredRenderer::CreatePipelines(rhi::Device& device, const std::filesystem::path& shaderDirectory)
{
    m_gbufferVertexShader = device.CreateShader(
        rhi::ShaderStage::Vertex,
        (shaderDirectory / "gbuffer.vert.spv").string());
    m_gbufferFragmentShader = device.CreateShader(
        rhi::ShaderStage::Fragment,
        (shaderDirectory / "gbuffer.frag.spv").string());
    m_lightingVertexShader = device.CreateShader(
        rhi::ShaderStage::Vertex,
        (shaderDirectory / "lighting.vert.spv").string());
    m_lightingFragmentShader = device.CreateShader(
        rhi::ShaderStage::Fragment,
        (shaderDirectory / "lighting.frag.spv").string());

    if (m_gbufferVertexShader == nullptr || m_gbufferFragmentShader == nullptr ||
        m_lightingVertexShader == nullptr || m_lightingFragmentShader == nullptr)
    {
        MRE_CRITICAL("Failed to load the deferred shaders from '{}'", shaderDirectory.string());
        return false;
    }

    rhi::VertexLayout vertexLayout{};
    vertexLayout.AddAttribute(0, rhi::Format::R32G32B32Sfloat);
    vertexLayout.AddAttribute(1, rhi::Format::R32G32B32Sfloat);

    rhi::PipelineDesc gbufferPipelineDesc{};
    gbufferPipelineDesc.vertexShader = m_gbufferVertexShader.get();
    gbufferPipelineDesc.fragmentShader = m_gbufferFragmentShader.get();
    gbufferPipelineDesc.vertexLayout = vertexLayout;
    gbufferPipelineDesc.depthTest = true;
    gbufferPipelineDesc.depthWrite = true;
    gbufferPipelineDesc.blend = false;
    gbufferPipelineDesc.cullMode = rhi::CullMode::None;
    gbufferPipelineDesc.pushConstantSize = sizeof(ObjectPushConstants);
    gbufferPipelineDesc.pushConstantStages = rhi::ShaderStageMask::VertexFragment;
    gbufferPipelineDesc.descriptorSetLayouts = {m_sceneSetLayout.get()};
    gbufferPipelineDesc.colorFormats = {
        rhi::Format::R8G8B8A8Unorm,
        rhi::Format::R8G8B8A8Unorm,
        rhi::Format::R16G16B16A16Sfloat};
    gbufferPipelineDesc.depthFormat = rhi::Format::D32Sfloat;
    m_gbufferPipeline = device.CreatePipeline(gbufferPipelineDesc);

    rhi::PipelineDesc lightingPipelineDesc{};
    lightingPipelineDesc.vertexShader = m_lightingVertexShader.get();
    lightingPipelineDesc.fragmentShader = m_lightingFragmentShader.get();
    lightingPipelineDesc.depthTest = false;
    lightingPipelineDesc.depthWrite = false;
    lightingPipelineDesc.blend = false;
    lightingPipelineDesc.cullMode = rhi::CullMode::None;
    lightingPipelineDesc.descriptorSetLayouts = {m_lightingSetLayout.get()};
    lightingPipelineDesc.colorFormats = {device.GetSwapchainInfo().format};
    m_lightingPipeline = device.CreatePipeline(lightingPipelineDesc);

    return m_gbufferPipeline != nullptr && m_lightingPipeline != nullptr;
}

bool DeferredRenderer::CreateScene()
{
    MRE_ASSERT(m_bufferManager != nullptr, "DeferredRenderer requires a buffer manager");

    m_cubeVertexBuffer = &m_bufferManager->CreateVertexBuffer(kCubeVertices, sizeof(kCubeVertices));
    m_cubeIndexBuffer = &m_bufferManager->CreateIndexBuffer(kCubeIndices, sizeof(kCubeIndices));
    m_planeVertexBuffer = &m_bufferManager->CreateVertexBuffer(kPlaneVertices, sizeof(kPlaneVertices));
    m_planeIndexBuffer = &m_bufferManager->CreateIndexBuffer(kPlaneIndices, sizeof(kPlaneIndices));

    const auto makeItem = [](rhi::Buffer* vertexBuffer,
                             rhi::Buffer* indexBuffer,
                             std::uint32_t indexCount,
                             const glm::vec3& position,
                             const glm::vec4& albedo,
                             float roughness,
                             float metalness) {
        DrawItem item{};
        item.vertexBuffer = vertexBuffer;
        item.indexBuffer = indexBuffer;
        item.indexCount = indexCount;
        item.firstIndex = 0;
        item.pushConstants.model = glm::translate(glm::mat4{1.0f}, position);
        item.pushConstants.albedo = albedo;
        item.pushConstants.params = glm::vec4{roughness, metalness, 0.0f, 0.0f};
        return item;
    };

    constexpr std::uint32_t cubeIndexCount = static_cast<std::uint32_t>(std::size(kCubeIndices));
    constexpr std::uint32_t planeIndexCount = static_cast<std::uint32_t>(std::size(kPlaneIndices));

    m_drawItems.push_back(makeItem(
        m_planeVertexBuffer,
        m_planeIndexBuffer,
        planeIndexCount,
        glm::vec3{0.0f, 0.0f, 0.0f},
        glm::vec4{0.35f, 0.35f, 0.38f, 1.0f},
        0.9f,
        0.0f));
    m_drawItems.push_back(makeItem(
        m_cubeVertexBuffer,
        m_cubeIndexBuffer,
        cubeIndexCount,
        glm::vec3{-2.5f, 0.5f, 0.0f},
        glm::vec4{0.85f, 0.25f, 0.25f, 1.0f},
        0.35f,
        0.0f));
    m_drawItems.push_back(makeItem(
        m_cubeVertexBuffer,
        m_cubeIndexBuffer,
        cubeIndexCount,
        glm::vec3{0.0f, 0.5f, -2.5f},
        glm::vec4{0.25f, 0.8f, 0.35f, 1.0f},
        0.6f,
        0.0f));
    m_drawItems.push_back(makeItem(
        m_cubeVertexBuffer,
        m_cubeIndexBuffer,
        cubeIndexCount,
        glm::vec3{2.5f, 0.5f, 0.5f},
        glm::vec4{0.3f, 0.45f, 0.9f, 1.0f},
        0.25f,
        1.0f));
    m_drawItems.push_back(makeItem(
        m_cubeVertexBuffer,
        m_cubeIndexBuffer,
        cubeIndexCount,
        glm::vec3{0.0f, 0.5f, 2.5f},
        glm::vec4{0.9f, 0.8f, 0.3f, 1.0f},
        0.5f,
        0.5f));

    MRE_ASSERT(m_cubeVertexBuffer != nullptr, "Failed to create the cube vertex buffer");
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
        m_lightingSets.push_back(std::move(lightingSet));
    }
}

void DeferredRenderer::UpdateUniforms(
    std::uint32_t frameIndex,
    const glm::mat4& viewProjection,
    const glm::vec3& cameraPosition)
{
    m_sceneUniform.viewProjection = viewProjection;
    m_sceneUniform.cameraPosition = glm::vec4{cameraPosition, 1.0f};
    m_uniformBuffers[frameIndex]->Write(&m_sceneUniform, sizeof(SceneUniform));
}

} // namespace mukki
