#include "Renderer/Deferred/DeferredRenderer.h"

#include "Assets/GltfLoader.h"
#include "Core/JobSystem.h"
#include "Renderer/RHI/Device.h"
#include "Utilities/Assert.h"
#include "Utilities/Log.h"
#include "Utilities/PathUtils.h"

#include <algorithm>
#include <cstddef>
#include <future>
#include <iterator>
#include <string>

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
    glm::vec2 uv{0.0f};
};

constexpr Vertex kPlaneVertices[] = {
    {{-10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
    {{-10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
};

constexpr std::uint32_t kPlaneIndices[] = {0, 1, 2, 0, 2, 3};

constexpr std::uint32_t kWhitePixel = 0xFFFFFFFFu;

} // namespace

bool DeferredRenderer::Init(rhi::Device& device, const Desc& desc)
{
    MRE_ASSERT(!m_initialized, "DeferredRenderer::Init called twice");
    MRE_ASSERT(!desc.shaderDirectory.empty(), "DeferredRenderer requires a shader directory");

    m_device = &device;
    m_jobSystem = desc.jobSystem;
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

    const std::vector<rhi::DescriptorBinding> sceneBindings{
        rhi::DescriptorBinding{0, rhi::DescriptorType::UniformBuffer, rhi::ShaderStageMask::VertexFragment},
    };
    const std::vector<rhi::DescriptorBinding> materialBindings{
        rhi::DescriptorBinding{0, rhi::DescriptorType::CombinedImageSampler, rhi::ShaderStageMask::Fragment},
        rhi::DescriptorBinding{1, rhi::DescriptorType::CombinedImageSampler, rhi::ShaderStageMask::Fragment},
    };
    const std::vector<rhi::DescriptorBinding> lightingBindings{
        rhi::DescriptorBinding{0, rhi::DescriptorType::UniformBuffer, rhi::ShaderStageMask::VertexFragment},
        rhi::DescriptorBinding{1, rhi::DescriptorType::CombinedImageSampler, rhi::ShaderStageMask::Fragment},
        rhi::DescriptorBinding{2, rhi::DescriptorType::CombinedImageSampler, rhi::ShaderStageMask::Fragment},
        rhi::DescriptorBinding{3, rhi::DescriptorType::CombinedImageSampler, rhi::ShaderStageMask::Fragment},
    };
    m_sceneSetLayout = device.CreateDescriptorSetLayout(sceneBindings);
    m_materialSetLayout = device.CreateDescriptorSetLayout(materialBindings);
    m_lightingSetLayout = device.CreateDescriptorSetLayout(lightingBindings);

    m_defaultMaterialSet = device.CreateDescriptorSet(*m_materialSetLayout);
    m_defaultMaterialSet->BindTexture(0, *m_whiteTexture, *m_gbufferSampler);
    m_defaultMaterialSet->BindTexture(1, *m_whiteTexture, *m_gbufferSampler);

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

    m_gbufferSecondaries.clear();
    m_gbufferSecondaries.resize(framesInFlight);
    for (std::uint32_t frame = 0; frame < framesInFlight; ++frame)
    {
        m_gbufferSecondaries[frame].reserve(m_secondaryBatchCount);
        for (std::uint32_t batch = 0; batch < m_secondaryBatchCount; ++batch)
        {
            m_gbufferSecondaries[frame].push_back(device.CreateSecondaryCommandList());
        }
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
    UnloadModel();

    m_lightingSets.clear();
    m_sceneSets.clear();
    m_materials.clear();
    m_defaultMaterialSet.reset();
    m_lightingPipeline.reset();
    m_gbufferPipeline.reset();
    m_lightingFragmentShader.reset();
    m_lightingVertexShader.reset();
    m_gbufferFragmentShader.reset();
    m_gbufferVertexShader.reset();
    m_sceneDrawItems.clear();
    m_gbufferSecondaries.clear();
    m_lightingSetLayout.reset();
    m_materialSetLayout.reset();
    m_sceneSetLayout.reset();
    m_whiteTexture.reset();
    m_gbufferSampler.reset();
    m_planeIndexBuffer = nullptr;
    m_planeVertexBuffer = nullptr;
    m_uniformBuffers.clear();
    m_bufferManager.reset();
    m_gbuffer.Shutdown();
    m_device = nullptr;
    m_jobSystem = nullptr;
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

bool DeferredRenderer::LoadModel(const std::filesystem::path& path)
{
    MRE_ASSERT(m_initialized, "DeferredRenderer::LoadModel called before DeferredRenderer::Init");
    MRE_ASSERT(m_device != nullptr, "DeferredRenderer requires a device");

    GltfModel model{};
    if (!LoadGltfModel(path, model))
    {
        return false;
    }

    return UploadModel(path, model);
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

    return UploadModel(result.path, result.model);
}

bool DeferredRenderer::UploadModel(const std::filesystem::path& path, GltfModel& model)
{
    MRE_ASSERT(m_device != nullptr, "DeferredRenderer requires a device");

    m_device->WaitIdle();
    UnloadModel();

    const glm::vec3 center = (model.minBounds + model.maxBounds) * 0.5f;
    const glm::vec3 extent = model.maxBounds - model.minBounds;
    const float maxExtent = std::max({extent.x, extent.y, extent.z});
    const float scale = maxExtent > 0.0f ? 2.5f / maxExtent : 1.0f;
    const float lift = (center.y - model.minBounds.y) * scale;

    const glm::mat4 fit = glm::translate(glm::mat4{1.0f}, glm::vec3{0.0f, lift, 0.0f}) *
                          glm::scale(glm::mat4{1.0f}, glm::vec3{scale}) *
                          glm::translate(glm::mat4{1.0f}, -center);

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

    m_materials.reserve(model.materials.size());
    for (const GltfMaterial& source : model.materials)
    {
        Material material{};
        material.baseColorTexture = createTextureFromImage(source.baseColorImage, rhi::Format::R8G8B8A8Srgb);
        material.metallicRoughnessTexture =
            createTextureFromImage(source.metallicRoughnessImage, rhi::Format::R8G8B8A8Unorm);

        material.descriptorSet = m_device->CreateDescriptorSet(*m_materialSetLayout);
        material.descriptorSet->BindTexture(
            0,
            material.baseColorTexture != nullptr ? *material.baseColorTexture : *m_whiteTexture,
            *m_gbufferSampler);
        material.descriptorSet->BindTexture(
            1,
            material.metallicRoughnessTexture != nullptr ? *material.metallicRoughnessTexture : *m_whiteTexture,
            *m_gbufferSampler);

        m_materials.push_back(std::move(material));
    }

    m_modelVertexBuffers.reserve(model.primitives.size());
    m_modelIndexBuffers.reserve(model.primitives.size());
    for (const GltfPrimitive& primitive : model.primitives)
    {
        const std::uint64_t vertexBytes = sizeof(GltfVertex) * primitive.vertices.size();
        const std::uint64_t indexBytes = sizeof(std::uint32_t) * primitive.indices.size();

        m_modelVertexBuffers.push_back(m_device->CreateBuffer(
            rhi::BufferDesc{vertexBytes, rhi::BufferUsage::Vertex, false},
            primitive.vertices.data(),
            vertexBytes));
        m_modelIndexBuffers.push_back(m_device->CreateBuffer(
            rhi::BufferDesc{indexBytes, rhi::BufferUsage::Index, false},
            primitive.indices.data(),
            indexBytes));
    }

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
            item.vertexBuffer = m_modelVertexBuffers[primitiveIndex].get();
            item.indexBuffer = m_modelIndexBuffers[primitiveIndex].get();
            item.indexCount = static_cast<std::uint32_t>(primitive.indices.size());
            item.firstIndex = 0;
            item.pushConstants.model = fit * node.worldTransform;

            if (primitive.materialIndex >= 0 &&
                primitive.materialIndex < static_cast<int>(m_materials.size()))
            {
                const GltfMaterial& material = model.materials[static_cast<std::size_t>(primitive.materialIndex)];
                item.pushConstants.baseColorFactor = material.baseColorFactor;
                item.pushConstants.params = glm::vec4{
                    material.roughnessFactor,
                    material.metallicFactor,
                    0.0f,
                    0.0f};
                item.materialSet = m_materials[static_cast<std::size_t>(primitive.materialIndex)].descriptorSet.get();
            }
            else
            {
                item.pushConstants.baseColorFactor = glm::vec4{0.8f, 0.8f, 0.8f, 1.0f};
                item.pushConstants.params = glm::vec4{0.6f, 0.0f, 0.0f, 0.0f};
                item.materialSet = m_defaultMaterialSet.get();
            }

            m_modelDrawItems.push_back(item);
        }
    }

    m_modelName = PathToUtf8(path.filename());
    m_modelVertexCount = model.vertexCount;
    m_modelTriangleCount = model.triangleCount;

    std::size_t textureCount = 0;
    for (const Material& material : m_materials)
    {
        if (material.baseColorTexture != nullptr)
        {
            ++textureCount;
        }
        if (material.metallicRoughnessTexture != nullptr)
        {
            ++textureCount;
        }
    }

    MRE_INFO(
        "Model '{}' uploaded ({} draw items, {} textures)",
        m_modelName,
        m_modelDrawItems.size(),
        textureCount);
    return true;
}

void DeferredRenderer::UnloadModel()
{
    m_asyncLoader.Reset();
    m_modelDrawItems.clear();
    m_modelVertexBuffers.clear();
    m_modelIndexBuffers.clear();
    m_materials.clear();
    m_modelName.clear();
    m_modelVertexCount = 0;
    m_modelTriangleCount = 0;
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

    std::vector<const DrawItem*> gbufferItems{};
    gbufferItems.reserve(m_sceneDrawItems.size() + m_modelDrawItems.size());
    for (const DrawItem& item : m_sceneDrawItems)
    {
        gbufferItems.push_back(&item);
    }
    for (const DrawItem& item : m_modelDrawItems)
    {
        gbufferItems.push_back(&item);
    }

    const auto drawItems = [&commandList, frameIndex](const std::vector<const DrawItem*>& items) {
        for (const DrawItem* item : items)
        {
            commandList.BindVertexBuffer(*item->vertexBuffer);
            commandList.BindIndexBuffer(*item->indexBuffer);
            if (item->materialSet != nullptr)
            {
                commandList.BindDescriptorSet(1, *item->materialSet);
            }
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
        drawItems(gbufferItems);
        commandList.EndRendering();
    }

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

std::uint32_t DeferredRenderer::RecordGBufferSecondaries(
    std::uint32_t frameIndex,
    const rhi::RenderingDesc& desc,
    const std::vector<const DrawItem*>& items)
{
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

    const std::size_t itemsPerBatch = (items.size() + batchCount - 1) / batchCount;

    std::vector<std::future<void>> futures{};
    futures.reserve(batchCount);

    for (std::uint32_t batch = 0; batch < batchCount; ++batch)
    {
        const std::size_t begin = static_cast<std::size_t>(batch) * itemsPerBatch;
        const std::size_t end = std::min(begin + itemsPerBatch, items.size());
        if (begin >= end)
        {
            break;
        }

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

    for (std::size_t index = begin; index < end; ++index)
    {
        const DrawItem& item = *items[index];
        secondary.BindVertexBuffer(*item.vertexBuffer);
        secondary.BindIndexBuffer(*item.indexBuffer);
        if (item.materialSet != nullptr)
        {
            secondary.BindDescriptorSet(1, *item.materialSet);
        }
        secondary.SetPushConstants(
            &item.pushConstants,
            sizeof(ObjectPushConstants),
            rhi::ShaderStageMask::VertexFragment);
        secondary.DrawIndexed(item.indexCount, item.firstIndex);
    }

    secondary.End();
}

bool DeferredRenderer::CreatePipelines(rhi::Device& device, const std::filesystem::path& shaderDirectory)
{
    m_gbufferVertexShader = device.CreateShader(
        rhi::ShaderStage::Vertex,
        shaderDirectory / "gbuffer.vert.spv");
    m_gbufferFragmentShader = device.CreateShader(
        rhi::ShaderStage::Fragment,
        shaderDirectory / "gbuffer.frag.spv");
    m_lightingVertexShader = device.CreateShader(
        rhi::ShaderStage::Vertex,
        shaderDirectory / "lighting.vert.spv");
    m_lightingFragmentShader = device.CreateShader(
        rhi::ShaderStage::Fragment,
        shaderDirectory / "lighting.frag.spv");

    if (m_gbufferVertexShader == nullptr || m_gbufferFragmentShader == nullptr ||
        m_lightingVertexShader == nullptr || m_lightingFragmentShader == nullptr)
    {
        MRE_CRITICAL("Failed to load the deferred shaders from '{}'", PathToUtf8(shaderDirectory));
        return false;
    }

    rhi::VertexLayout vertexLayout{};
    vertexLayout.AddAttribute(0, rhi::Format::R32G32B32Sfloat);
    vertexLayout.AddAttribute(1, rhi::Format::R32G32B32Sfloat);
    vertexLayout.AddAttribute(2, rhi::Format::R32G32Sfloat);

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
    gbufferPipelineDesc.descriptorSetLayouts = {m_sceneSetLayout.get(), m_materialSetLayout.get()};
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

    m_planeVertexBuffer = &m_bufferManager->CreateVertexBuffer(kPlaneVertices, sizeof(kPlaneVertices));
    m_planeIndexBuffer = &m_bufferManager->CreateIndexBuffer(kPlaneIndices, sizeof(kPlaneIndices));

    DrawItem ground{};
    ground.vertexBuffer = m_planeVertexBuffer;
    ground.indexBuffer = m_planeIndexBuffer;
    ground.indexCount = static_cast<std::uint32_t>(std::size(kPlaneIndices));
    ground.firstIndex = 0;
    ground.materialSet = m_defaultMaterialSet.get();
    ground.pushConstants.model = glm::mat4{1.0f};
    ground.pushConstants.baseColorFactor = glm::vec4{0.35f, 0.35f, 0.38f, 1.0f};
    ground.pushConstants.params = glm::vec4{0.9f, 0.0f, 0.0f, 0.0f};
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
