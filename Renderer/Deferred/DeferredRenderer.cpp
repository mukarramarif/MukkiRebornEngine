#include "Renderer/Deferred/DeferredRenderer.h"

#include "Assets/GltfLoader.h"
#include "Core/JobSystem.h"
#include "Renderer/RHI/Device.h"
#include "Scene/SceneDescription.h"
#include "Utilities/Assert.h"
#include "Utilities/Log.h"
#include "Utilities/PathUtils.h"
#include "Utilities/Profiling.h"

#include <algorithm>
#include <cmath>
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

constexpr Vertex kPlaneVertices[] = {
    {{-10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{10.0f, 0.0f, -10.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
    {{-10.0f, 0.0f, 10.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
};

constexpr std::uint32_t kPlaneIndices[] = {0, 1, 2, 0, 2, 3};

constexpr std::uint32_t kWhitePixel = 0xFFFFFFFFu;

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

    const std::vector<rhi::DescriptorBinding> sceneBindings{
        rhi::DescriptorBinding{0, rhi::DescriptorType::UniformBuffer, rhi::ShaderStageMask::VertexFragment},
    };
    const std::vector<rhi::DescriptorBinding> bindlessBindings{
        rhi::DescriptorBinding{
            0,
            rhi::DescriptorType::SampledImage,
            rhi::ShaderStageMask::Fragment,
            rhi::kMaxBindlessTextures,
            true},
        rhi::DescriptorBinding{
            1,
            rhi::DescriptorType::Sampler,
            rhi::ShaderStageMask::Fragment,
            rhi::kMaxBindlessSamplers},
    };
    const std::vector<rhi::DescriptorBinding> lightingBindings{
        rhi::DescriptorBinding{0, rhi::DescriptorType::UniformBuffer, rhi::ShaderStageMask::VertexFragment},
        rhi::DescriptorBinding{1, rhi::DescriptorType::CombinedImageSampler, rhi::ShaderStageMask::Fragment},
        rhi::DescriptorBinding{2, rhi::DescriptorType::CombinedImageSampler, rhi::ShaderStageMask::Fragment},
        rhi::DescriptorBinding{3, rhi::DescriptorType::CombinedImageSampler, rhi::ShaderStageMask::Fragment},
        rhi::DescriptorBinding{4, rhi::DescriptorType::StorageBuffer, rhi::ShaderStageMask::Fragment},
        rhi::DescriptorBinding{5, rhi::DescriptorType::StorageBuffer, rhi::ShaderStageMask::Fragment},
        rhi::DescriptorBinding{6, rhi::DescriptorType::StorageBuffer, rhi::ShaderStageMask::Fragment},
        rhi::DescriptorBinding{7, rhi::DescriptorType::UniformBuffer, rhi::ShaderStageMask::Fragment},
        rhi::DescriptorBinding{8, rhi::DescriptorType::SampledImage, rhi::ShaderStageMask::Fragment},
        rhi::DescriptorBinding{9, rhi::DescriptorType::Sampler, rhi::ShaderStageMask::Fragment},
    };
    const std::vector<rhi::DescriptorBinding> clusterBindings{
        rhi::DescriptorBinding{0, rhi::DescriptorType::UniformBuffer, rhi::ShaderStageMask::Compute},
        rhi::DescriptorBinding{1, rhi::DescriptorType::StorageBuffer, rhi::ShaderStageMask::Compute},
        rhi::DescriptorBinding{2, rhi::DescriptorType::StorageBuffer, rhi::ShaderStageMask::Compute},
        rhi::DescriptorBinding{3, rhi::DescriptorType::StorageBuffer, rhi::ShaderStageMask::Compute},
    };
    const std::vector<rhi::DescriptorBinding> shadowBindings{
        rhi::DescriptorBinding{0, rhi::DescriptorType::UniformBuffer, rhi::ShaderStageMask::Vertex},
    };
    m_sceneSetLayout = device.CreateDescriptorSetLayout(sceneBindings);
    m_bindlessSetLayout = device.CreateDescriptorSetLayout(bindlessBindings);
    m_lightingSetLayout = device.CreateDescriptorSetLayout(lightingBindings);
    m_clusterSetLayout = device.CreateDescriptorSetLayout(clusterBindings);
    m_shadowSetLayout = device.CreateDescriptorSetLayout(shadowBindings);

    m_bindlessSet = device.CreateDescriptorSet(*m_bindlessSetLayout);
    m_bindlessSet->BindTexture(0, *m_whiteTexture, m_whiteTexture->GetBindlessIndex());
    static_cast<void>(GetSamplerIndex(rhi::AddressMode::ClampToEdge, rhi::AddressMode::ClampToEdge));

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

    if (!CreateClusterResources(device) || !CreateShadowResources(device))
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

        const std::uint32_t shadowBatchCount = m_secondaryBatchCount * kShadowCascadeCount;
        m_shadowSecondaries[frame].reserve(shadowBatchCount);
        for (std::uint32_t batch = 0; batch < shadowBatchCount; ++batch)
        {
            m_shadowSecondaries[frame].push_back(device.CreateSecondaryCommandList());
        }
    }

    if (!CreateScene() || !CreatePipelines(device, desc.shaderDirectory))
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
    m_shadowMap.reset();
    m_shadowSampler.reset();
    m_shadowUniformBuffers.clear();
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

        resources->materials.push_back(std::move(material));
    }

    resources->vertexBuffers.reserve(model.primitives.size());
    resources->indexBuffers.reserve(model.primitives.size());
    for (const GltfPrimitive& primitive : model.primitives)
    {
        const std::uint64_t vertexBytes = sizeof(GltfVertex) * primitive.vertices.size();
        const std::uint64_t indexBytes = sizeof(std::uint32_t) * primitive.indices.size();

        resources->vertexBuffers.push_back(m_device->CreateBuffer(
            rhi::BufferDesc{vertexBytes, rhi::BufferUsage::Vertex, false},
            primitive.vertices.data(),
            vertexBytes));
        resources->indexBuffers.push_back(m_device->CreateBuffer(
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
            item.vertexBuffer = resources->vertexBuffers[primitiveIndex].get();
            item.indexBuffer = resources->indexBuffers[primitiveIndex].get();
            item.indexCount = static_cast<std::uint32_t>(primitive.indices.size());
            item.firstIndex = 0;
            item.pushConstants.model = fit * node.worldTransform;

            const std::uint32_t whiteIndex = m_whiteTexture->GetBindlessIndex();
            item.pushConstants.textureIndices = glm::uvec2{whiteIndex, whiteIndex};

            if (primitive.materialIndex >= 0 &&
                primitive.materialIndex < static_cast<int>(resources->materials.size()))
            {
                const GltfMaterial& material = model.materials[static_cast<std::size_t>(primitive.materialIndex)];
                item.pushConstants.baseColorFactor = material.baseColorFactor;
                item.pushConstants.params = glm::vec4{
                    material.roughnessFactor,
                    material.metallicFactor,
                    0.0f,
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
}

void DeferredRenderer::UpdateSceneObjectTransform(std::size_t objectIndex, const glm::mat4& transform)
{
    MRE_ASSERT(objectIndex < m_objectInstances.size(), "Scene object index out of range");

    SceneObjectInstance& sceneObject = m_objectInstances[objectIndex];
    sceneObject.transform = transform;

    const ModelResources& model = *m_models[sceneObject.modelIndex];
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
}

void DeferredRenderer::ClearModelInstances()
{
    m_modelDrawItems.clear();
    m_modelBaseTransforms.clear();
    m_objectInstances.clear();
}

void DeferredRenderer::ClearModels()
{
    m_modelCache.clear();
    m_models.clear();
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

    const bool shadowsActive = m_shadowSettings.enabled && FindShadowLight() != nullptr &&
                               !gbufferItems.empty() && m_shadowMap != nullptr;
    if (shadowsActive)
    {
        commandList.BeginGpuZone("ShadowPass");

        rhi::RenderingDesc shadowDesc{};
        shadowDesc.depthAttachment.texture = m_shadowMap.get();
        shadowDesc.depthAttachment.clearDepth = 1.0f;
        shadowDesc.depthAttachment.clear = true;

        const bool useShadowSecondaries =
            m_jobSystem != nullptr && m_secondaryBatchCount > 1 && gbufferItems.size() > 1 &&
            frameIndex < m_shadowSecondaries.size();

        if (useShadowSecondaries)
        {
            const std::uint32_t batchCount = RecordShadowSecondaries(frameIndex, gbufferItems);
            shadowDesc.secondaryCommandBuffers = true;

            for (std::uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
            {
                std::vector<const rhi::CommandList*> secondaries{};
                secondaries.reserve(batchCount);
                for (std::uint32_t batch = 0; batch < batchCount; ++batch)
                {
                    secondaries.push_back(
                        m_shadowSecondaries[frameIndex][cascade * m_secondaryBatchCount + batch].get());
                }

                shadowDesc.depthAttachment.layer = cascade;
                commandList.BeginRendering(shadowDesc);
                commandList.ExecuteCommands(secondaries);
                commandList.EndRendering();
            }
        }
        else
        {
            for (std::uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
            {
                shadowDesc.depthAttachment.layer = cascade;
                commandList.BeginRendering(shadowDesc);
                commandList.BindPipeline(*m_shadowPipeline);
                commandList.BindDescriptorSet(0, *m_shadowSets[frameIndex]);
                commandList.SetDepthBias(m_shadowSettings.depthBias, m_shadowSettings.depthBias * 1.5f, 0.0f);

                ShadowPushConstants pushConstants{};
                pushConstants.cascadeIndex = cascade;
                for (const DrawItem* item : gbufferItems)
                {
                    pushConstants.object = item->pushConstants;
                    commandList.BindVertexBuffer(*item->vertexBuffer);
                    commandList.BindIndexBuffer(*item->indexBuffer);
                    commandList.SetPushConstants(
                        &pushConstants,
                        sizeof(ShadowPushConstants),
                        rhi::ShaderStageMask::Vertex);
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

    commandList.BeginGpuZone("Lighting");
    commandList.BeginSwapchainRendering(rhi::ClearColor{0.0f, 0.0f, 0.0f, 1.0f}, true);
    commandList.BindPipeline(*m_lightingPipeline);
    commandList.BindDescriptorSet(0, *m_lightingSets[frameIndex]);
    commandList.Draw(3);
    commandList.EndSwapchainRendering();
    commandList.EndGpuZone();
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

std::uint32_t DeferredRenderer::RecordShadowSecondaries(
    std::uint32_t frameIndex,
    const std::vector<const DrawItem*>& items)
{
    MRE_PROFILE_ZONE_N("RecordShadowSecondaries");
    MRE_ASSERT(m_jobSystem != nullptr, "RecordShadowSecondaries requires a job system");
    MRE_ASSERT(frameIndex < m_shadowSecondaries.size(), "Frame index out of range");
    MRE_ASSERT(!items.empty(), "RecordShadowSecondaries requires draw items");

    const std::uint32_t batchCount = std::min(
        m_secondaryBatchCount,
        static_cast<std::uint32_t>(items.size()));
    MRE_ASSERT(batchCount > 0, "RecordShadowSecondaries requires at least one batch");
    MRE_ASSERT(
        static_cast<std::size_t>(batchCount) * kShadowCascadeCount <= m_shadowSecondaries[frameIndex].size(),
        "Not enough shadow secondary command lists were allocated");

    rhi::RenderingDesc desc{};
    desc.depthAttachment.texture = m_shadowMap.get();
    desc.depthAttachment.clearDepth = 1.0f;
    desc.depthAttachment.clear = true;

    const std::size_t itemsPerBatch = (items.size() + batchCount - 1) / batchCount;

    std::vector<std::future<void>> futures{};
    futures.reserve(static_cast<std::size_t>(batchCount) * kShadowCascadeCount);

    for (std::uint32_t cascade = 0; cascade < kShadowCascadeCount; ++cascade)
    {
        for (std::uint32_t batch = 0; batch < batchCount; ++batch)
        {
            const std::size_t begin = static_cast<std::size_t>(batch) * itemsPerBatch;
            const std::size_t end = std::min(begin + itemsPerBatch, items.size());
            if (begin >= end)
            {
                break;
            }

            rhi::CommandList* secondary =
                m_shadowSecondaries[frameIndex][cascade * m_secondaryBatchCount + batch].get();
            futures.push_back(m_jobSystem->Submit(
                [this, frameIndex, secondary, &items, begin, end, cascade, &desc]() {
                    RecordShadowBatch(frameIndex, items, begin, end, cascade, *secondary, desc);
                }));
        }
    }

    for (std::future<void>& future : futures)
    {
        future.get();
    }

    return batchCount;
}

void DeferredRenderer::RecordShadowBatch(
    std::uint32_t frameIndex,
    const std::vector<const DrawItem*>& items,
    std::size_t begin,
    std::size_t end,
    std::uint32_t cascadeIndex,
    rhi::CommandList& secondary,
    const rhi::RenderingDesc& desc)
{
    MRE_ASSERT(m_shadowPipeline != nullptr, "RecordShadowBatch requires the shadow pipeline");
    MRE_ASSERT(frameIndex < m_shadowSets.size(), "Frame index out of range");

    secondary.BeginSecondary(desc);
    secondary.BindPipeline(*m_shadowPipeline);
    secondary.BindDescriptorSet(0, *m_shadowSets[frameIndex]);
    secondary.SetDepthBias(m_shadowSettings.depthBias, m_shadowSettings.depthBias * 1.5f, 0.0f);
    secondary.BeginGpuZone("ShadowBatch");

    ShadowPushConstants pushConstants{};
    pushConstants.cascadeIndex = cascadeIndex;

    for (std::size_t index = begin; index < end; ++index)
    {
        const DrawItem& item = *items[index];
        pushConstants.object = item.pushConstants;
        secondary.BindVertexBuffer(*item.vertexBuffer);
        secondary.BindIndexBuffer(*item.indexBuffer);
        secondary.SetPushConstants(
            &pushConstants,
            sizeof(ShadowPushConstants),
            rhi::ShaderStageMask::Vertex);
        secondary.DrawIndexed(item.indexCount, item.firstIndex);
    }

    secondary.EndGpuZone();
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
    m_clusterCullingShader = device.CreateShader(
        rhi::ShaderStage::Compute,
        shaderDirectory / "cluster_culling.comp.spv");
    m_shadowVertexShader = device.CreateShader(
        rhi::ShaderStage::Vertex,
        shaderDirectory / "shadow.vert.spv");
    m_shadowFragmentShader = device.CreateShader(
        rhi::ShaderStage::Fragment,
        shaderDirectory / "shadow.frag.spv");

    if (m_gbufferVertexShader == nullptr || m_gbufferFragmentShader == nullptr ||
        m_lightingVertexShader == nullptr || m_lightingFragmentShader == nullptr ||
        m_clusterCullingShader == nullptr || m_shadowVertexShader == nullptr ||
        m_shadowFragmentShader == nullptr)
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
    gbufferPipelineDesc.descriptorSetLayouts = {m_sceneSetLayout.get(), m_bindlessSetLayout.get()};
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

    rhi::PipelineDesc clusterPipelineDesc{};
    clusterPipelineDesc.computeShader = m_clusterCullingShader.get();
    clusterPipelineDesc.descriptorSetLayouts = {m_clusterSetLayout.get()};
    m_clusterCullingPipeline = device.CreatePipeline(clusterPipelineDesc);

    // The shadow pass only consumes positions but binds the same interleaved vertex buffers.
    rhi::VertexLayout shadowVertexLayout{};
    shadowVertexLayout.AddAttribute(0, rhi::Format::R32G32B32Sfloat);
    shadowVertexLayout.SetBindingStride(vertexLayout.GetStride());

    rhi::PipelineDesc shadowPipelineDesc{};
    shadowPipelineDesc.vertexShader = m_shadowVertexShader.get();
    shadowPipelineDesc.fragmentShader = m_shadowFragmentShader.get();
    shadowPipelineDesc.vertexLayout = shadowVertexLayout;
    shadowPipelineDesc.depthTest = true;
    shadowPipelineDesc.depthWrite = true;
    shadowPipelineDesc.depthBiasEnable = true;
    shadowPipelineDesc.cullMode = rhi::CullMode::None;
    shadowPipelineDesc.pushConstantSize = sizeof(ShadowPushConstants);
    shadowPipelineDesc.pushConstantStages = rhi::ShaderStageMask::Vertex;
    shadowPipelineDesc.descriptorSetLayouts = {m_shadowSetLayout.get()};
    shadowPipelineDesc.depthFormat = rhi::Format::D32Sfloat;
    m_shadowPipeline = device.CreatePipeline(shadowPipelineDesc);

    return m_gbufferPipeline != nullptr && m_lightingPipeline != nullptr &&
           m_clusterCullingPipeline != nullptr && m_shadowPipeline != nullptr;
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
    ground.pushConstants.textureIndices = glm::uvec2{whiteIndex, whiteIndex};
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
        "Shadow map array created ({}x{}, {} cascades)",
        kShadowResolution,
        kShadowResolution,
        kShadowCascadeCount);
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

void DeferredRenderer::UpdateUniforms(std::uint32_t frameIndex, const FrameView& view)
{
    MRE_ASSERT(frameIndex < m_uniformBuffers.size(), "Frame index out of range");
    MRE_ASSERT(frameIndex < m_lightBuffers.size(), "Frame index out of range");

    const float tanHalfFovY = std::abs(view.projection[1][1]) > 0.0f
        ? 1.0f / std::abs(view.projection[1][1])
        : std::tan(view.fovYRadians * 0.5f);

    m_gpuLights.clear();
    m_gpuLights.reserve(m_lights.size());
    for (const Light& light : m_lights)
    {
        if (m_gpuLights.size() >= kMaxLights)
        {
            break;
        }
        m_gpuLights.push_back(ToGpuLight(light));
    }

    m_sceneUniform.viewProjection = view.viewProjection;
    m_sceneUniform.view = view.view;
    m_sceneUniform.cameraPosition = glm::vec4{view.position, 1.0f};
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
        0.0f};
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

    const Light* shadowLight = FindShadowLight();
    if (m_shadowSettings.enabled && shadowLight != nullptr)
    {
        const std::vector<ShadowCascade> cascades =
            ComputeShadowCascades(view, shadowLight->direction, m_shadowSettings);

        for (std::size_t index = 0; index < cascades.size() && index < kShadowCascadeCount; ++index)
        {
            m_shadowUniform.lightViewProjection[index] = cascades[index].lightViewProjection;
            if (index + 1 < cascades.size())
            {
                m_shadowUniform.cascadeSplits[static_cast<glm::length_t>(index)] = cascades[index].splitFar;
            }
        }

        m_shadowUniform.params = glm::vec4{
            m_shadowSettings.depthBias,
            m_shadowSettings.normalBias,
            cascades.empty() ? m_shadowSettings.distance : cascades.back().splitFar,
            1.0f};
    }

    m_shadowUniformBuffers[frameIndex]->Write(&m_shadowUniform, sizeof(ShadowUniform));
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
