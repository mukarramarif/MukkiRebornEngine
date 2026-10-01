#include "Renderer/Vulkan/Core/VkDevice.h"

#include "Renderer/Vulkan/Core/VkConversions.h"
#include "Utilities/Assert.h"
#include "Utilities/Log.h"
#include "Utilities/Profiling.h"

#include <algorithm>
#include <array>
#include <limits>

namespace mukki::vulkan
{

VkDevice::~VkDevice()
{
    Shutdown();
}

bool VkDevice::Init(const rhi::DeviceDesc& desc)
{
    MRE_ASSERT(!m_initialized, "VkDevice::Init called twice");
    MRE_ASSERT(desc.nativeWindowHandle != nullptr, "VkDevice::Init requires a native window handle");
    MRE_ASSERT(
        desc.framesInFlight >= 1 && desc.framesInFlight <= 3,
        "VkDevice supports between 1 and 3 frames in flight");

    m_framesInFlight = desc.framesInFlight;
    m_requestedExtent = desc.initialExtent;

    if (!m_instance.Init(desc.applicationName, desc.nativeWindowHandle, desc.enableValidation))
    {
        return false;
    }

    if (!PickPhysicalDevice() || !CreateLogicalDevice() || !CreateAllocator() || !CreateDescriptorPool() ||
        !CreateSwapchain() || !CreateFrameResources())
    {
        Shutdown();
        return false;
    }

    if (m_rayTracingSupported)
    {
        // Ray tracing entry points are extensions and are not exported by the loader import
        // library, so they are resolved through vkGetDeviceProcAddr.
        if (!m_rayTracingDispatch.Load(*m_instance.GetInstance(), *m_device))
        {
            MRE_WARN("Failed to resolve the ray tracing entry points, ray tracing is disabled");
            m_rayTracingSupported = false;
        }
    }

    m_resourceAllocator.Init(
        m_device,
        m_allocator,
        *m_graphicsQueue,
        m_graphicsQueueFamily,
        m_rayTracingSupported ? &m_rayTracingDispatch : nullptr);

#ifdef MRE_ENABLE_TRACY
    MRE_ASSERT(!m_commandLists.empty(), "Tracy calibration requires at least one command list");
    m_tracyContext = tracy::CreateVkContext(
        static_cast<::VkPhysicalDevice>(*m_physicalDevice),
        static_cast<::VkDevice>(*m_device),
        static_cast<::VkQueue>(*m_graphicsQueue),
        static_cast<::VkCommandBuffer>(*m_commandLists.front()->GetHandle()),
        nullptr,
        nullptr);
    if (m_tracyContext != nullptr)
    {
        m_tracyContext->Name("Mukki", 5);
        for (std::unique_ptr<VkCommandList>& commandList : m_commandLists)
        {
            commandList->SetTracyContext(m_tracyContext);
        }
        MRE_INFO("Tracy Vulkan context created");
    }
#endif

    m_initialized = true;
    MRE_INFO("Vulkan device initialized");
    return true;
}

void VkDevice::Shutdown()
{
    if (*m_device != nullptr)
    {
        m_device.waitIdle();
    }

    m_commandLists.clear();
#ifdef MRE_ENABLE_TRACY
    if (m_tracyContext != nullptr)
    {
        tracy::DestroyVkContext(m_tracyContext);
        m_tracyContext = nullptr;
    }
#endif
    m_frameSync.Shutdown();
    m_swapchain.reset();
    m_resourceAllocator.Shutdown();
    m_descriptorPool = vk::raii::DescriptorPool{nullptr};
    if (m_allocator != VK_NULL_HANDLE)
    {
        vmaDestroyAllocator(m_allocator);
        m_allocator = VK_NULL_HANDLE;
    }
    m_graphicsQueue = vk::raii::Queue{nullptr};
    m_device = vk::raii::Device{nullptr};
    m_instance.Shutdown();

    m_physicalDevice = vk::raii::PhysicalDevice{nullptr};
    m_swapchainInfo = rhi::SwapchainInfo{};
    m_requestedExtent = rhi::Extent2D{};
    m_descriptorIndexingFeatures = vk::PhysicalDeviceDescriptorIndexingFeatures{};
    m_descriptorIndexingProperties = vk::PhysicalDeviceDescriptorIndexingProperties{};
    m_accelerationStructureFeatures = vk::PhysicalDeviceAccelerationStructureFeaturesKHR{};
    m_rayQueryFeatures = vk::PhysicalDeviceRayQueryFeaturesKHR{};
    m_graphicsQueueFamily = 0;
    m_apiVersion = VK_API_VERSION_1_3;
    m_framesInFlight = 2;
    m_frameIndex = 0;
    m_currentImageIndex = 0;
    m_nextBindlessIndex = 0;
    m_descriptorIndexingSupported = false;
    m_rayTracingSupported = false;
    m_swapchainDirty = false;
    m_initialized = false;
}

bool VkDevice::BeginFrame()
{
    MRE_PROFILE_ZONE_N("VkDevice::BeginFrame");
    MRE_ASSERT(m_initialized, "VkDevice::BeginFrame called before VkDevice::Init");

    if (m_swapchainDirty)
    {
        m_swapchainDirty = false;
        RecreateSwapchain();
        if (m_swapchain == nullptr || !m_swapchain->IsInitialized())
        {
            return false;
        }
    }

    WaitForFrame(m_frameIndex);

    VkCommandList& commandList = *m_commandLists[m_frameIndex];

    try
    {
        vk::AcquireNextImageInfoKHR acquireInfo{};
        acquireInfo.swapchain = m_swapchain->GetHandle();
        acquireInfo.timeout = std::numeric_limits<std::uint64_t>::max();
        acquireInfo.semaphore = m_frameSync.GetImageAvailable(m_frameIndex);
        acquireInfo.deviceMask = 1;

        const auto [result, imageIndex] = m_device.acquireNextImage2KHR(acquireInfo);
        m_currentImageIndex = imageIndex;
        if (result == vk::Result::eSuboptimalKHR)
        {
            m_swapchainDirty = true;
        }
    }
    catch (const vk::OutOfDateKHRError&)
    {
        m_swapchainDirty = true;
        return false;
    }

    commandList.SetSwapchainTarget(
        m_swapchain->GetImage(m_currentImageIndex),
        m_swapchain->GetImageView(m_currentImageIndex),
        m_currentImageIndex,
        m_swapchain->IsImageInitialized(m_currentImageIndex),
        m_swapchain->GetExtent());

    return true;
}

void VkDevice::EndFrame()
{
    MRE_PROFILE_ZONE_N("VkDevice::EndFrame");
    MRE_ASSERT(m_initialized, "VkDevice::EndFrame called before VkDevice::Init");

    VkCommandList& commandList = *m_commandLists[m_frameIndex];

    const vk::Semaphore imageAvailable = m_frameSync.GetImageAvailable(m_frameIndex);
    const vk::Semaphore renderFinished = m_swapchain->GetRenderFinished(m_currentImageIndex);
    const vk::Semaphore timeline = m_frameSync.GetTimeline();
    const std::uint64_t signalValue = m_frameSync.AdvanceTimelineValue();

    vk::SemaphoreSubmitInfo waitInfo{};
    waitInfo.semaphore = imageAvailable;
    waitInfo.stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    waitInfo.value = 0;

    std::array<vk::SemaphoreSubmitInfo, 2> signalInfos{};
    signalInfos[0].semaphore = renderFinished;
    signalInfos[0].stageMask = vk::PipelineStageFlagBits2::eAllCommands;
    signalInfos[0].value = 0;
    signalInfos[1].semaphore = timeline;
    signalInfos[1].stageMask = vk::PipelineStageFlagBits2::eAllCommands;
    signalInfos[1].value = signalValue;

    vk::CommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.commandBuffer = *commandList.GetHandle();

    vk::SubmitInfo2 submitInfo{};
    submitInfo.waitSemaphoreInfoCount = 1;
    submitInfo.pWaitSemaphoreInfos = &waitInfo;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;
    submitInfo.signalSemaphoreInfoCount = static_cast<std::uint32_t>(signalInfos.size());
    submitInfo.pSignalSemaphoreInfos = signalInfos.data();

    m_graphicsQueue.submit2(submitInfo);
    m_frameSync.SetFrameValue(m_frameIndex, signalValue);

    if (commandList.WasSwapchainRendered())
    {
        m_swapchain->MarkImageInitialized(m_currentImageIndex);
    }

    const vk::SwapchainKHR swapchainHandle = m_swapchain->GetHandle();
    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinished;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchainHandle;
    presentInfo.pImageIndices = &m_currentImageIndex;

    const vk::Result presentResult = m_graphicsQueue.presentKHR(presentInfo);
    if (presentResult == vk::Result::eSuboptimalKHR || presentResult == vk::Result::eErrorOutOfDateKHR)
    {
        m_swapchainDirty = true;
    }

    m_frameIndex = (m_frameIndex + 1) % m_framesInFlight;
}

void VkDevice::WaitIdle()
{
    MRE_PROFILE_ZONE_N("VkDevice::WaitIdle");
    MRE_ASSERT(m_initialized, "VkDevice::WaitIdle called before VkDevice::Init");
    m_device.waitIdle();
}

void VkDevice::Resize(const rhi::Extent2D& extent)
{
    if (extent.width == 0 || extent.height == 0)
    {
        return;
    }

    if (extent.width == m_requestedExtent.width && extent.height == m_requestedExtent.height)
    {
        return;
    }

    m_requestedExtent = extent;
    m_swapchainDirty = true;
}

rhi::CommandList& VkDevice::GetCommandList()
{
    MRE_ASSERT(m_initialized, "VkDevice::GetCommandList called before VkDevice::Init");
    MRE_ASSERT(m_frameIndex < m_commandLists.size(), "Frame index out of range");
    return *m_commandLists[m_frameIndex];
}

std::unique_ptr<rhi::CommandList> VkDevice::CreateSecondaryCommandList()
{
    MRE_ASSERT(m_initialized, "VkDevice::CreateSecondaryCommandList called before VkDevice::Init");

    auto commandList = std::make_unique<VkCommandList>();
    commandList->Init(m_device, m_graphicsQueueFamily, true);
#ifdef MRE_ENABLE_TRACY
    commandList->SetTracyContext(m_tracyContext);
#endif
    return commandList;
}

const rhi::SwapchainInfo& VkDevice::GetSwapchainInfo() const
{
    return m_swapchainInfo;
}

std::unique_ptr<rhi::Buffer> VkDevice::CreateBuffer(
    const rhi::BufferDesc& desc,
    const void* initialData,
    std::uint64_t initialDataSize)
{
    MRE_ASSERT(m_initialized, "VkDevice::CreateBuffer called before VkDevice::Init");
    return m_resourceAllocator.CreateBuffer(desc, initialData, initialDataSize);
}

std::unique_ptr<rhi::Texture> VkDevice::CreateTexture(const rhi::TextureDesc& desc)
{
    MRE_ASSERT(m_initialized, "VkDevice::CreateTexture called before VkDevice::Init");

    std::unique_ptr<rhi::Texture> texture = m_resourceAllocator.CreateTexture(desc);
    if (texture != nullptr)
    {
        MRE_ASSERT(
            m_nextBindlessIndex < rhi::kMaxBindlessTextures,
            "Bindless texture limit of {} reached",
            rhi::kMaxBindlessTextures);
        static_cast<VkTexture*>(texture.get())->SetBindlessIndex(m_nextBindlessIndex++);
    }

    return texture;
}

std::unique_ptr<rhi::Sampler> VkDevice::CreateSampler(const rhi::SamplerDesc& desc)
{
    MRE_ASSERT(m_initialized, "VkDevice::CreateSampler called before VkDevice::Init");
    return std::make_unique<VkSampler>(m_device, desc);
}

std::unique_ptr<rhi::DescriptorSetLayout> VkDevice::CreateDescriptorSetLayout(
    const std::vector<rhi::DescriptorBinding>& bindings)
{
    MRE_ASSERT(m_initialized, "VkDevice::CreateDescriptorSetLayout called before VkDevice::Init");
    return std::make_unique<VkDescriptorSetLayout>(m_device, bindings);
}

std::unique_ptr<rhi::DescriptorSet> VkDevice::CreateDescriptorSet(const rhi::DescriptorSetLayout& layout)
{
    MRE_ASSERT(m_initialized, "VkDevice::CreateDescriptorSet called before VkDevice::Init");

    const auto& vkLayout = static_cast<const VkDescriptorSetLayout&>(layout);
    return std::make_unique<VkDescriptorSet>(m_device, m_descriptorPool, vkLayout);
}

std::unique_ptr<rhi::Shader> VkDevice::CreateShader(
    rhi::ShaderStage stage,
    const std::filesystem::path& spirvPath)
{
    MRE_ASSERT(m_initialized, "VkDevice::CreateShader called before VkDevice::Init");

    const std::vector<std::uint32_t> code = VkShader::LoadSpirv(spirvPath);
    if (code.empty())
    {
        return nullptr;
    }

    return std::make_unique<VkShader>(m_device, stage, code);
}

std::unique_ptr<rhi::Pipeline> VkDevice::CreatePipeline(const rhi::PipelineDesc& desc)
{
    MRE_ASSERT(m_initialized, "VkDevice::CreatePipeline called before VkDevice::Init");
    return std::make_unique<VkPipeline>(m_device, desc);
}

std::unique_ptr<rhi::AccelerationStructure> VkDevice::CreateBottomLevelAccelerationStructure(
    const rhi::BottomLevelAccelerationStructureDesc& desc)
{
    MRE_ASSERT(m_initialized, "VkDevice::CreateBottomLevelAccelerationStructure called before VkDevice::Init");
    MRE_ASSERT(m_rayTracingSupported, "Ray tracing is not supported by the selected device");
    return m_resourceAllocator.CreateBottomLevelAccelerationStructure(desc);
}

std::unique_ptr<rhi::AccelerationStructure> VkDevice::CreateTopLevelAccelerationStructure(
    const rhi::TopLevelAccelerationStructureDesc& desc)
{
    MRE_ASSERT(m_initialized, "VkDevice::CreateTopLevelAccelerationStructure called before VkDevice::Init");
    MRE_ASSERT(m_rayTracingSupported, "Ray tracing is not supported by the selected device");
    return m_resourceAllocator.CreateTopLevelAccelerationStructure(desc);
}

void VkDevice::UpdateTopLevelAccelerationStructure(
    rhi::AccelerationStructure& accelerationStructure,
    std::span<const rhi::AccelerationStructureInstance> instances)
{
    MRE_ASSERT(m_initialized, "VkDevice::UpdateTopLevelAccelerationStructure called before VkDevice::Init");
    static_cast<VkAccelerationStructure&>(accelerationStructure).UpdateInstances(instances);
}

bool VkDevice::PickPhysicalDevice()
{
    const std::vector<vk::raii::PhysicalDevice> devices = m_instance.GetInstance().enumeratePhysicalDevices();
    if (devices.empty())
    {
        MRE_CRITICAL("No Vulkan-capable physical device found");
        return false;
    }

    int bestScore = -1;
    vk::raii::PhysicalDevice best{nullptr};
    std::uint32_t bestFamily = 0;
    std::uint32_t bestApiVersion = 0;
    vk::PhysicalDeviceDescriptorIndexingFeatures bestIndexingFeatures{};
    vk::PhysicalDeviceDescriptorIndexingProperties bestIndexingProperties{};
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR bestAccelerationStructureFeatures{};
    vk::PhysicalDeviceRayQueryFeaturesKHR bestRayQueryFeatures{};
    bool bestRayTracingSupported = false;

    for (const vk::raii::PhysicalDevice& candidate : devices)
    {
        const vk::PhysicalDeviceProperties properties = candidate.getProperties();
        if (properties.apiVersion < VK_API_VERSION_1_3)
        {
            continue;
        }

        vk::PhysicalDeviceVulkan12Features features12{};
        vk::PhysicalDeviceFeatures2 features2{};
        features2.pNext = &features12;
        (*candidate).getFeatures2(&features2);

        if (!features12.descriptorIndexing)
        {
            MRE_WARN(
                "Physical device '{}' does not support descriptor indexing, skipping",
                properties.deviceName.data());
            continue;
        }

        vk::PhysicalDeviceDescriptorIndexingFeatures indexingFeatures{};
        vk::PhysicalDeviceFeatures2 indexingFeatures2{};
        indexingFeatures2.pNext = &indexingFeatures;
        (*candidate).getFeatures2(&indexingFeatures2);

        vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};
        vk::PhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
        vk::PhysicalDeviceFeatures2 rayTracingFeatures2{};
        rayTracingFeatures2.pNext = &rayQueryFeatures;
        rayQueryFeatures.pNext = &accelerationStructureFeatures;
        (*candidate).getFeatures2(&rayTracingFeatures2);

        const bool rayTracingSupported = accelerationStructureFeatures.accelerationStructure &&
                                         rayQueryFeatures.rayQuery &&
                                         features12.bufferDeviceAddress;

        const std::vector<vk::QueueFamilyProperties> families = candidate.getQueueFamilyProperties();
        std::uint32_t graphicsFamily = 0;
        bool graphicsFound = false;
        for (std::uint32_t index = 0; index < families.size(); ++index)
        {
            const bool supportsGraphics =
                static_cast<bool>(families[index].queueFlags & vk::QueueFlagBits::eGraphics);
            const bool supportsPresent = candidate.getSurfaceSupportKHR(index, *m_instance.GetSurface());
            if (supportsGraphics && supportsPresent)
            {
                graphicsFamily = index;
                graphicsFound = true;
                break;
            }
        }

        if (!graphicsFound)
        {
            continue;
        }

        const std::vector<vk::SurfaceFormatKHR> formats = candidate.getSurfaceFormatsKHR(*m_instance.GetSurface());
        const std::vector<vk::PresentModeKHR> presentModes =
            candidate.getSurfacePresentModesKHR(*m_instance.GetSurface());
        if (formats.empty() || presentModes.empty())
        {
            continue;
        }

        int score = 0;
        switch (properties.deviceType)
        {
        case vk::PhysicalDeviceType::eDiscreteGpu:
            score = 3000;
            break;
        case vk::PhysicalDeviceType::eIntegratedGpu:
            score = 2000;
            break;
        case vk::PhysicalDeviceType::eVirtualGpu:
            score = 1000;
            break;
        default:
            score = 500;
            break;
        }

        // Prefer ray tracing capable devices within the same device class.
        if (rayTracingSupported)
        {
            score += 500;
        }

        if (score > bestScore)
        {
            vk::PhysicalDeviceDescriptorIndexingProperties indexingProperties{};
            vk::PhysicalDeviceProperties2 properties2{};
            properties2.pNext = &indexingProperties;
            (*candidate).getProperties2(&properties2);

            bestScore = score;
            best = candidate;
            bestFamily = graphicsFamily;
            bestApiVersion = std::min(properties.apiVersion, VK_API_VERSION_1_4);
            bestIndexingFeatures = indexingFeatures;
            bestIndexingProperties = indexingProperties;
            bestAccelerationStructureFeatures = accelerationStructureFeatures;
            bestRayQueryFeatures = rayQueryFeatures;
            bestRayTracingSupported = rayTracingSupported;
        }
    }

    if (*best == nullptr)
    {
        MRE_CRITICAL(
            "No suitable Vulkan physical device found "
            "(Vulkan 1.3, descriptor indexing, graphics and present queue required)");
        return false;
    }

    m_physicalDevice = best;
    m_graphicsQueueFamily = bestFamily;
    m_apiVersion = bestApiVersion;
    m_descriptorIndexingFeatures = bestIndexingFeatures;
    m_descriptorIndexingProperties = bestIndexingProperties;
    m_descriptorIndexingSupported = true;
    m_accelerationStructureFeatures = bestAccelerationStructureFeatures;
    m_rayQueryFeatures = bestRayQueryFeatures;
    m_rayTracingSupported = bestRayTracingSupported;

    const vk::PhysicalDeviceProperties properties = best.getProperties();
    MRE_INFO(
        "Selected GPU '{}' (Vulkan {}.{}.{})",
        properties.deviceName.data(),
        VK_API_VERSION_MAJOR(bestApiVersion),
        VK_API_VERSION_MINOR(bestApiVersion),
        VK_API_VERSION_PATCH(bestApiVersion));
    MRE_INFO(
        "Descriptor indexing supported ({} update-after-bind sampled images per set, {} per stage)",
        m_descriptorIndexingProperties.maxDescriptorSetUpdateAfterBindSampledImages,
        m_descriptorIndexingProperties.maxPerStageDescriptorUpdateAfterBindSampledImages);
    MRE_INFO(
        "Ray tracing {} (acceleration structures, ray query{})",
        m_rayTracingSupported ? "supported" : "not supported",
        m_rayTracingSupported ? ", buffer device address" : "");

    return true;
}

bool VkDevice::CreateLogicalDevice()
{
    const float queuePriority = 1.0f;

    vk::DeviceQueueCreateInfo queueInfo{};
    queueInfo.queueFamilyIndex = m_graphicsQueueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    vk::PhysicalDeviceVulkan11Features features11{};
    features11.shaderDrawParameters = true;

    vk::PhysicalDeviceVulkan12Features features12{};
    features12.timelineSemaphore = true;
    features12.descriptorIndexing = true;
    features12.shaderSampledImageArrayNonUniformIndexing = true;
    features12.shaderUniformBufferArrayNonUniformIndexing = true;
    features12.shaderStorageBufferArrayNonUniformIndexing = true;
    features12.descriptorBindingUniformBufferUpdateAfterBind = true;
    features12.descriptorBindingSampledImageUpdateAfterBind = true;
    features12.descriptorBindingStorageBufferUpdateAfterBind = true;
    features12.descriptorBindingPartiallyBound = true;
    features12.runtimeDescriptorArray = true;

    vk::PhysicalDeviceVulkan13Features features13{};
    features13.dynamicRendering = true;
    features13.synchronization2 = true;

    MRE_ASSERT(m_descriptorIndexingSupported, "The selected physical device must support descriptor indexing");

    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};
    vk::PhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
    if (m_rayTracingSupported)
    {
        features12.bufferDeviceAddress = true;
        accelerationStructureFeatures.accelerationStructure = true;
        rayQueryFeatures.rayQuery = true;
        features13.pNext = &accelerationStructureFeatures;
        accelerationStructureFeatures.pNext = &rayQueryFeatures;
    }

    features11.pNext = &features12;
    features12.pNext = &features13;

    vk::PhysicalDeviceFeatures2 features2{};
    features2.pNext = &features11;

    std::vector<const char*> extensions{vk::KHRSwapchainExtensionName};
    if (m_rayTracingSupported)
    {
        extensions.push_back(vk::KHRDeferredHostOperationsExtensionName);
        extensions.push_back(vk::KHRAccelerationStructureExtensionName);
        extensions.push_back(vk::KHRRayQueryExtensionName);
    }

    vk::DeviceCreateInfo createInfo{};
    createInfo.pNext = &features2;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueInfo;
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    try
    {
        m_device = vk::raii::Device{m_physicalDevice, createInfo};
    }
    catch (const vk::SystemError& error)
    {
        MRE_CRITICAL("Failed to create the logical device: {}", error.what());
        return false;
    }

    m_graphicsQueue = m_device.getQueue(m_graphicsQueueFamily, 0);
    MRE_INFO("Logical device created (graphics queue family {})", m_graphicsQueueFamily);
    return true;
}

bool VkDevice::CreateAllocator()
{
    VmaAllocatorCreateInfo createInfo{};
    createInfo.physicalDevice = *m_physicalDevice;
    createInfo.device = *m_device;
    createInfo.instance = *m_instance.GetInstance();
    createInfo.vulkanApiVersion = m_apiVersion;
    if (m_rayTracingSupported)
    {
        createInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }

    const VkResult result = vmaCreateAllocator(&createInfo, &m_allocator);
    if (result != VK_SUCCESS)
    {
        MRE_CRITICAL("Failed to create the VMA allocator (VkResult {})", static_cast<int>(result));
        return false;
    }

    MRE_INFO("VMA allocator created");
    return true;
}

bool VkDevice::CreateDescriptorPool()
{
    const std::array<vk::DescriptorPoolSize, 7> poolSizes{
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 256},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 2048},
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 256},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, rhi::kMaxBindlessTextures},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 64},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 256},
        vk::DescriptorPoolSize{vk::DescriptorType::eAccelerationStructureKHR, 64},
    };

    const vk::DescriptorPoolCreateInfo createInfo =
        vk::DescriptorPoolCreateInfo{}
            .setFlags(
                vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
                vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind)
            .setMaxSets(512)
            .setPoolSizes(poolSizes);
    m_descriptorPool = vk::raii::DescriptorPool{m_device, createInfo};
    return true;
}

bool VkDevice::CreateSwapchain()
{
    auto swapchain = std::make_unique<VkSwapchain>();
    if (!swapchain->Init(
            m_physicalDevice,
            m_device,
            m_instance.GetSurface(),
            m_graphicsQueueFamily,
            m_requestedExtent,
            nullptr))
    {
        return false;
    }

    m_swapchain = std::move(swapchain);
    m_swapchainInfo.format = FromVkFormat(m_swapchain->GetFormat());
    m_swapchainInfo.extent = rhi::Extent2D{m_swapchain->GetExtent().width, m_swapchain->GetExtent().height};
    m_swapchainInfo.imageCount = m_swapchain->GetImageCount();
    return true;
}

bool VkDevice::CreateFrameResources()
{
    m_frameSync.Init(m_device, m_framesInFlight);

    m_commandLists.clear();
    m_commandLists.reserve(m_framesInFlight);
    for (std::uint32_t frame = 0; frame < m_framesInFlight; ++frame)
    {
        auto commandList = std::make_unique<VkCommandList>();
        commandList->Init(m_device, m_graphicsQueueFamily);
        m_commandLists.push_back(std::move(commandList));
    }

    m_frameIndex = 0;
    MRE_INFO("Frame resources created ({} frames in flight)", m_framesInFlight);
    return true;
}

void VkDevice::RecreateSwapchain()
{
    MRE_ASSERT(m_swapchain != nullptr, "RecreateSwapchain requires an existing swapchain");

    m_device.waitIdle();

    auto swapchain = std::make_unique<VkSwapchain>();
    if (!swapchain->Init(
            m_physicalDevice,
            m_device,
            m_instance.GetSurface(),
            m_graphicsQueueFamily,
            m_requestedExtent,
            m_swapchain->GetHandle()))
    {
        MRE_CRITICAL("Failed to recreate the swapchain");
        return;
    }

    m_swapchain = std::move(swapchain);
    m_swapchainInfo.format = FromVkFormat(m_swapchain->GetFormat());
    m_swapchainInfo.extent = rhi::Extent2D{m_swapchain->GetExtent().width, m_swapchain->GetExtent().height};
    m_swapchainInfo.imageCount = m_swapchain->GetImageCount();
}

void VkDevice::WaitForFrame(std::uint32_t frame)
{
    const std::uint64_t frameValue = m_frameSync.GetFrameValue(frame);
    if (frameValue == 0)
    {
        return;
    }

    const vk::Semaphore timeline = m_frameSync.GetTimeline();
    vk::SemaphoreWaitInfo waitInfo{};
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &timeline;
    waitInfo.pValues = &frameValue;

    const vk::Result result = m_device.waitSemaphores(waitInfo, std::numeric_limits<std::uint64_t>::max());
    if (result != vk::Result::eSuccess)
    {
        MRE_ERROR("Failed to wait for frame {} ({})", frame, vk::to_string(result));
    }
}

std::unique_ptr<rhi::Device> CreateVulkanDevice(const rhi::DeviceDesc& desc)
{
    auto device = std::make_unique<VkDevice>();
    if (!device->Init(desc))
    {
        return nullptr;
    }

    return device;
}

} // namespace mukki::vulkan
