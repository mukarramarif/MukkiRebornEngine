#include "Renderer/Vulkan/Resources/ResourceAllocator.h"

#include "Renderer/Vulkan/Core/VkConversions.h"
#include "Renderer/Vulkan/Resources/AccelerationStructure.h"
#include "Renderer/Vulkan/Resources/Buffer.h"
#include "Renderer/Vulkan/Resources/Texture.h"
#include "Utilities/Assert.h"
#include "Utilities/Log.h"

#include <array>
#include <cstring>
#include <vector>

namespace mukki::vulkan
{
namespace
{

vk::ImageMemoryBarrier2 MakeImageBarrier(
    const vk::Image& image,
    vk::ImageLayout oldLayout,
    vk::ImageLayout newLayout,
    vk::ImageAspectFlags aspect,
    vk::PipelineStageFlags2 srcStage,
    vk::AccessFlags2 srcAccess,
    vk::PipelineStageFlags2 dstStage,
    vk::AccessFlags2 dstAccess,
    std::uint32_t layerCount = 1)
{
    vk::ImageMemoryBarrier2 barrier{};
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.image = image;
    barrier.subresourceRange = vk::ImageSubresourceRange{}
                                   .setAspectMask(aspect)
                                   .setBaseMipLevel(0)
                                   .setLevelCount(1)
                                   .setBaseArrayLayer(0)
                                   .setLayerCount(layerCount);
    return barrier;
}

vk::ImageUsageFlags ToVkTextureUsage(rhi::TextureUsage usage)
{
    switch (usage)
    {
    case rhi::TextureUsage::Sampled:
        return vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
    case rhi::TextureUsage::ColorAttachment:
        return vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
    case rhi::TextureUsage::DepthAttachment:
        return vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled;
    case rhi::TextureUsage::Storage:
        return vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
               vk::ImageUsageFlagBits::eTransferDst;
    }

    MRE_UNREACHABLE();
}

} // namespace

void VkResourceAllocator::Init(
    const vk::raii::Device& device,
    VmaAllocator allocator,
    vk::Queue queue,
    std::uint32_t queueFamily,
    const RayTracingDispatch* rayTracingDispatch)
{
    MRE_ASSERT(allocator != VK_NULL_HANDLE, "VkResourceAllocator::Init requires a valid VMA allocator");
    MRE_ASSERT(queue != nullptr, "VkResourceAllocator::Init requires a valid queue");

    m_device = &device;
    m_allocator = allocator;
    m_queue = queue;
    m_rayTracingDispatch = rayTracingDispatch;

    const vk::CommandPoolCreateInfo poolInfo = vk::CommandPoolCreateInfo{}
                                                   .setFlags(vk::CommandPoolCreateFlagBits::eTransient)
                                                   .setQueueFamilyIndex(queueFamily);
    m_commandPool = vk::raii::CommandPool{device, poolInfo};
}

void VkResourceAllocator::Shutdown()
{
    m_commandPool = vk::raii::CommandPool{nullptr};
    m_device = nullptr;
    m_allocator = VK_NULL_HANDLE;
    m_queue = nullptr;
}

std::unique_ptr<rhi::Buffer> VkResourceAllocator::CreateBuffer(
    const rhi::BufferDesc& desc,
    const void* initialData,
    std::uint64_t initialDataSize)
{
    MRE_ASSERT(m_allocator != VK_NULL_HANDLE, "VkResourceAllocator::CreateBuffer called before Init");
    MRE_ASSERT(desc.size > 0, "Buffer size must be greater than zero");
    MRE_ASSERT(initialDataSize <= desc.size, "Initial data exceeds the buffer size");
    MRE_ASSERT(
        initialData != nullptr || initialDataSize == 0,
        "Initial data size requires a valid initial data pointer");

    const bool useStaging = initialData != nullptr && initialDataSize > 0 && !desc.hostVisible;

    vk::BufferUsageFlags usage = ToVkBufferUsage(desc.usage);
    if (desc.deviceAddress)
    {
        // Geometry buffers registered in the bindless array are read as storage buffers by ray
        // traced shaders.
        usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress |
                 vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                 vk::BufferUsageFlagBits::eStorageBuffer;
    }
    if (useStaging)
    {
        usage |= vk::BufferUsageFlagBits::eTransferDst;
    }

    const vk::BufferCreateInfo bufferInfo = vk::BufferCreateInfo{}
                                                .setSize(desc.size)
                                                .setUsage(usage)
                                                .setSharingMode(vk::SharingMode::eExclusive);

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = desc.hostVisible ? VMA_MEMORY_USAGE_AUTO : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationInfo.minAlignment = desc.alignment;
    if (desc.hostVisible)
    {
        allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                               VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    ::VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    const VkBufferCreateInfo nativeInfo = static_cast<VkBufferCreateInfo>(bufferInfo);
    const VkResult result = vmaCreateBuffer(m_allocator, &nativeInfo, &allocationInfo, &buffer, &allocation, nullptr);
    if (result != VK_SUCCESS)
    {
        MRE_ERROR("Failed to create a {} byte buffer (VkResult {})", desc.size, static_cast<int>(result));
        return nullptr;
    }

    std::uint64_t deviceAddress = 0;
    if (desc.deviceAddress || desc.usage == rhi::BufferUsage::AccelerationStructure ||
        desc.usage == rhi::BufferUsage::AccelerationStructureInput ||
        desc.usage == rhi::BufferUsage::AccelerationStructureScratch)
    {
        const vk::BufferDeviceAddressInfo addressInfo = vk::BufferDeviceAddressInfo{}.setBuffer(buffer);
        deviceAddress = m_device->getBufferAddress(addressInfo);
    }

    auto vkBuffer = std::make_unique<VkBuffer>(
        m_allocator,
        buffer,
        allocation,
        desc.size,
        desc.hostVisible,
        deviceAddress);

    if (initialData != nullptr && initialDataSize > 0)
    {
        if (desc.hostVisible)
        {
            vkBuffer->Write(initialData, initialDataSize, 0);
        }
        else
        {
            const vk::BufferCreateInfo stagingInfo = vk::BufferCreateInfo{}
                                                         .setSize(initialDataSize)
                                                         .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
                                                         .setSharingMode(vk::SharingMode::eExclusive);

            VmaAllocationCreateInfo stagingAllocationInfo{};
            stagingAllocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
            stagingAllocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                          VMA_ALLOCATION_CREATE_MAPPED_BIT;

            ::VkBuffer stagingBuffer = VK_NULL_HANDLE;
            VmaAllocation stagingAllocation = VK_NULL_HANDLE;
            VmaAllocationInfo stagingAllocationResult{};
            const VkBufferCreateInfo nativeStagingInfo = static_cast<VkBufferCreateInfo>(stagingInfo);
            const VkResult stagingResult = vmaCreateBuffer(
                m_allocator,
                &nativeStagingInfo,
                &stagingAllocationInfo,
                &stagingBuffer,
                &stagingAllocation,
                &stagingAllocationResult);
            if (stagingResult != VK_SUCCESS)
            {
                MRE_ERROR("Failed to create a staging buffer (VkResult {})", static_cast<int>(stagingResult));
                return vkBuffer;
            }

            std::memcpy(stagingAllocationResult.pMappedData, initialData, initialDataSize);

            ImmediateSubmit([&](vk::CommandBuffer commandBuffer) {
                const vk::BufferCopy region{0, 0, initialDataSize};
                commandBuffer.copyBuffer(stagingBuffer, buffer, region);
            });

            vmaDestroyBuffer(m_allocator, stagingBuffer, stagingAllocation);
        }
    }

    return vkBuffer;
}

std::unique_ptr<rhi::Texture> VkResourceAllocator::CreateTexture(const rhi::TextureDesc& desc)
{
    MRE_ASSERT(m_allocator != VK_NULL_HANDLE, "VkResourceAllocator::CreateTexture called before Init");
    MRE_ASSERT(desc.width > 0 && desc.height > 0, "Texture dimensions must be greater than zero");
    MRE_ASSERT(desc.format != rhi::Format::Undefined, "Texture format must be defined");
    MRE_ASSERT(desc.arrayLayers > 0, "Texture layer count must be greater than zero");

    const bool hasData = desc.initialData != nullptr && desc.initialDataSize > 0;
    MRE_ASSERT(!hasData || desc.arrayLayers == 1, "Initial data is only supported for single-layer textures");
    const bool isDepth = desc.usage == rhi::TextureUsage::DepthAttachment;
    const vk::ImageAspectFlags aspect = isDepth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;

    if (hasData)
    {
        MRE_ASSERT(desc.usage == rhi::TextureUsage::Sampled, "Initial data is only supported for sampled textures");
        MRE_ASSERT(
            desc.format == rhi::Format::R8G8B8A8Unorm || desc.format == rhi::Format::R8G8B8A8Srgb,
            "Initial data requires an R8G8B8A8 format");
        MRE_ASSERT(
            desc.initialDataSize == static_cast<std::uint64_t>(desc.width) * desc.height * 4,
            "Initial data size must match width * height * 4");
    }

    vk::ImageUsageFlags usage = ToVkTextureUsage(desc.usage);
    if (hasData)
    {
        usage |= vk::ImageUsageFlagBits::eTransferDst;
    }

    vk::ImageCreateInfo imageInfo{};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = ToVkFormat(desc.format);
    imageInfo.extent = vk::Extent3D{desc.width, desc.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = desc.arrayLayers;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.usage = usage;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    ::VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    const VkImageCreateInfo nativeInfo = static_cast<VkImageCreateInfo>(imageInfo);
    const VkResult result = vmaCreateImage(m_allocator, &nativeInfo, &allocationInfo, &image, &allocation, nullptr);
    if (result != VK_SUCCESS)
    {
        MRE_ERROR(
            "Failed to create a {}x{} texture (VkResult {})",
            desc.width,
            desc.height,
            static_cast<int>(result));
        return nullptr;
    }

    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    if (desc.usage == rhi::TextureUsage::Sampled)
    {
        layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    else if (desc.usage == rhi::TextureUsage::Storage)
    {
        // Storage textures are read and written by compute passes, so they stay in the general
        // layout for their whole lifetime.
        layout = vk::ImageLayout::eGeneral;
    }

    if (hasData)
    {
        const vk::BufferCreateInfo stagingInfo = vk::BufferCreateInfo{}
                                                     .setSize(desc.initialDataSize)
                                                     .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
                                                     .setSharingMode(vk::SharingMode::eExclusive);

        VmaAllocationCreateInfo stagingAllocationInfo{};
        stagingAllocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                      VMA_ALLOCATION_CREATE_MAPPED_BIT;

        ::VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VmaAllocation stagingAllocation = VK_NULL_HANDLE;
        VmaAllocationInfo stagingAllocationResult{};
        const VkBufferCreateInfo nativeStagingInfo = static_cast<VkBufferCreateInfo>(stagingInfo);
        const VkResult stagingResult = vmaCreateBuffer(
            m_allocator,
            &nativeStagingInfo,
            &stagingAllocationInfo,
            &stagingBuffer,
            &stagingAllocation,
            &stagingAllocationResult);
        if (stagingResult != VK_SUCCESS)
        {
            MRE_ERROR("Failed to create a texture staging buffer (VkResult {})", static_cast<int>(stagingResult));
            vmaDestroyImage(m_allocator, image, allocation);
            return nullptr;
        }

        std::memcpy(stagingAllocationResult.pMappedData, desc.initialData, desc.initialDataSize);

        ImmediateSubmit([&](vk::CommandBuffer commandBuffer) {
            const vk::ImageMemoryBarrier2 toTransfer = MakeImageBarrier(
                image,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal,
                aspect,
                vk::PipelineStageFlagBits2::eNone,
                vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eTransfer,
                vk::AccessFlagBits2::eTransferWrite);
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(toTransfer));

            const vk::BufferImageCopy copyRegion = vk::BufferImageCopy{}
                                                       .setBufferOffset(0)
                                                       .setBufferRowLength(0)
                                                       .setBufferImageHeight(0)
                                                       .setImageSubresource(vk::ImageSubresourceLayers{}
                                                                                .setAspectMask(aspect)
                                                                                .setMipLevel(0)
                                                                                .setBaseArrayLayer(0)
                                                                                .setLayerCount(1))
                                                       .setImageOffset(vk::Offset3D{0, 0, 0})
                                                       .setImageExtent(vk::Extent3D{desc.width, desc.height, 1});
            commandBuffer.copyBufferToImage(
                stagingBuffer,
                image,
                vk::ImageLayout::eTransferDstOptimal,
                copyRegion);

            const vk::ImageMemoryBarrier2 toShaderRead = MakeImageBarrier(
                image,
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal,
                aspect,
                vk::PipelineStageFlagBits2::eTransfer,
                vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eFragmentShader,
                vk::AccessFlagBits2::eShaderSampledRead);
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(toShaderRead));
        });

        vmaDestroyBuffer(m_allocator, stagingBuffer, stagingAllocation);
        layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    else if (isDepth)
    {
        // Depth targets are sampled by later passes (shadow maps, depth debug views); start them in
        // a shader readable layout so they can be bound before their first render pass.
        ImmediateSubmit([&](vk::CommandBuffer commandBuffer) {
            const vk::ImageMemoryBarrier2 toShaderRead = MakeImageBarrier(
                image,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eShaderReadOnlyOptimal,
                aspect,
                vk::PipelineStageFlagBits2::eNone,
                vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eFragmentShader,
                vk::AccessFlagBits2::eShaderSampledRead,
                desc.arrayLayers);
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(toShaderRead));
        });
        layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    else if (desc.usage == rhi::TextureUsage::Storage)
    {
        // Storage textures stay in the general layout for their whole lifetime; transition them up
        // front so descriptors written before the first compute pass reference a valid layout.
        ImmediateSubmit([&](vk::CommandBuffer commandBuffer) {
            const vk::ImageMemoryBarrier2 toGeneral = MakeImageBarrier(
                image,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eGeneral,
                aspect,
                vk::PipelineStageFlagBits2::eNone,
                vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader,
                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                desc.arrayLayers);
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{}.setImageMemoryBarriers(toGeneral));
        });
    }

    const vk::ImageViewCreateInfo viewInfo = vk::ImageViewCreateInfo{}
                                                 .setImage(image)
                                                 .setViewType(
                                                     desc.arrayLayers > 1 ? vk::ImageViewType::e2DArray
                                                                          : vk::ImageViewType::e2D)
                                                 .setFormat(ToVkFormat(desc.format))
                                                 .setSubresourceRange(vk::ImageSubresourceRange{}
                                                                          .setAspectMask(aspect)
                                                                          .setBaseMipLevel(0)
                                                                          .setLevelCount(1)
                                                                          .setBaseArrayLayer(0)
                                                                          .setLayerCount(desc.arrayLayers));
    vk::raii::ImageView view{*m_device, viewInfo};

    // Depth arrays are rendered one layer at a time, which needs per-layer views.
    std::vector<vk::raii::ImageView> layerViews{};
    if (isDepth && desc.arrayLayers > 1)
    {
        layerViews.reserve(desc.arrayLayers);
        for (std::uint32_t layer = 0; layer < desc.arrayLayers; ++layer)
        {
            const vk::ImageViewCreateInfo layerViewInfo = vk::ImageViewCreateInfo{}
                                                              .setImage(image)
                                                              .setViewType(vk::ImageViewType::e2D)
                                                              .setFormat(ToVkFormat(desc.format))
                                                              .setSubresourceRange(vk::ImageSubresourceRange{}
                                                                                       .setAspectMask(aspect)
                                                                                       .setBaseMipLevel(0)
                                                                                       .setLevelCount(1)
                                                                                       .setBaseArrayLayer(layer)
                                                                                       .setLayerCount(1));
            layerViews.emplace_back(*m_device, layerViewInfo);
        }
    }

    return std::make_unique<VkTexture>(
        m_allocator,
        image,
        allocation,
        std::move(view),
        std::move(layerViews),
        desc,
        layout);
}

std::unique_ptr<rhi::AccelerationStructure> VkResourceAllocator::CreateBottomLevelAccelerationStructure(
    const rhi::BottomLevelAccelerationStructureDesc& desc)
{
    MRE_ASSERT(m_device != nullptr, "VkResourceAllocator requires an initialized device");
    MRE_ASSERT(m_rayTracingDispatch != nullptr, "Bottom level structures require ray tracing support");
    MRE_ASSERT(desc.vertexBuffer != nullptr, "Bottom level structures require a vertex buffer");
    MRE_ASSERT(desc.indexBuffer != nullptr, "Bottom level structures require an index buffer");
    MRE_ASSERT(desc.vertexStride > 0, "Bottom level structures require a vertex stride");
    MRE_ASSERT(desc.indexCount >= 3 && desc.indexCount % 3 == 0, "Invalid triangle index count");

    const vk::AccelerationStructureGeometryTrianglesDataKHR triangles =
        vk::AccelerationStructureGeometryTrianglesDataKHR{}
            .setVertexFormat(vk::Format::eR32G32B32Sfloat)
            .setVertexData(desc.vertexBuffer->GetDeviceAddress())
            .setVertexStride(desc.vertexStride)
            .setMaxVertex(desc.vertexCount > 0 ? desc.vertexCount - 1 : 0)
            .setIndexType(vk::IndexType::eUint32)
            .setIndexData(desc.indexBuffer->GetDeviceAddress());

    const vk::AccelerationStructureGeometryKHR geometry =
        vk::AccelerationStructureGeometryKHR{}
            .setGeometryType(vk::GeometryTypeKHR::eTriangles)
            .setGeometry(vk::AccelerationStructureGeometryDataKHR{triangles})
            .setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    const vk::AccelerationStructureBuildGeometryInfoKHR buildInfo =
        vk::AccelerationStructureBuildGeometryInfoKHR{}
            .setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
            .setFlags(desc.fastTrace ? vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
                                     : vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild)
            .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
            .setGeometries(geometry);

    const std::uint32_t primitiveCount = desc.indexCount / 3;
    const std::array<std::uint32_t, 1> maxPrimitiveCounts{primitiveCount};
    const vk::Device device = *m_device;
    const vk::AccelerationStructureBuildSizesInfoKHR sizes =
        device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice,
            buildInfo,
            maxPrimitiveCounts,
            *m_rayTracingDispatch);

    std::unique_ptr<rhi::Buffer> backingBuffer = CreateBuffer(rhi::BufferDesc{sizes.accelerationStructureSize, rhi::BufferUsage::AccelerationStructure, false}, nullptr, 0);
    std::unique_ptr<rhi::Buffer> scratchBuffer = CreateBuffer(rhi::BufferDesc{sizes.buildScratchSize, rhi::BufferUsage::AccelerationStructureScratch, false, false, 256}, nullptr, 0);
    if (backingBuffer == nullptr || scratchBuffer == nullptr)
    {
        MRE_ERROR("Failed to allocate bottom level acceleration structure buffers");
        return nullptr;
    }

    const auto* backing = static_cast<const VkBuffer*>(backingBuffer.get());
    const vk::AccelerationStructureCreateInfoKHR createInfo =
        vk::AccelerationStructureCreateInfoKHR{}
            .setBuffer(backing->GetHandle())
            .setOffset(0)
            .setSize(sizes.accelerationStructureSize)
            .setType(vk::AccelerationStructureTypeKHR::eBottomLevel);

    const vk::AccelerationStructureKHR structure =
        device.createAccelerationStructureKHR(createInfo, nullptr, *m_rayTracingDispatch);
    const vk::AccelerationStructureDeviceAddressInfoKHR addressInfo =
        vk::AccelerationStructureDeviceAddressInfoKHR{}.setAccelerationStructure(structure);
    const std::uint64_t address =
        device.getAccelerationStructureAddressKHR(addressInfo, *m_rayTracingDispatch);

    vk::AccelerationStructureBuildRangeInfoKHR buildRange{};
    buildRange.primitiveCount = primitiveCount;

    return std::make_unique<VkAccelerationStructure>(
        device,
        *m_rayTracingDispatch,
        structure,
        std::move(backingBuffer),
        std::move(scratchBuffer),
        nullptr,
        address,
        sizes.accelerationStructureSize,
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        desc.fastTrace ? vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
                       : vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild,
        geometry,
        buildRange);
}

std::unique_ptr<rhi::AccelerationStructure> VkResourceAllocator::CreateTopLevelAccelerationStructure(
    const rhi::TopLevelAccelerationStructureDesc& desc)
{
    MRE_ASSERT(m_device != nullptr, "VkResourceAllocator requires an initialized device");
    MRE_ASSERT(m_rayTracingDispatch != nullptr, "Top level structures require ray tracing support");
    MRE_ASSERT(!desc.instances.empty(), "Top level structures require at least one instance");

    const std::vector<vk::AccelerationStructureInstanceKHR> nativeInstances = ToVkInstances(desc.instances);
    const std::uint64_t instanceBytes = nativeInstances.size() * sizeof(vk::AccelerationStructureInstanceKHR);

    std::unique_ptr<rhi::Buffer> instanceBuffer = CreateBuffer(
        rhi::BufferDesc{instanceBytes, rhi::BufferUsage::AccelerationStructureInput, true},
        nativeInstances.data(),
        instanceBytes);
    if (instanceBuffer == nullptr)
    {
        MRE_ERROR("Failed to allocate the acceleration structure instance buffer");
        return nullptr;
    }

    const vk::AccelerationStructureGeometryInstancesDataKHR instances =
        vk::AccelerationStructureGeometryInstancesDataKHR{}
            .setArrayOfPointers(false)
            .setData(instanceBuffer->GetDeviceAddress());

    const vk::AccelerationStructureGeometryKHR geometry =
        vk::AccelerationStructureGeometryKHR{}
            .setGeometryType(vk::GeometryTypeKHR::eInstances)
            .setGeometry(vk::AccelerationStructureGeometryDataKHR{instances})
            .setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    const vk::AccelerationStructureBuildGeometryInfoKHR buildInfo =
        vk::AccelerationStructureBuildGeometryInfoKHR{}
            .setType(vk::AccelerationStructureTypeKHR::eTopLevel)
            .setFlags(desc.fastTrace ? vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
                                     : vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild)
            .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
            .setGeometries(geometry);

    const std::array<std::uint32_t, 1> maxPrimitiveCounts{static_cast<std::uint32_t>(nativeInstances.size())};
    const vk::Device device = *m_device;
    const vk::AccelerationStructureBuildSizesInfoKHR sizes =
        device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice,
            buildInfo,
            maxPrimitiveCounts,
            *m_rayTracingDispatch);

    std::unique_ptr<rhi::Buffer> backingBuffer = CreateBuffer(rhi::BufferDesc{sizes.accelerationStructureSize, rhi::BufferUsage::AccelerationStructure, false}, nullptr, 0);
    std::unique_ptr<rhi::Buffer> scratchBuffer = CreateBuffer(rhi::BufferDesc{sizes.buildScratchSize, rhi::BufferUsage::AccelerationStructureScratch, false, false, 256}, nullptr, 0);
    if (backingBuffer == nullptr || scratchBuffer == nullptr)
    {
        MRE_ERROR("Failed to allocate top level acceleration structure buffers");
        return nullptr;
    }

    const auto* backing = static_cast<const VkBuffer*>(backingBuffer.get());
    const vk::AccelerationStructureCreateInfoKHR createInfo =
        vk::AccelerationStructureCreateInfoKHR{}
            .setBuffer(backing->GetHandle())
            .setOffset(0)
            .setSize(sizes.accelerationStructureSize)
            .setType(vk::AccelerationStructureTypeKHR::eTopLevel);

    const vk::AccelerationStructureKHR structure =
        device.createAccelerationStructureKHR(createInfo, nullptr, *m_rayTracingDispatch);
    const vk::AccelerationStructureDeviceAddressInfoKHR addressInfo =
        vk::AccelerationStructureDeviceAddressInfoKHR{}.setAccelerationStructure(structure);
    const std::uint64_t address =
        device.getAccelerationStructureAddressKHR(addressInfo, *m_rayTracingDispatch);

    vk::AccelerationStructureBuildRangeInfoKHR buildRange{};
    buildRange.primitiveCount = static_cast<std::uint32_t>(nativeInstances.size());

    return std::make_unique<VkAccelerationStructure>(
        device,
        *m_rayTracingDispatch,
        structure,
        std::move(backingBuffer),
        std::move(scratchBuffer),
        std::move(instanceBuffer),
        address,
        sizes.accelerationStructureSize,
        vk::AccelerationStructureTypeKHR::eTopLevel,
        desc.fastTrace ? vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
                       : vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild,
        geometry,
        buildRange);
}

void VkResourceAllocator::ImmediateSubmit(const std::function<void(vk::CommandBuffer)>& record)
{
    MRE_ASSERT(m_device != nullptr, "VkResourceAllocator::ImmediateSubmit requires an initialized allocator");

    std::vector<vk::raii::CommandBuffer> commandBuffers = m_device->allocateCommandBuffers(
        vk::CommandBufferAllocateInfo{}
            .setCommandPool(*m_commandPool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(1));
    MRE_ASSERT(commandBuffers.size() == 1, "Expected exactly one command buffer");

    vk::raii::CommandBuffer& commandBuffer = commandBuffers.front();
    commandBuffer.begin(vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    record(*commandBuffer);
    commandBuffer.end();

    const vk::CommandBufferSubmitInfo commandBufferInfo = vk::CommandBufferSubmitInfo{}.setCommandBuffer(*commandBuffer);
    const vk::SubmitInfo2 submitInfo = vk::SubmitInfo2{}.setCommandBufferInfos(commandBufferInfo);
    m_queue.submit2(submitInfo);
    m_queue.waitIdle();
}

} // namespace mukki::vulkan

