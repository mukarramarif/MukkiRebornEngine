#include "Renderer/Vulkan/Resources/AccelerationStructure.h"

#include <array>
#include <cstring>

#include "Utilities/Assert.h"

namespace mukki::vulkan
{

std::vector<vk::AccelerationStructureInstanceKHR> ToVkInstances(
    std::span<const rhi::AccelerationStructureInstance> instances)
{
    std::vector<vk::AccelerationStructureInstanceKHR> nativeInstances(instances.size());
    for (std::size_t index = 0; index < instances.size(); ++index)
    {
        const rhi::AccelerationStructureInstance& source = instances[index];
        MRE_ASSERT(source.bottomLevel != nullptr, "Acceleration structure instances require a bottom level structure");

        vk::AccelerationStructureInstanceKHR& target = nativeInstances[index];
        std::memcpy(&target.transform, source.transform, sizeof(source.transform));
        target.instanceCustomIndex = source.instanceCustomIndex;
        target.instanceShaderBindingTableRecordOffset = 0;
        target.flags = static_cast<VkGeometryInstanceFlagsKHR>(
            vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
        target.accelerationStructureReference = source.bottomLevel->GetDeviceAddress();
        target.mask = source.instanceMask;
    }
    return nativeInstances;
}

VkAccelerationStructure::VkAccelerationStructure(
    vk::Device device,
    const RayTracingDispatch& dispatch,
    vk::AccelerationStructureKHR structure,
    std::unique_ptr<rhi::Buffer> backingBuffer,
    std::unique_ptr<rhi::Buffer> scratchBuffer,
    std::unique_ptr<rhi::Buffer> instanceBuffer,
    std::uint64_t deviceAddress,
    std::uint64_t size,
    vk::AccelerationStructureTypeKHR type,
    vk::BuildAccelerationStructureFlagBitsKHR flags,
    vk::AccelerationStructureGeometryKHR geometry,
    vk::AccelerationStructureBuildRangeInfoKHR buildRange)
    : m_device(device)
    , m_dispatch(&dispatch)
    , m_backingBuffer(std::move(backingBuffer))
    , m_scratchBuffer(std::move(scratchBuffer))
    , m_instanceBuffer(std::move(instanceBuffer))
    , m_structure(structure)
    , m_buildRange(buildRange)
    , m_deviceAddress(deviceAddress)
    , m_size(size)
{
    MRE_ASSERT(m_device != nullptr, "VkAccelerationStructure requires a valid device");
    MRE_ASSERT(m_dispatch != nullptr, "VkAccelerationStructure requires a valid dispatcher");
    MRE_ASSERT(m_backingBuffer != nullptr, "VkAccelerationStructure requires a backing buffer");
    MRE_ASSERT(m_scratchBuffer != nullptr, "VkAccelerationStructure requires a scratch buffer");
    MRE_ASSERT(m_structure != nullptr, "VkAccelerationStructure requires a valid handle");

    m_geometries.reserve(1);
    m_geometries.push_back(geometry);
    m_buildInfo = vk::AccelerationStructureBuildGeometryInfoKHR{}
                      .setType(type)
                      .setFlags(flags)
                      .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
                      .setGeometries(m_geometries);
}

VkAccelerationStructure::~VkAccelerationStructure()
{
    // The acceleration structure handle must be destroyed before its backing buffer.
    if (m_structure != nullptr && m_device != nullptr && m_dispatch != nullptr)
    {
        m_device.destroyAccelerationStructureKHR(m_structure, nullptr, *m_dispatch);
        m_structure = nullptr;
    }

    m_instanceBuffer.reset();
    m_scratchBuffer.reset();
    m_backingBuffer.reset();
}

void VkAccelerationStructure::UpdateInstances(std::span<const rhi::AccelerationStructureInstance> instances)
{
    MRE_ASSERT(m_instanceBuffer != nullptr, "Only top level structures have an instance buffer");
    MRE_ASSERT(!instances.empty(), "Acceleration structure updates require instances");

    const std::vector<vk::AccelerationStructureInstanceKHR> nativeInstances = ToVkInstances(instances);
    m_instanceBuffer->Write(
        nativeInstances.data(),
        nativeInstances.size() * sizeof(vk::AccelerationStructureInstanceKHR));

    m_buildRange.primitiveCount = static_cast<std::uint32_t>(nativeInstances.size());
}

void VkAccelerationStructure::RecordBuild(vk::CommandBuffer commandBuffer) const
{
    MRE_ASSERT(m_structure != nullptr, "RecordBuild requires a valid acceleration structure");

    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo = m_buildInfo;
    buildInfo.setMode(vk::BuildAccelerationStructureModeKHR::eBuild);
    buildInfo.setDstAccelerationStructure(m_structure);
    buildInfo.setScratchData(m_scratchBuffer->GetDeviceAddress());

    const std::array<vk::AccelerationStructureBuildGeometryInfoKHR, 1> buildInfos{buildInfo};
    const std::array<const vk::AccelerationStructureBuildRangeInfoKHR*, 1> buildRanges{&m_buildRange};
    commandBuffer.buildAccelerationStructuresKHR(buildInfos, buildRanges, *m_dispatch);
}

} // namespace mukki::vulkan
