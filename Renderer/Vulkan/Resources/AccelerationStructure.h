#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "Renderer/RHI/Resources.h"
#include "Renderer/Vulkan/Core/VkRayTracingDispatch.h"

namespace mukki::vulkan
{

[[nodiscard]] std::vector<vk::AccelerationStructureInstanceKHR> ToVkInstances(
    std::span<const rhi::AccelerationStructureInstance> instances);

// Ray tracing entry points are extensions, so unlike core Vulkan they are not exported by the
// loader import library. Acceleration structures therefore use a dedicated dynamic dispatcher.
class VkAccelerationStructure final : public rhi::AccelerationStructure
{
public:
    VkAccelerationStructure(
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
        vk::AccelerationStructureBuildRangeInfoKHR buildRange);
    ~VkAccelerationStructure() override;

    [[nodiscard]] std::uint64_t GetDeviceAddress() const override { return m_deviceAddress; }
    [[nodiscard]] std::uint64_t GetSize() const override { return m_size; }
    [[nodiscard]] vk::AccelerationStructureKHR GetStructure() const { return m_structure; }

    // Rewrites the host-visible instance buffer of a top level structure.
    void UpdateInstances(std::span<const rhi::AccelerationStructureInstance> instances);

    // Records a full build into the command buffer.
    void RecordBuild(vk::CommandBuffer commandBuffer) const;

private:
    vk::Device m_device{nullptr};
    const RayTracingDispatch* m_dispatch = nullptr;
    std::unique_ptr<rhi::Buffer> m_backingBuffer{};
    std::unique_ptr<rhi::Buffer> m_scratchBuffer{};
    std::unique_ptr<rhi::Buffer> m_instanceBuffer{};
    vk::AccelerationStructureKHR m_structure{nullptr};
    // Owns the geometry so the build info pointers stay valid.
    std::vector<vk::AccelerationStructureGeometryKHR> m_geometries{};
    vk::AccelerationStructureBuildGeometryInfoKHR m_buildInfo{};
    vk::AccelerationStructureBuildRangeInfoKHR m_buildRange{};
    std::uint64_t m_deviceAddress = 0;
    std::uint64_t m_size = 0;
};

} // namespace mukki::vulkan
