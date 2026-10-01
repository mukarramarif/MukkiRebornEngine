#pragma once

#include <vulkan/vulkan.hpp>

namespace mukki::vulkan
{

// Minimal dispatcher for the ray tracing extension entry points. Device extensions are not
// exported by the Vulkan loader import library, so they are resolved through
// vkGetDeviceProcAddr and passed explicitly to Vulkan-Hpp calls.
struct RayTracingDispatch
{
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR = nullptr;
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR = nullptr;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR = nullptr;

    // Required by Vulkan-Hpp's generated wrappers when they validate the dispatch object.
    [[nodiscard]] std::uint32_t getVkHeaderVersion() const { return VK_HEADER_VERSION; }

    [[nodiscard]] bool Load(vk::Instance instance, vk::Device device);
};

} // namespace mukki::vulkan
