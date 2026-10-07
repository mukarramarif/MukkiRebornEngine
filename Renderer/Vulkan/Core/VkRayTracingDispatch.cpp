#include "Renderer/Vulkan/Core/VkRayTracingDispatch.h"

namespace mukki::vulkan
{

bool RayTracingDispatch::Load(vk::Instance instance, vk::Device device)
{
    const auto getDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        ::vkGetInstanceProcAddr(instance, "vkGetDeviceProcAddr"));
    if (getDeviceProcAddr == nullptr)
    {
        return false;
    }

    const auto load = [device, getDeviceProcAddr](const char* name) {
        return getDeviceProcAddr(device, name);
    };

    vkGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        load("vkGetAccelerationStructureBuildSizesKHR"));
    vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
        load("vkCreateAccelerationStructureKHR"));
    vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
        load("vkDestroyAccelerationStructureKHR"));
    vkGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        load("vkGetAccelerationStructureDeviceAddressKHR"));
    vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
        load("vkCmdBuildAccelerationStructuresKHR"));

    return vkGetAccelerationStructureBuildSizesKHR != nullptr &&
           vkCreateAccelerationStructureKHR != nullptr &&
           vkDestroyAccelerationStructureKHR != nullptr &&
           vkGetAccelerationStructureDeviceAddressKHR != nullptr &&
           vkCmdBuildAccelerationStructuresKHR != nullptr;
}

} // namespace mukki::vulkan
