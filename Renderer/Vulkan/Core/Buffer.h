#pragma once
#include "vulkan/vulkan.h"

struct Buffer{
    VkBuffer vk_buf;
    VmaAllocation vma_alloc;
    VKDeviceMemory vk_device_memory;

}
