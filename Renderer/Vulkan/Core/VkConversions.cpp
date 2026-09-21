#include "Renderer/Vulkan/Core/VkConversions.h"

#include "Utilities/Assert.h"

namespace mukki::vulkan
{

vk::Format ToVkFormat(rhi::Format format)
{
    switch (format)
    {
    case rhi::Format::R8G8B8A8Unorm:
        return vk::Format::eR8G8B8A8Unorm;
    case rhi::Format::R8G8B8A8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case rhi::Format::B8G8R8A8Unorm:
        return vk::Format::eB8G8R8A8Unorm;
    case rhi::Format::B8G8R8A8Srgb:
        return vk::Format::eB8G8R8A8Srgb;
    case rhi::Format::R16G16B16A16Sfloat:
        return vk::Format::eR16G16B16A16Sfloat;
    case rhi::Format::D32Sfloat:
        return vk::Format::eD32Sfloat;
    case rhi::Format::Undefined:
        return vk::Format::eUndefined;
    }

    MRE_UNREACHABLE();
}

rhi::Format FromVkFormat(vk::Format format)
{
    switch (format)
    {
    case vk::Format::eR8G8B8A8Unorm:
        return rhi::Format::R8G8B8A8Unorm;
    case vk::Format::eR8G8B8A8Srgb:
        return rhi::Format::R8G8B8A8Srgb;
    case vk::Format::eB8G8R8A8Unorm:
        return rhi::Format::B8G8R8A8Unorm;
    case vk::Format::eB8G8R8A8Srgb:
        return rhi::Format::B8G8R8A8Srgb;
    case vk::Format::eR16G16B16A16Sfloat:
        return rhi::Format::R16G16B16A16Sfloat;
    case vk::Format::eD32Sfloat:
        return rhi::Format::D32Sfloat;
    default:
        return rhi::Format::Undefined;
    }
}

vk::PresentModeKHR ToVkPresentMode(rhi::PresentMode mode)
{
    switch (mode)
    {
    case rhi::PresentMode::Fifo:
        return vk::PresentModeKHR::eFifo;
    case rhi::PresentMode::Mailbox:
        return vk::PresentModeKHR::eMailbox;
    case rhi::PresentMode::Immediate:
        return vk::PresentModeKHR::eImmediate;
    }

    MRE_UNREACHABLE();
}

} // namespace mukki::vulkan
