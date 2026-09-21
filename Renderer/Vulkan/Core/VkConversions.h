#pragma once

#include <vulkan/vulkan.hpp>

#include "Renderer/RHI/Resources.h"

namespace mukki::vulkan
{

[[nodiscard]] vk::Format ToVkFormat(rhi::Format format);
[[nodiscard]] rhi::Format FromVkFormat(vk::Format format);
[[nodiscard]] vk::PresentModeKHR ToVkPresentMode(rhi::PresentMode mode);

} // namespace mukki::vulkan
