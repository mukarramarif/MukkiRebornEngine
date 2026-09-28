#pragma once

#include <vulkan/vulkan.hpp>

#include "Renderer/RHI/Resources.h"

namespace mukki::vulkan
{

[[nodiscard]] vk::Format ToVkFormat(rhi::Format format);
[[nodiscard]] rhi::Format FromVkFormat(vk::Format format);
[[nodiscard]] vk::PresentModeKHR ToVkPresentMode(rhi::PresentMode mode);
[[nodiscard]] vk::BufferUsageFlags ToVkBufferUsage(rhi::BufferUsage usage);
[[nodiscard]] vk::ShaderStageFlagBits ToVkShaderStage(rhi::ShaderStage stage);
[[nodiscard]] vk::ShaderStageFlags ToVkShaderStageMask(rhi::ShaderStageMask mask);
[[nodiscard]] vk::PipelineStageFlags2 ToVkPipelineStageFlags2(rhi::ShaderStageMask mask);
[[nodiscard]] vk::CullModeFlags ToVkCullMode(rhi::CullMode mode);
[[nodiscard]] vk::DescriptorType ToVkDescriptorType(rhi::DescriptorType type);

} // namespace mukki::vulkan
