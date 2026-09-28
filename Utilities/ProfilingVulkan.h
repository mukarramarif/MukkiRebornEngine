#pragma once

// GPU-side Tracy profiling wrapper. Include this header only after the Vulkan headers.
// When MRE_ENABLE_TRACY is defined it provides the tracy::VkCtx / tracy::VkCtxScope types
// used by the Vulkan backend to emit and collect GPU timestamp zones. Without
// MRE_ENABLE_TRACY the Tracy headers are not available at all, so everything is excluded.

#include "Utilities/Profiling.h"

#ifdef MRE_ENABLE_TRACY
#include <tracy/TracyVulkan.hpp>
#endif
