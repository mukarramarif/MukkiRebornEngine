#pragma once

#include <string>

#ifndef MRE_VERSION_MAJOR
#define MRE_VERSION_MAJOR 0
#endif

#ifndef MRE_VERSION_MINOR
#define MRE_VERSION_MINOR 0
#endif

#ifndef MRE_VERSION_PATCH
#define MRE_VERSION_PATCH 0
#endif

namespace mukki
{

inline constexpr const char* kEngineName = "MukkiRebornEngine";

[[nodiscard]] std::string GetVersionString();

} // namespace mukki
