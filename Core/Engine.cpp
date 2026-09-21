#include "Core/Engine.h"

#include <format>

namespace mukki
{

std::string GetVersionString()
{
    return std::format("{}.{}.{}", MRE_VERSION_MAJOR, MRE_VERSION_MINOR, MRE_VERSION_PATCH);
}

} // namespace mukki
