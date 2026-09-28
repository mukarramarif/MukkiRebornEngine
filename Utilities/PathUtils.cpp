#include "Utilities/PathUtils.h"

namespace mukki
{

std::string PathToUtf8(const std::filesystem::path& path)
{
    const std::u8string utf8 = path.u8string();
    return std::string{reinterpret_cast<const char*>(utf8.data()), utf8.size()};
}

std::filesystem::path PathFromUtf8(std::string_view utf8)
{
    const std::u8string wide{reinterpret_cast<const char8_t*>(utf8.data()), utf8.size()};
    return std::filesystem::path{wide};
}

} // namespace mukki
