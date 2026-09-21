#include "Utilities/PathUtils.h"

namespace mukki
{

std::string PathToUtf8(const std::filesystem::path& path)
{
    const std::u8string utf8 = path.u8string();
    return std::string{reinterpret_cast<const char*>(utf8.data()), utf8.size()};
}

} // namespace mukki
