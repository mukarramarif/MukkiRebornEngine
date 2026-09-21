#pragma once

#include <filesystem>
#include <string>

namespace mukki
{

[[nodiscard]] std::string PathToUtf8(const std::filesystem::path& path);

} // namespace mukki
