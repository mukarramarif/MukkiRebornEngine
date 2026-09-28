#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "Scene/SceneDescription.h"

namespace mukki
{

// Reads and writes the human editable scene format. Relative model paths are resolved against
// SceneDescription::assetRoot, which in turn defaults to the directory of the JSON file.
class SceneJson
{
public:
    [[nodiscard]] static bool Load(
        const std::filesystem::path& path,
        SceneDescription& scene,
        std::string& error);

    [[nodiscard]] static bool Parse(
        std::string_view text,
        const std::filesystem::path& baseDirectory,
        SceneDescription& scene,
        std::string& error);

    [[nodiscard]] static bool Save(
        const std::filesystem::path& path,
        const SceneDescription& scene,
        std::string& error);

    [[nodiscard]] static std::string Serialize(const SceneDescription& scene);
};

} // namespace mukki
