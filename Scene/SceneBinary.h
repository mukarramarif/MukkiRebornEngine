#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "Scene/SceneDescription.h"

namespace mukki
{

// Compiled scene bytecode. The JSON scene is serialized into this compact, little endian format by
// the scene compiler and loaded directly by the engine.
class SceneBinary
{
public:
    static constexpr std::uint32_t kMagic = 0x5345524Du; // "MRES"
    static constexpr std::uint32_t kVersion = 1;

    [[nodiscard]] static std::vector<std::byte> Encode(const SceneDescription& scene);

    [[nodiscard]] static bool Decode(
        std::span<const std::byte> data,
        SceneDescription& scene,
        std::string& error);

    [[nodiscard]] static bool Write(
        const std::filesystem::path& path,
        const SceneDescription& scene,
        std::string& error);

    [[nodiscard]] static bool Read(
        const std::filesystem::path& path,
        SceneDescription& scene,
        std::string& error);
};

} // namespace mukki
