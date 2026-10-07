#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace mukki
{

// Linear RGB Radiance image (Radiance .hdr / RGBE decoded to float). Row 0 is the top of the image,
// matching the equirectangular convention used by the environment sampler.
struct HdrImage
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> pixels{}; // width * height * 3 floats
};

// Loads a Radiance .hdr file (both flat and RLE encoded scanlines). Returns false on parse errors.
[[nodiscard]] bool LoadHdrImage(const std::filesystem::path& path, HdrImage& outImage);

// Packs the RGB float pixels into RGBA16F (4 half floats per pixel) for texture upload.
[[nodiscard]] std::vector<std::uint16_t> PackHdrToRgba16(const HdrImage& image);

// IEEE 754 binary16 conversion used by PackHdrToRgba16.
[[nodiscard]] std::uint16_t FloatToHalf(float value);

} // namespace mukki
