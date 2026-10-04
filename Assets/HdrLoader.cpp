#include "Assets/HdrLoader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "Utilities/Log.h"
#include "Utilities/PathUtils.h"

namespace mukki
{
namespace
{

constexpr float kPi = 3.14159265358979323846f;

bool ReadLine(std::ifstream& stream, std::string& outLine)
{
    if (!std::getline(stream, outLine))
    {
        return false;
    }
    if (!outLine.empty() && outLine.back() == '\r')
    {
        outLine.pop_back();
    }
    return true;
}

bool IsRleScanline(const std::uint8_t* header, std::uint32_t width)
{
    return header[0] == 2 && header[1] == 2 && header[2] == static_cast<std::uint8_t>(width >> 8) &&
           header[3] == static_cast<std::uint8_t>(width & 0xFF) && width >= 8 && width < 32768;
}

void RgbeToFloat(const std::uint8_t rgbe[4], float* outRgb)
{
    if (rgbe[3] == 0)
    {
        outRgb[0] = 0.0f;
        outRgb[1] = 0.0f;
        outRgb[2] = 0.0f;
        return;
    }

    const float scale = std::ldexp(1.0f, static_cast<int>(rgbe[3]) - (128 + 8));
    outRgb[0] = static_cast<float>(rgbe[0]) * scale;
    outRgb[1] = static_cast<float>(rgbe[1]) * scale;
    outRgb[2] = static_cast<float>(rgbe[2]) * scale;
}

} // namespace

std::uint16_t FloatToHalf(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23) & 0xFFu;
    const std::uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent == 0xFFu)
    {
        // Infinity or NaN (keep at least one mantissa bit for NaN).
        return static_cast<std::uint16_t>(sign | 0x7C00u | (mantissa != 0 ? 0x0200u : 0u));
    }

    const int newExponent = static_cast<int>(exponent) - 127 + 15;
    if (newExponent >= 0x1F)
    {
        return static_cast<std::uint16_t>(sign | 0x7C00u); // overflow to infinity
    }
    if (newExponent <= 0)
    {
        if (newExponent < -10)
        {
            return static_cast<std::uint16_t>(sign);
        }
        const std::uint32_t shifted = (mantissa | 0x800000u) >> (1 - newExponent);
        const std::uint32_t rounded = shifted + 0x1000u;
        return static_cast<std::uint16_t>(sign | (rounded >> 13));
    }

    const std::uint32_t rounded = mantissa + 0x1000u;
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(newExponent) << 10) | ((rounded >> 13) & 0x03FFu));
}

bool LoadHdrImage(const std::filesystem::path& path, HdrImage& outImage)
{
    std::ifstream stream{path, std::ios::binary};
    if (!stream)
    {
        MRE_ERROR("Failed to open the HDR file '{}'", PathToUtf8(path));
        return false;
    }

    std::string line{};
    bool formatOk = false;
    bool flipVertically = false;
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    // Header directives until the empty line.
    while (ReadLine(stream, line))
    {
        if (line.empty())
        {
            break;
        }
        if (line.rfind("FORMAT=", 0) == 0)
        {
            formatOk = line.find("32-bit_rle_rgbe") != std::string::npos;
        }
    }

    // The resolution line follows the blank line: "-Y H +X W" (top-down) or "+Y H +X W" (bottom-up).
    if (!ReadLine(stream, line))
    {
        MRE_ERROR("Missing HDR resolution line in '{}'", PathToUtf8(path));
        return false;
    }
    if (std::sscanf(line.c_str(), "-Y %u +X %u", &height, &width) == 2)
    {
        flipVertically = false;
    }
    else if (std::sscanf(line.c_str(), "+Y %u +X %u", &height, &width) == 2)
    {
        flipVertically = true;
    }
    else
    {
        MRE_ERROR("Unsupported HDR resolution line '{}'", line);
        return false;
    }

    if (!formatOk || width == 0 || height == 0)
    {
        MRE_ERROR("Invalid HDR header in '{}'", PathToUtf8(path));
        return false;
    }

    std::vector<std::uint8_t> rgbe(static_cast<std::size_t>(width) * height * 4);
    std::vector<std::uint8_t> scanline(static_cast<std::size_t>(width) * 4);

    for (std::uint32_t row = 0; row < height; ++row)
    {
        std::uint8_t header[4]{};
        stream.read(reinterpret_cast<char*>(header), 4);
        if (!stream)
        {
            MRE_ERROR("Truncated HDR file '{}'", PathToUtf8(path));
            return false;
        }

        if (IsRleScanline(header, width))
        {
            for (std::uint32_t channel = 0; channel < 4; ++channel)
            {
                std::uint32_t x = 0;
                while (x < width)
                {
                    std::uint8_t count = 0;
                    stream.read(reinterpret_cast<char*>(&count), 1);
                    if (!stream)
                    {
                        MRE_ERROR("Truncated HDR RLE data in '{}'", PathToUtf8(path));
                        return false;
                    }

                    if (count > 128)
                    {
                        std::uint8_t value = 0;
                        stream.read(reinterpret_cast<char*>(&value), 1);
                        const std::uint32_t runLength = std::min<std::uint32_t>(count - 128, width - x);
                        for (std::uint32_t run = 0; run < runLength; ++run)
                        {
                            scanline[static_cast<std::size_t>(x + run) * 4 + channel] = value;
                        }
                        x += runLength;
                    }
                    else
                    {
                        const std::uint32_t literalLength = std::min<std::uint32_t>(count, width - x);
                        for (std::uint32_t literal = 0; literal < literalLength; ++literal)
                        {
                            std::uint8_t value = 0;
                            stream.read(reinterpret_cast<char*>(&value), 1);
                            scanline[static_cast<std::size_t>(x + literal) * 4 + channel] = value;
                        }
                        x += literalLength;
                    }
                }
            }
        }
        else
        {
            // Flat scanline: the 4 header bytes are the first pixel, the rest is raw RGBe.
            std::memcpy(scanline.data(), header, 4);
            if (width > 1)
            {
                stream.read(
                    reinterpret_cast<char*>(scanline.data() + 4),
                    static_cast<std::streamsize>((width - 1) * 4));
                if (!stream)
                {
                    MRE_ERROR("Truncated HDR scanline in '{}'", PathToUtf8(path));
                    return false;
                }
            }
        }

        const std::uint32_t targetRow = flipVertically ? (height - 1 - row) : row;
        std::memcpy(
            rgbe.data() + static_cast<std::size_t>(targetRow) * width * 4,
            scanline.data(),
            static_cast<std::size_t>(width) * 4);
    }

    outImage.width = width;
    outImage.height = height;
    outImage.pixels.assign(static_cast<std::size_t>(width) * height * 3, 0.0f);
    float maxRadiance = 0.0f;
    for (std::size_t index = 0; index < static_cast<std::size_t>(width) * height; ++index)
    {
        RgbeToFloat(rgbe.data() + index * 4, outImage.pixels.data() + index * 3);
        maxRadiance = std::max(
            {maxRadiance,
             outImage.pixels[index * 3 + 0],
             outImage.pixels[index * 3 + 1],
             outImage.pixels[index * 3 + 2]});
    }

    MRE_INFO(
        "Loaded HDR environment '{}' ({}x{}, max radiance {:.1f})",
        PathToUtf8(path),
        width,
        height,
        maxRadiance);
    return true;
}

std::vector<std::uint16_t> PackHdrToRgba16(const HdrImage& image)
{
    std::vector<std::uint16_t> packed(static_cast<std::size_t>(image.width) * image.height * 4);
    const std::size_t pixelCount = static_cast<std::size_t>(image.width) * image.height;
    for (std::size_t index = 0; index < pixelCount; ++index)
    {
        packed[index * 4 + 0] = FloatToHalf(image.pixels[index * 3 + 0]);
        packed[index * 4 + 1] = FloatToHalf(image.pixels[index * 3 + 1]);
        packed[index * 4 + 2] = FloatToHalf(image.pixels[index * 3 + 2]);
        packed[index * 4 + 3] = FloatToHalf(1.0f);
    }
    return packed;
}

} // namespace mukki
