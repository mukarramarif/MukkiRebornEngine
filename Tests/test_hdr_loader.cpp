#include "Assets/HdrLoader.h"

#include <cstdint>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

namespace
{

// Writes a tiny flat (uncompressed scanline) Radiance file with known pixels. Row 0 is the top.
std::filesystem::path WriteTestHdr()
{
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "mukki_test.hdr";

    std::ofstream stream{path, std::ios::binary};
    stream << "#?RADIANCE\n";
    stream << "FORMAT=32-bit_rle_rgbe\n";
    stream << "\n";
    stream << "-Y 2 +X 2\n";

    // RGBE encoded pixels: (1, 0, 0), (0, 2, 0), (0, 0, 4), (8, 8, 8) at exposure 0.
    // The exponent byte stores value * 2^(e - 136).
    const std::uint8_t rgbe[4 * 4] = {
        128, 0, 0, 129,
        0, 128, 0, 130,
        0, 0, 128, 131,
        128, 128, 128, 133};

    stream.write(reinterpret_cast<const char*>(rgbe), sizeof(rgbe));
    stream.close();
    return path;
}

TEST(HdrLoader, DecodesFlatScanlines)
{
    const std::filesystem::path path = WriteTestHdr();
    mukki::HdrImage image{};
    ASSERT_TRUE(mukki::LoadHdrImage(path, image));
    std::filesystem::remove(path);

    ASSERT_EQ(image.width, 2u);
    ASSERT_EQ(image.height, 2u);
    ASSERT_EQ(image.pixels.size(), 2u * 2u * 3u);

    // 128 * 2^(129-136) = 1, 128 * 2^(130-136) = 2, 128 * 2^(131-136) = 4.
    EXPECT_NEAR(image.pixels[0], 1.0f, 1e-5f);
    EXPECT_NEAR(image.pixels[1], 0.0f, 1e-5f);
    EXPECT_NEAR(image.pixels[2], 0.0f, 1e-5f);
    EXPECT_NEAR(image.pixels[4], 2.0f, 1e-5f);
    EXPECT_NEAR(image.pixels[8], 4.0f, 1e-5f);
    // 128 * 2^(133 - 136) = 16.
    EXPECT_NEAR(image.pixels[9], 16.0f, 1e-3f);
    EXPECT_NEAR(image.pixels[10], 16.0f, 1e-3f);
    EXPECT_NEAR(image.pixels[11], 16.0f, 1e-3f);
}

TEST(HdrLoader, PacksRgba16)
{
    // 1.0, 0.5, 0.25, 2.0 round trip through binary16.
    EXPECT_EQ(mukki::FloatToHalf(1.0f), 0x3C00u);
    EXPECT_EQ(mukki::FloatToHalf(0.5f), 0x3800u);
    EXPECT_EQ(mukki::FloatToHalf(0.0f), 0x0000u);
    EXPECT_EQ(mukki::FloatToHalf(-1.0f), 0xBC00u);
    EXPECT_EQ(mukki::FloatToHalf(-2.0f), 0xC000u);

    mukki::HdrImage image{};
    image.width = 1;
    image.height = 1;
    image.pixels = {1.0f, 0.5f, 0.25f};
    const std::vector<std::uint16_t> packed = mukki::PackHdrToRgba16(image);
    ASSERT_EQ(packed.size(), 4u);
    EXPECT_EQ(packed[0], 0x3C00u);
    EXPECT_EQ(packed[1], 0x3800u);
    EXPECT_EQ(packed[2], 0x3400u);
    EXPECT_EQ(packed[3], 0x3C00u);
}

} // namespace
