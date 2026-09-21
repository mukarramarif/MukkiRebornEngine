#include "Utilities/PathUtils.h"

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

namespace
{

TEST(PathUtils, ConvertsAsciiPath)
{
    const std::filesystem::path path{"Assets/Models/Box.gltf"};
    EXPECT_EQ(mukki::PathToUtf8(path), "Assets/Models/Box.gltf");
}

TEST(PathUtils, ConvertsNonAsciiPathWithoutThrowing)
{
    const std::filesystem::path path{u8"Unicode\u2764Test.gltf"};

    std::string utf8{};
    EXPECT_NO_THROW(utf8 = mukki::PathToUtf8(path));
    EXPECT_FALSE(utf8.empty());
    EXPECT_NE(utf8.find("Unicode"), std::string::npos);
}

} // namespace
