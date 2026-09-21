#include "Core/Engine.h"

#include <algorithm>
#include <string_view>

#include <gtest/gtest.h>

namespace
{

TEST(Engine, VersionStringHasThreeComponents)
{
    const auto version = mukki::GetVersionString();
    EXPECT_EQ(std::count(version.begin(), version.end(), '.'), 2);
}

TEST(Engine, NameIsNotEmpty)
{
    ASSERT_NE(mukki::kEngineName, nullptr);
    EXPECT_GT(std::string_view(mukki::kEngineName).size(), 0u);
}

} // namespace
