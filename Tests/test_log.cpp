#include "Utilities/Log.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

namespace
{

std::filesystem::path TempLogPath(std::string_view name)
{
    return std::filesystem::temp_directory_path() / std::string{name};
}

std::string ReadFile(const std::filesystem::path& path)
{
    std::ifstream stream{path};
    std::ostringstream contents;
    contents << stream.rdbuf();
    return contents.str();
}

TEST(Log, WritesMessagesToFile)
{
    const std::filesystem::path path = TempLogPath("mukki_log_test.txt");
    std::filesystem::remove(path);

    mukki::LogConfig config{};
    config.filePath = path.string();
    config.consoleLevel = mukki::LogLevel::Off;
    mukki::Log::Init(config);

    MRE_INFO("marker {}", 42);
    mukki::Log::Shutdown();

    const std::string contents = ReadFile(path);
    EXPECT_NE(contents.find("marker 42"), std::string::npos);
}

TEST(Log, RespectsFileLevelFilter)
{
    const std::filesystem::path path = TempLogPath("mukki_log_filter_test.txt");
    std::filesystem::remove(path);

    mukki::LogConfig config{};
    config.filePath = path.string();
    config.consoleLevel = mukki::LogLevel::Off;
    config.fileLevel = mukki::LogLevel::Error;
    mukki::Log::Init(config);

    MRE_INFO("filtered out");
    MRE_ERROR("kept");
    mukki::Log::Shutdown();

    const std::string contents = ReadFile(path);
    EXPECT_EQ(contents.find("filtered out"), std::string::npos);
    EXPECT_NE(contents.find("kept"), std::string::npos);
}

} // namespace
