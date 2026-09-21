#include "Utilities/Log.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <mutex>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace mukki
{
namespace
{

std::mutex g_logMutex{};
LogConfig g_config{};
std::ofstream g_file{};
bool g_fileOpen = false;

constexpr std::string_view kLevelNames[] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"};

constexpr std::string_view kLevelColors[] = {
    "\x1b[90m",
    "\x1b[36m",
    "\x1b[32m",
    "\x1b[33m",
    "\x1b[31m",
    "\x1b[1;91m",
};

constexpr std::string_view kColorReset = "\x1b[0m";

std::string Timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::system_clock::to_time_t(now);
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm localTime{};
#ifdef _WIN32
    localtime_s(&localTime, &seconds);
#else
    localtime_r(&seconds, &localTime);
#endif

    char buffer[16]{};
    std::strftime(buffer, sizeof(buffer), "%H:%M:%S", &localTime);
    return std::format("{}.{:03}", buffer, static_cast<int>(milliseconds.count()));
}

std::string_view FileName(const char* path)
{
    const std::string_view view{path};
    const auto separator = view.find_last_of("\\/");
    return separator == std::string_view::npos ? view : view.substr(separator + 1);
}

void EnableVirtualTerminal()
{
#ifdef _WIN32
    const HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (handle != INVALID_HANDLE_VALUE && GetConsoleMode(handle, &mode) != 0)
    {
        SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
#endif
}

} // namespace

void Log::Init(const LogConfig& config)
{
    std::lock_guard lock{g_logMutex};

    if (g_file.is_open())
    {
        g_file.close();
    }

    g_config = config;
    g_fileOpen = false;

    if (g_config.enableFile)
    {
        g_file.open(g_config.filePath, std::ios::out | std::ios::trunc);
        g_fileOpen = g_file.is_open();
    }

    if (g_config.enableColors)
    {
        EnableVirtualTerminal();
    }

    if (g_config.enableFile && !g_fileOpen)
    {
        std::fprintf(stderr, "[mukki] Failed to open log file '%s'\n", g_config.filePath.c_str());
    }
}

void Log::Shutdown()
{
    std::lock_guard lock{g_logMutex};

    if (g_file.is_open())
    {
        g_file.flush();
        g_file.close();
    }

    g_fileOpen = false;
}

void Log::SetConsoleLevel(LogLevel level)
{
    std::lock_guard lock{g_logMutex};
    g_config.consoleLevel = level;
}

void Log::SetFileLevel(LogLevel level)
{
    std::lock_guard lock{g_logMutex};
    g_config.fileLevel = level;
}

void Log::SetColorsEnabled(bool enabled)
{
    std::lock_guard lock{g_logMutex};
    g_config.enableColors = enabled;
}

void Log::WriteMessage(LogLevel level, const std::source_location& location, std::string_view message)
{
    WriteImpl(level, location, std::string{message});
}

void Log::WriteImpl(LogLevel level, const std::source_location& location, const std::string& message)
{
    std::lock_guard lock{g_logMutex};

    const auto index = static_cast<std::size_t>(level);
    if (index >= std::size(kLevelNames))
    {
        return;
    }

    const bool writeConsole = level >= g_config.consoleLevel;
    const bool writeFile = g_fileOpen && level >= g_config.fileLevel;

    if (!writeConsole && !writeFile)
    {
        return;
    }

    const std::string line = std::format(
        "[{}] [{}] [{}:{}] {}\n",
        Timestamp(),
        kLevelNames[index],
        FileName(location.file_name()),
        location.line(),
        message);

    if (writeConsole)
    {
        if (g_config.enableColors)
        {
            std::fputs(std::format("{}{}{}", kLevelColors[index], line, kColorReset).c_str(), stdout);
        }
        else
        {
            std::fputs(line.c_str(), stdout);
        }
        std::fflush(stdout);
    }

    if (writeFile)
    {
        g_file << line;
        if (level >= LogLevel::Error)
        {
            g_file.flush();
        }
    }
}

} // namespace mukki
