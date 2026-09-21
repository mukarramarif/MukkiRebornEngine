#pragma once

#include <cstdint>
#include <format>
#include <source_location>
#include <string>
#include <string_view>

namespace mukki
{

enum class LogLevel : std::uint8_t
{
    Trace = 0,
    Debug,
    Info,
    Warn,
    Error,
    Critical,
    Off,
};

struct LogConfig
{
    LogLevel consoleLevel = LogLevel::Trace;
    LogLevel fileLevel = LogLevel::Trace;
    std::string filePath = "MukkiRebornEngine.log";
    bool enableFile = true;
    bool enableColors = true;
};

class Log
{
public:
    Log() = delete;

    static void Init(const LogConfig& config);
    static void Shutdown();

    static void SetConsoleLevel(LogLevel level);
    static void SetFileLevel(LogLevel level);
    static void SetColorsEnabled(bool enabled);

    template <typename... Args>
    static void Write(
        LogLevel level,
        const std::source_location& location,
        std::format_string<Args...> format,
        Args&&... args)
    {
        WriteImpl(level, location, std::format(format, std::forward<Args>(args)...));
    }

    static void WriteMessage(LogLevel level, const std::source_location& location, std::string_view message);

private:
    static void WriteImpl(LogLevel level, const std::source_location& location, const std::string& message);
};

} // namespace mukki

#define MRE_TRACE(...)                                                                                                 \
    ::mukki::Log::Write(::mukki::LogLevel::Trace, std::source_location::current(), __VA_ARGS__)
#define MRE_DEBUG(...)                                                                                                 \
    ::mukki::Log::Write(::mukki::LogLevel::Debug, std::source_location::current(), __VA_ARGS__)
#define MRE_INFO(...)                                                                                                  \
    ::mukki::Log::Write(::mukki::LogLevel::Info, std::source_location::current(), __VA_ARGS__)
#define MRE_WARN(...)                                                                                                  \
    ::mukki::Log::Write(::mukki::LogLevel::Warn, std::source_location::current(), __VA_ARGS__)
#define MRE_ERROR(...)                                                                                                 \
    ::mukki::Log::Write(::mukki::LogLevel::Error, std::source_location::current(), __VA_ARGS__)
#define MRE_CRITICAL(...)                                                                                              \
    ::mukki::Log::Write(::mukki::LogLevel::Critical, std::source_location::current(), __VA_ARGS__)
