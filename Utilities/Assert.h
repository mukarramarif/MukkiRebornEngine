#pragma once

#include <format>
#include <source_location>
#include <string_view>

#include "Utilities/Log.h"

#if defined(_MSC_VER)
#define MRE_DEBUG_BREAK() __debugbreak()
#else
#define MRE_DEBUG_BREAK() __builtin_trap()
#endif

namespace mukki
{

[[noreturn]] void OnAssertFailed(const char* expression, std::string_view message, const std::source_location& location);

} // namespace mukki

#if defined(MRE_DISABLE_ASSERTS) || defined(NDEBUG)
#define MRE_ASSERT(expr, ...) ((void)0)
#define MRE_VERIFY(expr, ...) (static_cast<void>(expr))
#else
#define MRE_ASSERT(expr, ...)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(expr)) [[unlikely]]                                                                                      \
        {                                                                                                              \
            ::mukki::OnAssertFailed(                                                                                   \
                #expr, __VA_OPT__(std::format(__VA_ARGS__)), std::source_location::current());                         \
        }                                                                                                              \
    } while (false)
#define MRE_VERIFY(expr, ...)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        const bool mreVerified = static_cast<bool>(expr);                                                              \
        MRE_ASSERT(mreVerified, __VA_ARGS__);                                                                          \
    } while (false)
#endif

#define MRE_UNREACHABLE()                                                                                              \
    ::mukki::OnAssertFailed("unreachable", "unreachable code reached", std::source_location::current())
