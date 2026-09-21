#include "Utilities/Assert.h"

#include <cstdlib>

namespace mukki
{

void OnAssertFailed(const char* expression, std::string_view message, const std::source_location& location)
{
    if (message.empty())
    {
        Log::WriteMessage(LogLevel::Critical, location, std::format("Assertion failed: {}", expression));
    }
    else
    {
        Log::WriteMessage(LogLevel::Critical, location, std::format("Assertion failed: {} | {}", expression, message));
    }

    MRE_DEBUG_BREAK();
    std::abort();
}

} // namespace mukki
