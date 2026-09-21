#pragma once

#include <cstdint>

namespace mukki::rhi
{

enum class Format : std::uint8_t
{
    Undefined = 0,
    R8G8B8A8Unorm,
    R8G8B8A8Srgb,
    B8G8R8A8Unorm,
    B8G8R8A8Srgb,
    R16G16B16A16Sfloat,
    D32Sfloat,
};

enum class PresentMode : std::uint8_t
{
    Fifo = 0,
    Mailbox,
    Immediate,
};

struct Extent2D
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct ClearColor
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 1.0f;
};

} // namespace mukki::rhi
