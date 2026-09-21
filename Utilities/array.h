#pragma once

// TODO: Custom dynamic array intended to replace std::vector in hot paths.
// Planned API:
//  - contiguous storage with geometric (2x) growth
//  - allocator aware construction (see memory.h)
//  - emplace_back / push_back / reserve / resize / data / size / begin / end
//  - std::span interop
//  - debug-only bounds checking via MRE_ASSERT
// For now the engine uses std::vector; only implement this once profiling shows a real need.
