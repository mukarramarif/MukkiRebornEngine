#pragma once

// TODO: CPU-side memory management utilities used by the engine.
// Planned API:
//  - arena/linear allocator for per-frame scratch memory
//  - stack allocator for nested scopes
//  - pool allocator for fixed-size objects
//  - aligned allocation helpers built on std::aligned_alloc
//  - allocation tracking hooks for the debug layer
// GPU memory is handled by VMA in Renderer/Vulkan/Resources/BufferAllocator.h.
// For now the engine uses std allocators; only implement this once profiling shows a real need.
