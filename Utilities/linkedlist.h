#pragma once

// TODO: Intrusive linked list plus free list used by engine internals.
// Planned API:
//  - singly linked LIFO stack (push_front / pop_front) for free lists
//  - doubly linked list with O(1) removal for registry/scene storage
//  - no per-node allocation; nodes are embedded in the stored objects
// For now the engine uses std containers; only implement this once profiling shows a real need.
