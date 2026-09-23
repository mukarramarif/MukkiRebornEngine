#pragma once

// CPU-side Tracy profiling macros. Include this header in any translation unit that
// should emit profiling zones and use the MRE_PROFILE_* macros.
//
// When MRE_ENABLE_TRACY is defined (see the *-tracy CMake presets), the macros map to
// Tracy instrumentation. Otherwise they expand to no-ops with zero overhead.

#ifdef MRE_ENABLE_TRACY

#ifndef TRACY_ENABLE
#define TRACY_ENABLE
#endif
#include <tracy/Tracy.hpp>

#define MRE_PROFILE_ZONE() ZoneScoped
#define MRE_PROFILE_ZONE_N(name) ZoneScopedN(name)
#define MRE_PROFILE_FRAME() FrameMark
#define MRE_PROFILE_FRAME_N(name) FrameMarkNamed(name)
#define MRE_PROFILE_SET_THREAD_NAME(name) ::tracy::SetThreadName(name)

#else

#define MRE_PROFILE_ZONE() (void)0
#define MRE_PROFILE_ZONE_N(name) (void)0
#define MRE_PROFILE_FRAME() (void)0
#define MRE_PROFILE_FRAME_N(name) (void)0
#define MRE_PROFILE_SET_THREAD_NAME(name) (void)0

#endif
