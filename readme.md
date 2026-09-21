# MukkiRebornEngine

A vendor-agnostic game engine: C++20, Vulkan 1.4 (vulkan.hpp + VMA), deferred rendering, Slang
shaders and an ImGui docking UI. Rewrite of an older Vulkan renderer into a proper engine.

## Requirements

- CMake 3.25+
- vcpkg, with the `VCPKG_ROOT` environment variable pointing at it
- Visual Studio 2022+ (Windows) or a C++20 compiler with Ninja (Linux)
- Vulkan SDK 1.4+ (provides `slangc` for shader compilation)

## Build

```sh
cmake --preset windows-debug
cmake --build --preset windows-debug
ctest --preset windows-debug
```

Use `windows-release` / `linux-debug` / `linux-release` presets accordingly.

## Layout

- `Application/` - windowing, input, keybinds, camera, ImGui layers
- `Core/` - engine core
- `Renderer/RHI/` - vendor-agnostic render hardware interface
- `Renderer/Vulkan/` - Vulkan backend implementation
- `Renderer/Deferred/` - deferred renderer (G-buffer, lighting, tonemapping)
- `Utilities/` - logging, asserts, future containers and allocators
- `Tests/` - GoogleTest unit tests
- `Sandbox/` - sandbox executable
