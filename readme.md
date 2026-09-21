# MukkiRebornEngine

A vendor-agnostic game engine: C++20, Vulkan 1.4 (vulkan.hpp + VMA), deferred rendering, Slang
shaders and an ImGui docking UI. Rewrite of an older Vulkan renderer into a proper engine.

## Requirements

- CMake 3.25 or newer
- [vcpkg](https://github.com/microsoft/vcpkg), with the `VCPKG_ROOT` environment variable set
- Visual Studio 2022+ with the "Desktop development with C++" workload (Windows), or a C++20
  compiler with Ninja (Linux)
- Vulkan SDK 1.4 or newer, for `slangc` (Vulkan headers and the loader come from vcpkg)
- Git

## Setup

Set `VCPKG_ROOT` once so the CMake presets can find the vcpkg toolchain:

```powershell
# Windows (persistent, restart your terminal afterwards)
[Environment]::SetEnvironmentVariable('VCPKG_ROOT', 'C:\path\to\vcpkg', 'User')
```

```sh
# Linux
echo 'export VCPKG_ROOT=$HOME/vcpkg' >> ~/.bashrc
```

Dependencies are declared in `vcpkg.json` and installed automatically in manifest mode on the first
configure into `build/<preset>/vcpkg_installed`. That first configure can take several minutes while
vcpkg builds the dependencies.

Shader compilation uses `slangc` from the Vulkan SDK, found through the `VULKAN_SDK` environment
variable (or on `PATH`). If `slangc` is missing, CMake prints a warning, shader targets are skipped
and the sandbox will fail to load its shaders.

## Build

Windows (Visual Studio generator, Debug or Release):

```powershell
cmake --preset windows-debug
cmake --build --preset windows-debug
ctest --preset windows-debug
```

Linux (Ninja):

```sh
sudo apt-get install ninja-build xorg-dev libgl1-mesa-dev

cmake --preset linux-debug
cmake --build --preset linux-debug
ctest --preset linux-debug
```

Use the `*-release` presets for optimized builds. The presets in `CMakePresets.json` use the
"Visual Studio 18 2026" generator; with another Visual Studio version, configure manually instead:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

CMake options:

- `MRE_BUILD_SANDBOX` (default `ON`) - build the sandbox executable
- `MRE_BUILD_TESTS` (default `ON`) - build the GoogleTest unit tests

## Run

The sandbox renders a deferred scene (G-buffer pass + PBR lighting pass with ACES tonemapping) with
an ImGui overlay, including a debug window showing the G-buffer channels. Compiled Slang shaders are
copied next to the executable automatically.

```powershell
.\build\windows-debug\Debug\mukki_sandbox.exe
```

Pass `--frames N` to run a fixed number of frames and exit (used for smoke testing).

Controls: `WASD`/`QE` to move, hold the right mouse button to look, `Shift` to move faster, `Esc` to
quit.

## Layout

- `Application/` - windowing, input, keybinds, camera, ImGui styles
- `Core/` - engine core (version, time)
- `Renderer/RHI/` - vendor-agnostic render hardware interface (device, command lists, resources)
- `Renderer/Vulkan/` - Vulkan backend implementation
- `Renderer/Deferred/` - deferred renderer (G-buffer, lighting, tonemapping)
- `Renderer/Shaders/` - Slang shaders compiled at build time
- `Renderer/BufferManager` - vertex/index/uniform buffer management
- `Utilities/` - logging, asserts, future containers and allocators
- `Tests/` - GoogleTest unit tests
- `Sandbox/` - sandbox executable

`Renderer/DX12/` and a render graph are planned.
