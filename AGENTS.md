# MukkiRebornEngine - Agent Guide

## Project

Rewrite of a Vulkan renderer into a proper game engine (C++20, Vulkan 1.4, deferred rendering,
multithreading in mind, Slang shaders). The engine is vendor agnostic: the RHI in `Renderer/RHI/`
must not expose backend types, and Vulkan types may only appear under `Renderer/Vulkan/`.
`Renderer/DX12/` is planned for the future.

## Locked decisions

- Windowing: GLFW. Math: glm. Containers: std.
- `Utilities/array.h`, `linkedlist.h` and `memory.h` are placeholders; implement later only if
  profiling justifies custom containers/allocators.
- Vulkan: vulkan.hpp (RAII) + VMA (`vulkan-memory-allocator`; the `vulkan-memory-allocator-hpp`
  port is incompatible with VMA 3.4.0 and must not be used).
- Modern Vulkan path: dynamic rendering, synchronization2, timeline semaphores; no
  `VkRenderPass`/`VkFramebuffer`.
- 2 frames in flight, no MSAA initially.
- Slang shaders compiled at build time with `slangc` (found through `$ENV{VULKAN_SDK}`).
- The engine will be using tinygltf and ktx libraries for loading glTF and KTX2 assets, respectively. The engine will support PBR materials and textures, as well as animation and skinning. tinygltf is pinned to 2.9.3 via a `vcpkg.json` override and served from `overlay-ports/tinygltf` because the registry port hashes are stale after GitHub regenerated its release tarballs. Sample models live in the shallow `ThirdParty/glTF-Sample-Models` submodule.
- Scenes created in the engine are to stored as json and loaded at runtime. The engine will support a scene graph and entity-component-system architecture for managing game objects and their components. The json will be serialized and deserialized using the nlohmann/json library.
- Namespace `mukki`; PascalCase files/types/methods; `m_` prefixed members.
- Multithreading: `Core/JobSystem` is the shared thread pool. glTF parsing runs asynchronously
  (`Assets/AsyncGltfLoader`) and the G-buffer draw calls are recorded into secondary command buffers
  on worker threads, then replayed from the primary command list with `ExecuteCommands`.
- Bindless: descriptor indexing is a hard device requirement. Textures get a bindless index from the
  device and are bound into a global update-after-bind `SampledImage` array set; shaders sample them
  through `Texture2D textures[]` with `NonUniformResourceIndex`.
- Lighting: clustered deferred. `Renderer/Lighting/` holds the light types (`LightType`/`Light`) and
  the cluster grid (`kClusterTileSize`, `kClusterZSliceCount`, `kMaxLightsPerCluster`, `kMaxLights`).
  `Renderer/Shaders/cluster_culling.comp.slang` bins the lights submitted through
  `DeferredRenderer::SetLights` into an exponential-slice cluster grid, and the lighting pass iterates
  the per-cluster light lists with a Cook-Torrance BRDF. Directional, point and spot lights are
  supported; lights are uploaded as an SSBO (`GpuLight`, packed float4s). `SetClusterDebugView`
  renders a light-count heatmap for debugging. Shadows (CSM and spot shadow maps) are not implemented
  yet.
- Profiling: Tracy (on-demand). Use the `MRE_PROFILE_*` macros from `Utilities/Profiling.h` for CPU
  zones and `rhi::CommandList::BeginGpuZone/EndGpuZone` for GPU pass zones. Enabled via the
  `*-tracy` CMake presets (`MRE_ENABLE_TRACY=ON`, `VCPKG_MANIFEST_FEATURES=tracy`); connect with the
  Tracy GUI.
- Build: CMake + vcpkg manifest (`vcpkg.json`), presets in `CMakePresets.json`.
- Tests: GoogleTest under `Tests/`. CI: GitHub Actions. UI: ImGui (docking).

## Rules

- Use .h/.cpp pairs for new files and functions.
- Always initialize variables.
- Assert assumptions in code with the engine assert macro (`Utilities/Assert.h`, phase 1).
- Implement incrementally and build/test after each change.
- For changes to utilities or core engine features, create a new branch and open a pull request.
- Keep dependencies and compiler/toolchain up to date.
