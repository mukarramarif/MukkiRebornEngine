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
- Slang shaders compiled at build time with `slangc` (found through `$ENV{VULKAN_SDK}`). SPIR-V is
  reflected at load time with SPIRV-Reflect (vcpkg port `spirv-reflect`, used only under
  `Renderer/Vulkan/`); `rhi::MergeShaderReflectionBindings` and `rhi::ApplyShaderReflection` build
  descriptor set layouts, vertex input layouts and push constant ranges from the shader interface.
  Only the bindless set (update-after-bind runtime arrays) is declared by hand.
- The engine will be using tinygltf and ktx libraries for loading glTF and KTX2 assets, respectively. The engine will support PBR materials and textures, as well as animation and skinning. tinygltf is pinned to 2.9.3 via a `vcpkg.json` override and served from `overlay-ports/tinygltf` because the registry port hashes are stale after GitHub regenerated its release tarballs. Sample models live in the shallow `ThirdParty/glTF-Sample-Models` submodule.
- Scenes: authored as JSON with nlohmann/json under `Scene/` (`SceneJson`), compiled offline or
  in-tool into the little-endian `SceneBinary` bytecode (magic `MRES`), and loaded by the engine
  from that bytecode at runtime. `SceneDescription` holds glTF object instances (model path plus
  position/rotation/scale) and named lights. Compile with `mukki_scene_compiler <scene.json>
  [scene.mres]`; the sandbox compiles and loads `Sandbox/Scenes/demo.scene.json`. `DeferredRenderer
  ::LoadScene` loads/caches the referenced glTF models and instantiates them. A scene graph and
  entity-component-system architecture is still planned.
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
  renders a light-count heatmap for debugging.
- Shadows: `Renderer/Lighting/ShadowCascade.h` drives cascaded shadow maps (4x1024 `D32Sfloat`
  array) for the first directional `Light` with `castShadows`, spot shadows (4x1024 array) and
  point shadows (2 lights x six 512x512 faces packed into one 2D atlas). The shadow pass renders
  every view (cascade layer, spot layer, atlas tile) with depth-only dynamic rendering and
  secondary command buffers, culling spot/point views by light range with per-instance bounding
  spheres. The RHI supports array textures with per-layer depth views, sub-rect render
  areas/viewports for atlas tiles, comparison samplers and dynamic depth bias. The lighting pass
  does 3x3 hardware PCF with a normal bias from `ShadowSettings` (UI-adjustable); spot/point slot
  indices are packed into `GpuLight::params.w`.
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
