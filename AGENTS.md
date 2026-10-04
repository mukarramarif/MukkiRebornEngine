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
- Materials: glTF `alphaMode = MASK` is supported. The flag and `alphaCutoff` are packed into
  `ObjectPushConstants.params.z/.w` (set by `DeferredRenderer` from `GltfMaterial`), the G-buffer
  discards `baseColor.a * baseColorFactor.a < cutoff`, and the shadow fragment stage alpha-tests
  through the bindless base color texture (the shadow pipeline binds the bindless set as set 1 and
  uses explicit vertex attribute offsets for position/uv). Ray traced hit shading also honours
  coverage: MASK hits below the cutoff are skipped and BLEND hits are stochastically accepted
  (`GpuGeometryData.params.z` = alpha mode 0/1/2, `.w` = cutoff). Emissive is supported
  (`emissiveFactor` scaled by `KHR_materials_emissive_strength` + `emissiveTexture`): the G-buffer
  stores the object index + 1 in `position.w` and the object UV in `motion.zw`, the lighting pass
  fetches `GpuObjectData` for the factor and samples the emissive texture, the transparent pass
  reads it through `gObjects`, and ray traced hit shading adds `GpuGeometryData.emissive`.
  `KHR_materials_specular` is supported: factor/color and the extension textures travel through the
  extended `GpuObjectData` SSBO (read by the G-buffer fragment), and the dielectric F0 is baked into
  the fifth G-buffer target (`GBuffer::GetSpecularF0`, 0.04 default); lighting and reflections use
  `f0 = lerp(specularF0, albedo, metalness)`.
  Transparency: `alphaMode = BLEND` surfaces skip the G-buffer and are drawn by a sorted
  back-to-front forward pass after lighting (`BeginSwapchainRendering` depth overload, depth test
  against the G-buffer depth, no depth write); shadow maps still include them. `KHR_materials_ior`,
  `transmission`, `volume` and `dispersion` are packed into G-buffer targets 5/6
  (`GBuffer::GetTransmission/GetAttenuation`); `rt_reflections` splits the microfacet lobe into
  reflection/refraction by fresnel and `ShadeHit` (`gi_shading.slang`) follows up to `params7.x`
  dielectric bounces with Beer-Lambert absorption and optional 3-channel dispersion. The lighting
  pass scales the surface terms by `(1 - transmission)` and composites the traced dielectric signal,
  so glass is reflected/refracted consistently in reflections, GI and probes.
- Debug gizmos: `Renderer/Debug/DebugLines.h` builds CPU line lists (`DebugLines`) for light and
  probe markers; `DeferredRenderer::RecordDebugLines` draws them at the end of the swapchain
  lighting scope with a `PrimitiveTopology::LineList` pipeline (push constant viewProjection, no
  depth). World space GI probes are authored as `SceneDescription::probes` (JSON + scene binary v4);
  the sandbox feeds `GiProbe` markers through `SetGiProbes` and toggles them with `SetGizmoView`.
  The probe UI can `Fill scene`/`Maximize probes`, distributing a Halton set of up to
  `kMaxGiProbes` probes over `DeferredRenderer::GetSceneBounds`.
  Probe markers are generated from the uploaded GI probe data and show the SH direction arrow.
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
- Ray traced sun shadows: optional (`ShadowSettings.rtSunShadows`, requires ray tracing).
  `Renderer/Shaders/rt_shadows.comp.slang` reconstructs the surface from the G-buffer position/normal
  and traces `rtShadowSamples` jittered rays toward the sun into a full-res `R16Sfloat` mask; the
  lighting pass samples binding 13 and uses it instead of the cascades while `ShadowUniform.rtParams.x`
  is set; the GI trace also samples it (set 0 binding 9, `GiUniform.params4.w`) for hit points that
  are visible on screen, falling back to jittered shadow rays otherwise. The pass binds the TLAS +
  G-buffer textures per frame after the G-buffer pass; the mask is bound as a fallback white texture
  when inactive. Spot/point shadows remain cascaded maps.
- Ray traced reflections: optional (`ReflectionSettings`, requires ray tracing).
  `rt_reflections.comp.slang` traces `samples` Heitz GGX VNDF rays per pixel at
  `ReflectionSettings::resolutionScale` (default 1.0, i.e. full resolution; the trace/temporal/
  a-trous/upsample sizes come from `GiUniform.reflectionSize`), shades hits through the shared
  `gi_shading.slang` (screen-space feedback disabled by binding the 1x1 black texture) and weights
  by `Schlick fresnel * Smith G2/G1`. Surfaces fade out toward `maxRoughness` while the GI specular
  signal fades in over the same band, so the specular integral is not double counted (when
  reflections are inactive `params6.w = 0` makes the GI trace cover smooth specular).
  `reflection_temporal.comp.slang` accumulates with moments and firefly rejection, the a-trous pass
  reuses the GI atrous pipeline with push-constant signal 2 (which switches the tap mapping to
  `reflectionSize`), and `reflection_upsample.comp.slang` bilaterally upsamples to full res
  (roughness-aware). The lighting pass adds binding 15 scaled by `SceneUniform.reflectionParams.x`.
  `GiUniform.params6` carries the reflection parameters and `params5.w` the history reset flag.
- Image based lighting: `Renderer/Lighting/Environment.h/.cpp` defines a parametric gradient sky
  (`EnvironmentSettings`); `ComputeEnvironmentSh` numerically projects it onto SH L2 with the
  Lambert band factors folded in (diffuse = `albedo * EvaluateSkySh`), and `BuildEnvironmentBrdfLut`
  generates the split-sum LUT (bound at lighting set 0 binding 21/22). `sky.slang` holds the shared
  evaluation (`SkyRadiance`, `EvaluateSkySh`, `PrefilteredSky`, `SkyBackground`) used by
  `lighting.frag` (sky background, SH ambient, prefiltered specular) and by the GI/reflection/probe
  sky misses through `GiSkyRadiance`/`GiIrradianceOverPi`. `SceneUniform`/`GiUniform` carry
  `skySh[9]`, `skyParams`, the zenith/horizon/ground colors and `skyMap` (bindless index, sampler,
  has-map). Disabling the environment falls back to the flat `ambientColor`/`skyColor`.
  Radiance `.hdr` equirectangular maps (`Assets/HdrLoader.h/.cpp`: flat + RLE RGBE decoding, RGBE
  to half-float packing, `PackHdrToRgba16`) replace the gradient when
  `EnvironmentSettings::environmentMap` is set (UI `HDR map (.hdr)` / `--env <path>`): the map is
  uploaded as an `R16G16B16A16Sfloat` texture in the bindless set, its SH is projected by
  `ComputeEnvironmentSh(equirectRgb, ...)` and the shaders sample it through
  `EnvironmentRadiance`/`EnvironmentPrefiltered`/`EnvironmentBackground` (equirect convention
  `u = atan2(z,x)/2pi + 0.5`, `v = acos(y)/pi`, matching the CPU projection).
- GI probes: world space `SceneDescription::probes` become irradiance probes. When probes are
  active (`GiSettings.useProbes`, requires ray tracing) `gi_probes.comp.slang` traces `probeRays`
  uniform-sphere rays per probe through the shared hit shading in `gi_shading.slang`, accumulates
  premultiplied SH L1 and blends with `probeHysteresis`; the lighting pass inverse-distance blends
  the probes (binding 14, count in `SceneUniform.ambientColor.w`) and the per-pixel trace + SVGF
  denoiser are skipped so indirect is not double counted (`IsPerPixelGiActive`). `gi_shading.slang`
  is an include, not a standalone shader; keep `GiUniform` in sync across all gi_*.slang files.
- Global illumination: optional hardware ray tracing (`VK_KHR_acceleration_structure`,
  `VK_KHR_ray_query`, `VK_KHR_deferred_host_operations`; `Device::IsRayTracingSupported()`). The RT
  entry points are loaded manually by `Renderer/Vulkan/Core/VkRayTracingDispatch` because the Vulkan
  loader import library does not export them. `Renderer/GlobalIllumination/` holds `GiTypes.h`
  (`GiSettings`/`GiUniform`, keep in sync with `Renderer/Shaders/gi_*.slang`) and the a-trous
  schedule (tested). Per frame: half-res trace (`traceSamples` diffuse+specular path pairs per
  pixel; sun visibility from the RT shadow mask when active, else `sunShadowSamples` jittered rays
  within `sunAngularRadiusDegrees`; previous-frame feedback) -> SVGF temporal accumulation with
  moments and firefly rejection -> a-trous ping-pong (variance-clamped taps) ->
  joint bilateral upsample into `indirect`, composited in `lighting.frag.slang` (`gIndirect`,
  gated by `SceneUniform.screenSize.w`). Storage textures live in `GENERAL` (transitioned once at
  creation); each a-trous iteration uses its own descriptor set because updating a descriptor set
  bound to a recording command buffer invalidates that command buffer. The bindless descriptor set
  declares `FragmentCompute` stages; geometry buffers need storage usage for the ray hit shading;
  RT pipelines use two descriptor set layouts (GI set 0 + bindless set 1). Slang gotchas: no
  lambdas, `RayQuery.CommittedInstanceID()`.
  Punctual lights feed the hit shading through a dedicated per-frame `GpuLight` buffer bound at set
  0 binding 13 in `gi_trace`/`gi_probes`/`rt_reflections` (`gi_structs.slang` `kMaxGiLights`,
  `SelectGiLights` in `Light.cpp`, `GiUniform.params8`: count/shadows/bias). `ShadeOpaqueSurface`
  shades the sun and every GI light with the shared `brdf.slang` Cook-Torrance evaluation plus one
  ray traced shadow ray per light (`TraceLightVisibility`); spot cones and the windowed
  inverse-square falloff match `lighting.frag`.
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
