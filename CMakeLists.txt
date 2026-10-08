cmake_minimum_required(VERSION 3.25)

project(MukkiRebornEngine
    VERSION 0.1.0
    DESCRIPTION "Vendor-agnostic game engine with a Vulkan RHI and a deferred renderer"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8 /Zc:preprocessor /FS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(MRE_BUILD_SANDBOX "Build the sandbox application" ON)
option(MRE_BUILD_TESTS "Build the unit tests" ON)
option(MRE_ENABLE_TRACY "Enable Tracy profiler instrumentation" OFF)

find_package(Vulkan REQUIRED)
find_package(box3d CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(unofficial-spirv-reflect CONFIG REQUIRED)
find_package(Threads REQUIRED)

if(MRE_ENABLE_TRACY)
    find_package(Tracy CONFIG REQUIRED)
endif()

find_path(MRE_TINYGLTF_INCLUDE_DIR NAMES tiny_gltf.h REQUIRED)
find_path(MRE_STB_INCLUDE_DIR NAMES stb_image.h REQUIRED)

find_program(MRE_SLANGC_EXECUTABLE
    NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    DOC "Slang compiler used to build .slang shaders at build time"
)

# Compiles the given .slang files (relative to the calling source dir) to SPIR-V
# and attaches the resulting custom target to `target`.
function(mre_add_slang_shaders target)
    if(NOT MRE_SLANGC_EXECUTABLE)
        message(WARNING "slangc was not found, shader compilation is disabled for '${target}'")
        return()
    endif()

    set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    set(spirv_outputs "")

    # Slang include dependencies are not tracked by the compiler invocation, so every shader
    # depends on all shaders in the directory. Without this, editing a shared include such as
    # gi_structs.slang leaves other shaders compiled against a stale struct layout, which reads
    # garbage indices and faults the GPU.
    if(ARGN)
        list(GET ARGN 0 mre_first_shader)
        get_filename_component(mre_shader_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/${mre_first_shader}" DIRECTORY)
        file(GLOB shader_sources "${mre_shader_directory}/*.slang")
    endif()

    foreach(shader IN LISTS ARGN)
        get_filename_component(shader_file_name "${shader}" NAME)
        string(REGEX REPLACE "\\.slang$" "" shader_name "${shader_file_name}")
        set(spirv "${output_dir}/${shader_name}.spv")
        add_custom_command(
            OUTPUT "${spirv}"
            COMMAND "${MRE_SLANGC_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
                -target spirv
                -matrix-layout-column-major
                -fvk-use-entrypoint-name
                -o "${spirv}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${shader}" ${shader_sources}
            COMMENT "Compiling Slang shader ${shader}"
            VERBATIM
        )
        list(APPEND spirv_outputs "${spirv}")
    endforeach()

    add_custom_target(${target}_shaders DEPENDS ${spirv_outputs})
    add_dependencies(${target} ${target}_shaders)
    set_target_properties(${target} PROPERTIES MRE_SHADER_OUTPUT_DIRECTORY "${output_dir}")

    add_custom_command(
        TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${output_dir}" "$<TARGET_FILE_DIR:${target}>/shaders"
        COMMENT "Copying compiled shaders next to ${target}"
        VERBATIM
    )
endfunction()

add_library(mukki_engine STATIC
    Application/Application.cpp
    Application/Camera.cpp
    Application/Input.cpp
    Application/Keybinds.cpp
    Application/PlaneController.cpp
    Application/UI/UiStyles.cpp
    Application/Window.cpp
    Assets/AsyncGltfLoader.cpp
    Assets/GltfLoader.cpp
    Assets/HdrLoader.cpp
    Core/Engine.cpp
    Core/JobSystem.cpp
    Core/Time.cpp
    Physics/PhysicsWorld.cpp
    Renderer/BufferManager.cpp
    Renderer/Clouds/CloudNoise.cpp
    Renderer/Debug/DebugLines.cpp
    Renderer/Deferred/DeferredRenderer.cpp
    Renderer/Deferred/GBuffer.cpp
    Renderer/GlobalIllumination/GiAtrousSchedule.cpp
Renderer/Lighting/ClusterGrid.cpp
Renderer/Lighting/Environment.cpp
Renderer/Lighting/Light.cpp
Renderer/Lighting/ShadowCascade.cpp
    Renderer/RHI/Resources.cpp
    Scene/SceneBinary.cpp
    Scene/SceneDescription.cpp
    Scene/SceneJson.cpp
    Renderer/Vulkan/Core/FrameSync.cpp
    Renderer/Vulkan/Core/VkCommandList.cpp
    Renderer/Vulkan/Core/VkConversions.cpp
    Renderer/Vulkan/Core/VkDevice.cpp
    Renderer/Vulkan/Core/VkInstance.cpp
    Renderer/Vulkan/Core/VkPipeline.cpp
    Renderer/Vulkan/Core/VkRayTracingDispatch.cpp
    Renderer/Vulkan/Core/VkShader.cpp
    Renderer/Vulkan/Core/VkSwapchain.cpp
    Renderer/Vulkan/Resources/AccelerationStructure.cpp
    Renderer/Vulkan/Resources/Buffer.cpp
    Renderer/Vulkan/Resources/DescriptorSet.cpp
    Renderer/Vulkan/Resources/ResourceAllocator.cpp
    Renderer/Vulkan/Resources/Sampler.cpp
    Renderer/Vulkan/Resources/Texture.cpp
    Renderer/Vulkan/UiLayer.cpp
    Renderer/Vulkan/VmaImplementation.cpp
    Utilities/Assert.cpp
    Utilities/Log.cpp
    Utilities/PathUtils.cpp
)
add_library(mukki::engine ALIAS mukki_engine)

target_include_directories(mukki_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(mukki_engine SYSTEM PRIVATE
    "${MRE_TINYGLTF_INCLUDE_DIR}"
    "${MRE_STB_INCLUDE_DIR}"
)
target_compile_definitions(mukki_engine PUBLIC
    MRE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    MRE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    MRE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)
target_compile_definitions(mukki_engine PRIVATE
    VMA_STATIC_VULKAN_FUNCTIONS=1
    VMA_DYNAMIC_VULKAN_FUNCTIONS=0
)
target_link_libraries(mukki_engine PRIVATE nlohmann_json::nlohmann_json unofficial::spirv-reflect box3d::box3d)
target_link_libraries(mukki_engine
    PUBLIC
        Vulkan::Vulkan
        GPUOpen::VulkanMemoryAllocator
        glfw
        glm::glm
        imgui::imgui
        Threads::Threads
)

if(MRE_ENABLE_TRACY)
    target_compile_definitions(mukki_engine PUBLIC MRE_ENABLE_TRACY)
    target_link_libraries(mukki_engine PUBLIC Tracy::TracyClient)
endif()

if(MRE_BUILD_SANDBOX)
    add_executable(mukki_sandbox
        Sandbox/main.cpp
    )
    target_link_libraries(mukki_sandbox PRIVATE mukki::engine)

    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/glTF-Sample-Models/2.0")
        message(STATUS "glTF sample models submodule not found; the demo scene models will not load")
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Sandbox/Scenes")
        target_compile_definitions(mukki_sandbox PRIVATE
            MRE_SCENES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Sandbox/Scenes")
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/AssetLibrary")
        target_compile_definitions(mukki_sandbox PRIVATE
            MRE_ASSET_LIBRARY_DIR="${CMAKE_CURRENT_SOURCE_DIR}/AssetLibrary")
    endif()

    mre_add_slang_shaders(mukki_sandbox
        Renderer/Shaders/gbuffer.vert.slang
        Renderer/Shaders/gbuffer.frag.slang
        Renderer/Shaders/transparent.frag.slang
        Renderer/Shaders/lighting.vert.slang
        Renderer/Shaders/lighting.frag.slang
        Renderer/Shaders/cluster_culling.comp.slang
        Renderer/Shaders/shadow.vert.slang
        Renderer/Shaders/shadow.frag.slang
        Renderer/Shaders/gi_trace.comp.slang
        Renderer/Shaders/gi_temporal.comp.slang
        Renderer/Shaders/gi_atrous.comp.slang
        Renderer/Shaders/gi_upsample.comp.slang
        Renderer/Shaders/gi_probes.comp.slang
        Renderer/Shaders/rt_reflections.comp.slang
        Renderer/Shaders/reflection_temporal.comp.slang
        Renderer/Shaders/reflection_upsample.comp.slang
        Renderer/Shaders/rt_shadows.comp.slang
        Renderer/Shaders/clouds.comp.slang
        Renderer/Shaders/cloud_reproject.comp.slang
        Renderer/Shaders/cloud_godrays.comp.slang
        Renderer/Shaders/cloud_upsample.comp.slang
        Renderer/Shaders/debug_line.vert.slang
        Renderer/Shaders/debug_line.frag.slang
    )
endif()

add_executable(mukki_scene_compiler
    Tools/SceneCompiler/main.cpp
)
target_link_libraries(mukki_scene_compiler PRIVATE mukki::engine)

if(MRE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif()
