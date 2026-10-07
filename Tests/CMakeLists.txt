find_package(GTest CONFIG REQUIRED)

add_executable(mukki_tests
    test_async_gltf_loader.cpp
    test_camera.cpp
    test_cloud_noise.cpp
    test_cluster_grid.cpp
    test_engine.cpp
    test_environment.cpp
    test_gi_atrous_schedule.cpp
    test_gi_probes.cpp
    test_gltf_loader.cpp
    test_hdr_loader.cpp
    test_job_system.cpp
    test_keybinds.cpp
    test_light_selection.cpp
    test_log.cpp
    test_path_utils.cpp
    test_resources.cpp
    test_scene_binary.cpp
    test_scene_json.cpp
    test_shader_reflection.cpp
    test_shadow_cascades.cpp
    test_vk_conversions.cpp
)
target_compile_definitions(mukki_tests PRIVATE MRE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(mukki_tests PRIVATE mukki::engine GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(mukki_tests)
