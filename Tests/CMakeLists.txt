find_package(GTest CONFIG REQUIRED)

add_executable(mukki_tests
    test_camera.cpp
    test_engine.cpp
    test_keybinds.cpp
    test_log.cpp
    test_resources.cpp
    test_vk_conversions.cpp
)
target_link_libraries(mukki_tests PRIVATE mukki::engine GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(mukki_tests)
