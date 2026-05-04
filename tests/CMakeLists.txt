add_executable(voco_tests
    main.cpp
    test_context.cpp
    test_buffer.cpp
    test_pipeline.cpp
    test_sync.cpp
)

target_link_libraries(voco_tests
    PRIVATE voco::voco
    PRIVATE doctest
)

target_include_directories(voco_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(voco_tests PRIVATE
    VOCO_TEST_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)
