add_executable(voco_tests
    main.cpp
    test_context.cpp
    test_buffer.cpp
)

target_link_libraries(voco_tests
    PRIVATE voco::voco
    PRIVATE doctest
)

target_include_directories(voco_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
