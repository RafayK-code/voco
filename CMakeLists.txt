cmake_minimum_required(VERSION 3.25)
project(voco VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)

option(VOCO_ENABLE_GLSL "Enable GLSL shader compilation via shaderc" OFF)
option(VOCO_ENABLE_HLSL "Enable HLSL shader compilation via DXC" OFF)

add_subdirectory(external)

add_library(voco
    src/device.cpp
    src/buffer.cpp
    src/image.cpp
    src/pipeline.cpp
    src/command_list.cpp
    src/retirement_queue.cpp

    include/voco/voco.h
    include/voco/context.h
    include/voco/utils.h
    include/voco/types.h
    include/voco/buffer.h
    include/voco/image.h
    include/voco/pipeline.h
    include/voco/command_list.h
    include/voco/device.h

    src/retirement_queue.h
    src/resource_state.h
)
add_library(voco::voco ALIAS voco)

target_include_directories(voco
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(voco PUBLIC Vulkan::Vulkan)
target_link_libraries(voco PRIVATE vma)
target_link_libraries(voco PRIVATE spirv-reflect)

if(VOCO_ENABLE_GLSL)
    # Requires Vulkan SDK built with shaderc (default in official SDK builds)
    find_package(Vulkan REQUIRED COMPONENTS shaderc_combined)
    target_compile_definitions(voco PRIVATE VOCO_ENABLE_GLSL)
    target_link_libraries(voco PRIVATE Vulkan::shaderc_combined)
endif()

if(VOCO_ENABLE_HLSL)
    target_compile_definitions(voco PRIVATE VOCO_ENABLE_HLSL)
    # TODO: find_package(dxc) and link
endif()
