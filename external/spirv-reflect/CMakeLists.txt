cmake_minimum_required(VERSION 3.25)
project(spirv-reflect)

set(SPIRV-REFLECT_SOURCES
    src/spirv_reflect.c
    include/spirv-reflect/spirv_reflect.h
)

add_library(${PROJECT_NAME} STATIC ${SPIRV-REFLECT_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC include)
