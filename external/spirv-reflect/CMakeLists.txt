cmake_minimum_required(VERSION 3.25)
project(spirv-reflect)

add_library(${PROJECT_NAME} STATIC
    src/spirv_reflect.c
    include/spirv-reflect/spirv_reflect.h
)

target_include_directories(${PROJECT_NAME} PUBLIC include)
