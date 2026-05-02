cmake_minimum_required(VERSION 3.25)
project(doctest)

set(DOCTEST_SOURCES
    src/doctest.cpp
    include/doctest/doctest.h
)

add_library(${PROJECT_NAME} STATIC ${DOCTEST_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC include)