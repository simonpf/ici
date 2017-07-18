cmake_minimum_required(VERSION 2.8)

project (bmci)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

# Find Python
find_package(PythonLibs REQUIRED)
find_package(NumPy      REQUIRED)

# Build Extension Module
include_directories(${PYTHON_INCLUDE_DIR})
add_library(bmci SHARED bmci.cxx)
set_property(TARGET bmci PROPERTY CXX_STANDARD 14)
