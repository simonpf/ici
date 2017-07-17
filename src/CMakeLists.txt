cmake_minimum_required(VERSION 2.8)

project (bmci)
set(CMAKE_BUILD_TYPE Release)

# Find Python
find_package(PythonLibs REQUIRED)

# Build Extension Module
include_directories(${PYTHON_INCLUDE_DIR})
add_library(bmci SHARED bmci.cxx)
set_property(TARGET bmci PROPERTY CXX_STANDARD 14)
