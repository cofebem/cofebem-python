cmake_minimum_required(VERSION 3.15)
project(flexcontact VERSION 0.1.0 LANGUAGES CXX)

# Options
option(BUILD_TESTING "Build tests" ON)
option(BUILD_PYTHON "Build Python bindings" ON)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add subdirectories
add_subdirectory(cpp)

if(BUILD_PYTHON)
    add_subdirectory(python)
endif()