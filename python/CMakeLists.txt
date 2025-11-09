cmake_minimum_required(VERSION 3.15)

project(flexcontact LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP QUIET)

pybind11_add_module(
    _core
    flexcontact/_core.cpp
    ../cpp/src/constrained_cg.cpp
)
target_include_directories(
    _core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/include
)
target_link_libraries(_core PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _core DESTINATION flexcontact)