cmake_minimum_required(VERSION 3.15)

# Library target
add_library(flexcontact_cpp 
    src/constrained_cg.cpp
    # ... other cpp files
)

target_include_directories(flexcontact_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(flexcontact_cpp PUBLIC cxx_std_17)

# Optimization flags
target_compile_options(flexcontact_cpp PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -ffast-math>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)

# OpenMP support (optional but recommended)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(flexcontact_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()

# Tests
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS flexcontact_cpp
    EXPORT flexcontact_cpp_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)