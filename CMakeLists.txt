cmake_minimum_required(VERSION 3.20)

project(TensAxon
    VERSION 0.2.0
    DESCRIPTION "Header-only, constexpr C++20 linear algebra and tensor library"
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Only build tests/examples by default when this is the top-level project.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(TENSAXON_IS_TOP_LEVEL ON)
else()
    set(TENSAXON_IS_TOP_LEVEL OFF)
endif()

option(TENSAXON_BUILD_TESTS    "Build TensAxon unit tests" ${TENSAXON_IS_TOP_LEVEL})
option(TENSAXON_BUILD_EXAMPLES "Build TensAxon examples"   ${TENSAXON_IS_TOP_LEVEL})

# ----- library target ---------------------------------------------------------
add_library(tensaxon INTERFACE)
add_library(TensAxon::tensaxon ALIAS tensaxon)

target_include_directories(tensaxon INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(tensaxon INTERFACE cxx_std_20)

# ----- tests & examples -------------------------------------------------------
if(TENSAXON_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TENSAXON_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ----- install / package ------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TENSAXON_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/TensAxon")

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS tensaxon EXPORT TensAxonTargets)

install(EXPORT TensAxonTargets
    FILE TensAxonTargets.cmake
    NAMESPACE TensAxon::
    DESTINATION ${TENSAXON_INSTALL_CMAKEDIR}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TensAxonConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT
)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/TensAxonConfig.cmake"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/TensAxonTargets.cmake\")\n"
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TensAxonConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TensAxonConfigVersion.cmake"
    DESTINATION ${TENSAXON_INSTALL_CMAKEDIR}
)
