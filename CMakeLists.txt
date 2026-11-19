cmake_minimum_required(VERSION 3.20)

project(linalg 
    VERSION 0.1
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(linalg INTERFACE)
add_library(LinAlg::linalg ALIAS linalg)

target_include_directories(linalg INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(linalg INTERFACE cxx_std_20)

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

include(GNUInstallDirs)

set(LINALG_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    TARGETS linalg
    EXPORT LinAlgTargets
)

install(
    EXPORT LinAlgTargets
    FILE LinAlgTargets.cmake
    NAMESPACE LinAlg::
    DESTINATION ${LINALG_INSTALL_CMAKEDIR}
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT
)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
[=[
@PACKAGE_INIT@
include("${CMAKE_CURRENT_LIST_DIR}/LinAlgTargets.cmake")
]=]
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${LINALG_INSTALL_CMAKEDIR}
)