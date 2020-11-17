cmake_minimum_required(VERSION 3.11)

project(NaturalVirtualInteraction VERSION 1.0 LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#Find Packages
find_package(Ceres REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV)

set(HTTPLIB_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/extern/httplib/include")
set(VPX_LIB "${PROJECT_SOURCE_DIR}/extern/libvpx/lib/x64/libvpx.lib")
set(VPX_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/extern/libvpx/include")

include(FetchContent)

FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_TAG v3.7.3)
FetchContent_GetProperties(json)

if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )

add_subdirectory(src)