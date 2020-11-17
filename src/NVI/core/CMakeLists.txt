file(GLOB SOURCES "*")
file(
  GLOB
  HEADER_LIST
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/NVI/core/*"
)

add_library(Core STATIC ${SOURCES} ${HEADER_LIST})

target_include_directories(
  Core
  PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
  Core 
  PUBLIC
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  Eigen3::Eigen
  PRIVATE
  Utilities
)