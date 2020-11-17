file(GLOB SOURCES "*")
file(
  GLOB
  HEADER_LIST
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/NVI/calibration/*"
)

add_executable(CameraCalibration ${SOURCES} ${HEADER_LIST})

target_include_directories(
  CameraCalibration
  PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
  CameraCalibration 
  PRIVATE
  ${OpenCV_LIBS}
)
