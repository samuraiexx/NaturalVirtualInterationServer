file(GLOB SOURCES "*")
file(
  GLOB
  HEADER_LIST
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/NVI/local-run/*"
)

add_executable(LocalRun ${SOURCES} ${HEADER_LIST})

target_include_directories(
  LocalRun
  PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
  LocalRun
  PRIVATE
  Core 
  Utilities
  ${OpenCV_LIBS}
)