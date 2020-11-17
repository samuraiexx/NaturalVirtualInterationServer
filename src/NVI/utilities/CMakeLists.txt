file(GLOB SOURCES "*")
file(
  GLOB
  HEADER_LIST
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/NVI/utilities/*"
)

add_library(Utilities STATIC ${SOURCES} ${HEADER_LIST})

target_include_directories(
  Utilities
  PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
  Utilities
  PRIVATE
  ${OpenCV_LIBS}
)