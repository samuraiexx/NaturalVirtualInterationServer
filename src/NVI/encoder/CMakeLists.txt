file(GLOB SOURCES "*")
file(
  GLOB
  HEADER_LIST
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/NVI/encoder/*"
)

add_library(Encoder SHARED ${SOURCES} ${HEADER_LIST})

target_include_directories(
  Encoder
  PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  PRIVATE 
  ${VPX_INCLUDE_DIR}
)

target_link_libraries(
  Encoder 
  PRIVATE
  ${VPX_LIB}
)
