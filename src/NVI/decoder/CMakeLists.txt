file(GLOB SOURCES "*")
file(
  GLOB
  HEADER_LIST
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/NVI/decoder/*"
)

add_library(Decoder STATIC ${SOURCES} ${HEADER_LIST})

target_include_directories(
  Decoder
  PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  PRIVATE 
  ${VPX_INCLUDE_DIR}
)

target_link_libraries(
  Decoder 
  PRIVATE
  ${VPX_LIB}
)
