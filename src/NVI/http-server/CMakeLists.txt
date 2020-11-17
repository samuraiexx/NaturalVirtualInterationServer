file(GLOB SOURCES "*")
file(
  GLOB
  HEADER_LIST
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/NVI/http-server/*"
)

add_executable(HttpServer ${SOURCES} ${HEADER_LIST})

target_include_directories(
  HttpServer
  PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${HTTPLIB_INCLUDE_DIR}
)

target_link_libraries(
  HttpServer 
  PRIVATE 
  Core
  Utilities
  Decoder
  nlohmann_json::nlohmann_json
  ${OpenCV_LIBS}
)

# Copy Caffe Models files to Executable Dir
add_custom_command(TARGET HttpServer POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/models/ $<TARGET_FILE_DIR:HttpServer>/models)

# Copy Caffe Models files to Executable Dir
add_custom_command(TARGET HttpServer POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_SOURCE_DIR}/IKWeights.in $<TARGET_FILE_DIR:HttpServer>/IKWeights.in)