cmake_minimum_required(VERSION 3.22)
project(EnergyContraption)

set(CMAKE_CXX_STANDARD 20)

add_executable(EnergyContraption main.cpp Utilities/RectangleTool.cpp Utilities/GeometryFeature.cpp Utilities/PCATool.cpp Utilities/ImageDebugUtility.cpp Utilities/CircleTool.cpp)

find_package(OpenCV REQUIRED)
target_include_directories(EnergyContraption PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(EnergyContraption PUBLIC  ${OpenCV_LIBS})

file(COPY SimulatedVideo.avi DESTINATION ${PROJECT_BINARY_DIR}/)
file(COPY RealWorldVideo.mp4 DESTINATION ${PROJECT_BINARY_DIR}/)
file(COPY RealWorldFocused.mp4 DESTINATION ${PROJECT_BINARY_DIR}/)
file(COPY RealWorldDynamic.mp4 DESTINATION ${PROJECT_BINARY_DIR}/)