cmake_minimum_required(VERSION 3.5)
project(bytetrack-opencv-cpp)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 4 REQUIRED NO_DEFAULT_PATH PATHS /home/punzeonlung/cvDemo/opencv480-linux/lib/cmake/opencv4)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(
    ${OpenCV_INCLUDE_DIRS}/include
    ${CMAKE_SOURCE_DIR}/include
   
    )
add_executable(bytetrack-opencv-cpp src/BYTETracker.cpp
src/kalmanFilter.cpp
src/lapjv.cpp
src/STrack.cpp
src/utils.cpp
src/inference.cpp
src/main.cpp)
target_link_libraries(bytetrack-opencv-cpp ${OpenCV_LIBS} Eigen3::Eigen)
