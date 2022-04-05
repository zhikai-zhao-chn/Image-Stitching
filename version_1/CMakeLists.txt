cmake_minimum_required(VERSION 2.8)
project(PicStitching)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#add_executable(PicStitching_my.cpp PicStitching_my.cpp)
#target_link_libraries(PicStitching_my.cpp ${OpenCV_LIBS})

add_executable(orb_cv.cpp orb_cv.cpp)
target_link_libraries(orb_cv.cpp ${OpenCV_LIBS})