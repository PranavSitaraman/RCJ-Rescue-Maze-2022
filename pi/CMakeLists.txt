cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)

project(maze-2022)

set(OpenCV_DIR /home/pi/opencv-4.5.5/lib/cmake/opencv4)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/pi/opencv-4.5.5/lib/cmake/opencv4)

find_package(OpenCV REQUIRED)

add_compile_options(-fsanitize=address -static-libasan)
add_link_options(-fsanitize=address -static-libasan)
add_compile_definitions(ARDUINO=100)
add_executable(maze-2022 main.cpp matrix.cpp matrix.hpp search.cpp search.hpp Serial.cpp Serial.hpp tile.cpp tile.hpp
        camera.cpp camera.hpp arduino-compat/Wire.h Wire.cpp SparkFun_AS726X_Arduino_Library/src/AS726X.cpp
        SparkFun_AS726X_Arduino_Library/src/AS726X.h arduino-compat/Arduino.h)
target_include_directories(maze-2022 PRIVATE ${OpenCV_INCLUDE_DIRS} SparkFun_AS726X_Arduino_Library/src arduino-compat)
target_link_libraries(maze-2022 PRIVATE ${OpenCV_LIBRARIES} stdc++fs)
