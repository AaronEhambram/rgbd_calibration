cmake_minimum_required(VERSION 3.5)
project(rgbd_calibration)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#set(OpenCV_DIR "/home/ehambram/workspaces/libraries/opencv/opencv-4.x/install/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(get_calibration_images_node src/get_calibration_images.cpp)
ament_target_dependencies(get_calibration_images_node rclcpp std_msgs cv_bridge sensor_msgs message_filters)
target_link_libraries(get_calibration_images_node ${OpenCV_LIBS})

add_executable(calibrate_node src/calibrate.cpp)
ament_target_dependencies(calibrate_node rclcpp std_msgs cv_bridge sensor_msgs message_filters)
target_link_libraries(calibrate_node ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(rgb_depth_registration_node src/rgb_depth_registration.cpp)
ament_target_dependencies(rgb_depth_registration_node rclcpp std_msgs cv_bridge sensor_msgs message_filters pcl_conversions)
target_link_libraries(rgb_depth_registration_node ${OpenCV_LIBS} OpenMP::OpenMP_CXX ${Boost_SYSTEM_LIBRARY} ${PCL_LIBRARIES})


# install the include folder
# install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

install(TARGETS
  rgb_depth_registration_node
  get_calibration_images_node
  calibrate_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
