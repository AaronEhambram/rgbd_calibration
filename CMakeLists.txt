cmake_minimum_required(VERSION 3.0.2)
project(rgbd_calibration)
SET(CMAKE_BUILD_TYPE Release)
set(OpenCV_DIR "/home/ehambram/workspaces/libraries/OpenCV/opencv-4.5.1/build")
find_package(OpenCV 4.5.1 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  visualization_msgs
  message_filters
)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(get_calibration_images_node src/get_calibration_images.cpp)
target_link_libraries(get_calibration_images_node ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(calibrate_node src/calibrate.cpp)
target_link_libraries(calibrate_node ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(rgb_depth_registration_node src/rgb_depth_registration.cpp)
target_link_libraries(rgb_depth_registration_node ${catkin_LIBRARIES} ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
