cmake_minimum_required(VERSION 3.0.2)
project(cpos_work_classification)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  audio_common_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs audio_common_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS test_miniaudio.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
