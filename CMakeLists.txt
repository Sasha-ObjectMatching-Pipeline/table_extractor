cmake_minimum_required(VERSION 2.8.3)
project(table_extractor)

find_package(catkin REQUIRED
genmsg
message_generation
message_runtime
std_msgs
geometry_msgs
rospy
roscpp
sensor_msgs
actionlib_msgs
obj_det_ppf_matching_msgs
)

catkin_package(CATKIN_DEPENDS 
    roscpp 
    rospy
    std_msgs
    message_generation
)

