#!/usr/bin/python

import open3d as o3d
import numpy as np
import rospy
import rospkg
import yaml

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from obj_det_ppf_matching_msgs.msg import Table
from visualization_msgs.msg import Marker, MarkerArray

from nav_msgs.msg import OccupancyGrid
import ros_numpy
from geometry_msgs.msg import Point
from table_viewpoint import TableViewpoint


from mongodb_store.message_store import MessageStoreProxy
import open3d as o3d
from open3d_ros_helper import open3d_ros_helper as orh

rospy.init_node('mongo_test')
msg_store = MessageStoreProxy()

rospack = rospkg.RosPack()
path = rospack.get_path('table_extractor')
with open(path+'/config.yaml', 'rb') as f:
    conf = yaml.load(f.read())    # load the config file

ext_conf = conf['table_extractor']
reconstruction_file = ext_conf['reconstruction_file']
colors = np.array(ext_conf['colors'], dtype=np.float32)/255
reconst = o3d.io.read_point_cloud(reconstruction_file)
reconst_ros = orh.o3dpc_to_rospc(reconst, 'map')
reconst_pub = rospy.Publisher('/table_extractor/reconstruction', PointCloud2, queue_size=10, latch=True)

reconst_pub.publish(reconst_ros)

names_pub = rospy.Publisher('/table_extractor/table_names', MarkerArray, queue_size=10, latch=True)
planes_pub = rospy.Publisher('/table_extractor/table_planes', MarkerArray, queue_size=10, latch=True)
names_list = []
planes_list = []
table_nr = 0

viewpoint = TableViewpoint()

for msg, meta in msg_store.query(Table._type):
    points = []

    name = Marker()
    name.header.frame_id = 'map'
    name.type = 9 #TEXT_VIEW_FACING=9
    name.id = table_nr
    name.pose.position = msg.center.point
    print(msg.id)
    print(msg.category)
    #print(msg.viewposes)
    for pose in msg.viewposes:
        viewpoint.draw_marker_rviz_posest(pose)

    name.pose.position.z = msg.center.point.z + 0.2
    name.text = '{}_{}'.format(msg.category, msg.id)
    name.color.a = 1.0
    name.color.r = 1.0
    name.color.g = 1.0
    name.color.b = 1.0
    name.scale.z = 0.15
    names_list.append(name)

    for point in msg.points:
        points.append([point.point.x, point.point.y, point.point.z])
    bbx = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))
    box_points = np.asarray(bbx.get_box_points())
    box_points_ros = []
    for i in range(8):
        box_points_ros.append(ros_numpy.msgify(Point, box_points[i,:]))
    bbx = Marker()
    bbx.header.frame_id = 'map'
    bbx.type = 4
    bbx.id = table_nr
    bbx.points = [x.point for x in msg.points]
    bbx.pose.orientation.w = 1.0
    bbx.color.a = 1.0
    bbx.color.r = colors[table_nr][0]
    bbx.color.g = colors[table_nr][1]
    bbx.color.b = colors[table_nr][2]
    bbx.scale.x = 0.1
    planes_list.append(bbx)

    table_nr += 1

names_pub.publish(names_list)
planes_pub.publish(planes_list)
rospy.spin()