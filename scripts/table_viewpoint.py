#!/usr/bin/python

import numpy as np
import rospy
import tf
import rospkg
import yaml
from scipy.spatial import ConvexHull
from skimage.measure import approximate_polygon

from geometry_msgs.msg import Point, PoseStamped, Quaternion
from visualization_msgs.msg import MarkerArray, Marker

from mongodb_store.message_store import MessageStoreProxy
from obj_det_ppf_matching_msgs.msg import Table 
import ros_numpy
import open3d as o3d

map_frame = 'map'

class TableViewpoint:
    def __init__(self):
        self.table_viewpoint_counter = 1
        self.previous_table_nr = 1

        self.marker_id = 0
        self.markers = MarkerArray()

        self.marker_pub = rospy.Publisher('table_viewpoint_markers', MarkerArray, queue_size=10, latch=True)

        self.msg_store = MessageStoreProxy()

        rospack = rospkg.RosPack()
        path = rospack.get_path('table_extractor')
        with open(path+'/config.yaml', 'rb') as f:
            conf = yaml.load(f.read())    # load the config file

        vp_conf = conf['table_viewpoint']
        self.contour_approx_param = vp_conf['contour_approx_param']
        self.edge_dist = vp_conf['viewpoint_edge_dist']
        self.dist_param = vp_conf['viewpoint_dist_param']
        rospack = rospkg.RosPack()

        ext_conf = conf['table_extractor']
        self.delta_x = ext_conf['map_deltax']
        self.delta_y = ext_conf['map_deltay']
        self.res = ext_conf['map_resolution']
        self.height = ext_conf['map_height']
        self.width = ext_conf['map_width']
        self.colors = np.array(ext_conf['colors'], dtype=np.float32)/255
        self.class_labels = ext_conf['class_labels']
        dwn_vox = ext_conf['downsample_vox_size']
        reconstruction_file = ext_conf['reconstruction_file']
        pcd = o3d.io.read_point_cloud(reconstruction_file)
        pcd = pcd.voxel_down_sample(voxel_size=2*dwn_vox)
        
        # remove floor
        floor_indices = []
        for cpoint, i in zip(np.asarray(pcd.colors), range(len(pcd.colors))):
            if all(np.isclose(cpoint, self.colors[1])):
                floor_indices.append(i)
        floor_pcd = pcd.select_down_sample(floor_indices, invert=False)
        self.floor_pcd_tree = o3d.geometry.KDTreeFlann(floor_pcd)
        pcd = pcd.select_down_sample(floor_indices, invert=True)
        self.pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        

    def calculate_viewposes(self, point_1, point_2, center=None):
        """
            takes two points and calculates the distance between them, and a viewpoint by using the normal vector
            the viewpoint is at a fixed distance to the center of the two points.
            Also adds a viewpoint for the first edge point. 

            :returns geometry_msgs/PoseStamped[]
        """

        center_of_line = (point_1 + point_2) / 2
        vector = point_2 - point_1
        length = np.linalg.norm(vector)
        number_pts = int(round(length/self.dist_param))
        vector_n = (vector[1], -vector[0]) / length
        yaw = 3.14 + np.arctan2(vector_n[1], np.dot(vector_n, (1, 0)))
        viewposes = []

        if center is not None:
            center_vector = point_1 - center[:2] 
            center_length = np.linalg.norm(center_vector)
            center_vector = center_vector/np.linalg.norm(center_vector)

            center_yaw = 3.14 + np.arctan2(center_vector[1], np.dot(center_vector, (1, 0)))
            p = center[:2] + center_vector * 1.5 * self.edge_dist
            viewpose = self.transform_to_pose_st((p[0],p[1]), center_yaw)
            if self.is_viewpose_free(viewpose) and self.is_viewpose_on_map(viewpose):
                self.draw_marker_rviz_posest(viewpose)
                viewposes.append(viewpose)
            
        for i in range(number_pts):
            p = point_1 + (i + 1) * vector / (number_pts + 1) + vector_n * self.edge_dist
            x = p[0]
            y = p[1]
            viewpose = self.transform_to_pose_st((x,y), yaw)
            if self.is_viewpose_free(viewpose) and self.is_viewpose_on_map(viewpose):
                self.draw_marker_rviz_posest(viewpose)
                viewposes.append(viewpose)
        return viewposes

    def draw_marker_rviz_posest(self, pose_st):
        """
        Draw a yellow arrow marker at the given pose_st
        """

        marker = Marker()
        marker.header.frame_id = map_frame
        marker.header.stamp = rospy.Time.now()

        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = pose_st.pose
        marker.scale.x = 0.2
        marker.scale.y = 0.1
        marker.scale.z = 0.1


        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # marker.lifetime.secs = 0.5
        marker.id = self.marker_id
        self.marker_id += 1

        ## print ('draw marker at x= {} y= {}'.format(pose_st.pose.position.x, pose_st.pose.position.y))
        self.markers.markers.append(marker)
        self.marker_pub.publish(self.markers)

    # get viewpoints from table messages in database
    def search_for_viewpoint(self):
        img = np.zeros([self.height, self.width, 1], dtype=np.int8)

        for msg, meta in self.msg_store.query(Table._type):
            points = []
            #print('#{}: {}'.format(msg.id, msg.category))
            # convert 3D hull points to numpy and format them
            for point in msg.points:
                point = ros_numpy.numpify(point.point)
                points.append(point)
            points = np.asarray(points)
            points = np.delete(points, 2, 1)
            center = ros_numpy.numpify(msg.center.point)
            
            # get 2D hull of those points and approximate the contour
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]    
            points = approximate_polygon(points[hull.vertices], self.contour_approx_param)
            # iterate through all the edges of the contour and calculate the viewpoints from those 
            viewposes = []
            for j in range(len(points) - 1):    
                # calculate lenght of the table edge, position and orientation of the viewpoint 
                viewposes_line = self.calculate_viewposes(points[j], points[j + 1], center)
                viewposes.extend(viewposes_line)
                # last edge is between last and first point
                if j == (len(points) - 2):
                    viewposes_line = self.calculate_viewposes(points[j + 1], points[0], center)
                    viewposes.extend(viewposes_line)
            msg.viewposes = viewposes
            #print(viewposes)
            self.msg_store.update_id(meta['_id'], msg)


    def transform_to_pose_st(self, pt, yaw):
        """
        transform pt (x and y coordinates in pixels) and yaw to a PoseStamped message
        """
        pose_st = PoseStamped()
        pose_st.header.frame_id = map_frame
        pose_st.header.stamp = rospy.Time.now()
        #pose_x, pose_y = self.pixel_to_pose(pt[0], pt[1])
        pose_st.pose.position.x = pt[0]
        pose_st.pose.position.y = pt[1]
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        pose_st.pose.orientation.x = q[0]
        pose_st.pose.orientation.y = q[1]
        pose_st.pose.orientation.z = q[2]
        pose_st.pose.orientation.w = q[3]
        return pose_st

    def is_viewpose_free(self, viewpose):
        """
        receives a viewpose (PoseStamped) and checks if the space around it is free in the reconstruction
        """
        for height in np.arange(0, 2, 0.15):
            point =[viewpose.pose.position.x, viewpose.pose.position.y, viewpose.pose.position.z+height]
            [k, idx, _] = self.pcd_tree.search_radius_vector_3d(point, 0.34)
            if k>0:
                return False
        
        return True

    def is_viewpose_on_map(self, viewpose):
        """
        receives a viewpose (PoseStamped) and checks if the space around it contains floor points from the reconstruction. 
        We discard poses that are not near the floor, since the robot didn't look at these areas in the reconstruction step.
        """
        point =[viewpose.pose.position.x, viewpose.pose.position.y, viewpose.pose.position.z]
        [k, idx, _] = self.floor_pcd_tree.search_radius_vector_3d(point, 0.1)
        if k>0:
            return True
        else:
            return False

if __name__ == '__main__':
    rospy.init_node('table_viewpoint')
    try:
        viewpoint = TableViewpoint()
        viewpoint.search_for_viewpoint()
        print('Finished table_viewpoint script')
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

