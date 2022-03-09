#!/usr/bin/env python
import cv2

import numpy as np
import open3d as o3d
import rosbag
import rospkg
import rospy

import transforms3d as tf3d
import yaml
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from mongodb_store.message_store import MessageStoreProxy
from sensor_msgs.msg import CameraInfo
from obj_det_ppf_matching_msgs.msg import Table, IdAction
from tf_bag import BagTfTransformer
import os
import actionlib



class ReadRosbag:
    def __init__(self):
        self.server = actionlib.SimpleActionServer('read_rosbag', IdAction, self.execute, False)
        self.server.start()

    def execute(self, goal):
        print('Rosbag extraction node')
        rospack = rospkg.RosPack()
        path = rospack.get_path('table_extractor')
        with open(path+'/config.yaml', 'rb') as f:
            conf = yaml.load(f.read())    # load the config file
        rosbag_conf = conf['rosbag']
        cam_conf = conf['camera_info']
        ci = CameraInfo()
        ci.D, ci.K, ci.R, ci.P = cam_conf['D'], cam_conf['K'], cam_conf['R'], cam_conf['P']
        ci.height, ci.width = cam_conf['height'], cam_conf['width'] 
        ci.header.frame_id = cam_conf['frame_id']
        ci.distortion_model = cam_conf['distortion_model']
        cam = PinholeCameraModel()
        cam.fromCameraInfo(ci)

        storage_folder = os.path.join(rosbag_conf['data_folder'], 'read_rosbag' ,'plane_'+str(goal.id))
        if not os.path.exists(storage_folder):
            os.makedirs(storage_folder)
        depth_folder = os.path.join(storage_folder,'depth')
        if not os.path.exists(depth_folder):
            os.makedirs(depth_folder)
        rgb_folder = os.path.join(storage_folder, 'rgb')
        if not os.path.exists(rgb_folder):
            os.makedirs(rgb_folder)
        plane_folder = os.path.join(storage_folder, 'planes')
        if not os.path.exists(plane_folder):
            os.makedirs(plane_folder)

        depth_topic = rosbag_conf['depth_topic']
        rgb_topic = rosbag_conf['rgb_topic']
        tf_topic = rosbag_conf['tf_topic']

        # Check topics
        #TODO 
        bag_file = os.path.join(rosbag_conf['data_folder'],'stare_at_tables' ,str(goal.id)+'.bag')
        bag = rosbag.Bag(bag_file)
        topics = bag.get_type_and_topic_info()[1].keys()
        rgb_topic = self.check_topics(rgb_topic, topics)
        depth_topic = self.check_topics(depth_topic, topics)
        tf_topic = self.check_topics(tf_topic, topics)
        if rgb_topic is None or depth_topic is None or tf_topic is None:
            self.server.set_aborted()
            return

        table_txt_file = open(os.path.join(storage_folder, "table.txt"), "w")

        bbx_points = None
        msg_store = MessageStoreProxy()
        nr_of_tables = len(msg_store.query(Table._type))
        print("Extracted {} table(s) from database".format(nr_of_tables))
        i = 0
        hull_points = []
        for msg, meta in msg_store.query(Table._type):
            table_txt_file.write(str(msg)+"\n")
            points = []
            for point in msg.points:
                print(point)
                points.append([point.point.x, point.point.y, point.point.z, 1.0])
            points = np.array(points)
            hull_points.append(points)
        points = np.array(bbx_points)
        table_txt_file.close()
        print('Points received')

        step_size = rosbag_conf['step_size']
        bridge = CvBridge()
        depth_count = 0
        rgb_count = 0
        tf_count = 0
        log = []

        log_txt_file = open(os.path.join(storage_folder, "log.txt"), "w")

        depth_txt_files = []
        rgb_txt_files = []
        tf_txt_files = []
        table_folders = []
        #for i in range(nr_of_tables):
        table_folder = plane_folder
        if not os.path.exists(table_folder):
            os.makedirs(table_folder)
        table_folders.append(plane_folder)
        depth_txt_files.append([])
        rgb_txt_files.append([])
        tf_txt_files.append([])

        table_seen = False
        table_seen_prev = False
        #table_seen_list = np.zeros((rosbag_conf['series_size']), dtype=bool)
        tf_cam_map = np.zeros(7, dtype=float) #x y z qx qy qz qw
        print('Fill Transformer from bag')
        bag_transformer = BagTfTransformer(bag)
        print('Check viewpoints')
        for topic, msg, t in bag.read_messages(topics=[tf_topic ,depth_topic, rgb_topic]):
            if topic == tf_topic:
                if tf_count % step_size == 0:
                    table_seen = np.zeros(nr_of_tables, dtype=bool)
                    translation, q = bag_transformer.lookupTransform(cam.tf_frame, 'map', t) #q = x,y,z,w
                    rot = tf3d.quaternions.quat2mat([q[3], q[0], q[1], q[2]]) #q = w,x,y,z
                    transmat = np.eye(4)
                    transmat[:3, :3] = rot
                    transmat[:3, 3] = translation

                    for i in range(len(hull_points)):
                        points_cf = np.matmul(transmat, hull_points[i].T).T #points transformed to camera frame
                        for point in points_cf:
                            if np.linalg.norm(point)>2.5:
                                continue
                            u, v = cam.project3dToPixel(point[:3])
                            if 0 <= u <= cam.width and 0 <= v <= cam.height:
                                table_seen[i] = True
                    #for j in range(nr_of_tables):
                    if (not table_seen_prev): #and all(table_seen_list[:]):
                        log.append('Table {}, start at {}\n'.format(goal.id, t.to_sec()))
                        log_txt_file.write('Table {}, start at {}. Transform:\n'.format(goal.id, t.to_sec()))
                        log_txt_file.write("{:.9f}".format(t.to_sec()) + " - " + str(transmat) + "\n")

                        depth_occ_filename = os.path.join(table_folder, "depth.txt")
                        depth_txt_file = open(depth_occ_filename, "w")
                        rgb_occ_filename = os.path.join(table_folder, "rgb.txt")
                        rgb_txt_file = open(rgb_occ_filename, "w")
                        tf_occ_filename = os.path.join(table_folder, "tf.txt")
                        tf_txt_file = open(tf_occ_filename, "w")
                    elif table_seen_prev and not any(table_seen):#any(table_seen_list[:]):
                        log.append('Table {}, stop at {}\n'.format(j, t.to_sec()))
                    #if #all(table_seen_list[:]):
                    #    table_seen_prev = True
                    #elif not any(table_seen_list[:]):
                    #    table_seen_prev = False
                    if table_seen_prev:
                        translation, q = bag_transformer.lookupTransform('map', cam.tf_frame, t)  # q = x,y,z,w
                        tf_cam_map = np.array(translation + q)
                        tf_txt_file.write("{:.9f} {} {} {} {} {} {} {}\n".format(t.to_sec(), translation[0], translation[1], translation[2], q[0], q[1], q[2], q[3]))
                    
                    #print(table_seen_list)
                    #table_seen_list = np.roll(table_seen_list, -1, 0)
                    #table_seen_list[-1] = table_seen
                    table_seen_prev = any(table_seen)
            if topic==depth_topic:
                if depth_count % step_size == 0:
                    cv_img = bridge.imgmsg_to_cv2(msg, "32FC1")
                    
                    cv_imgOrig = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    #get minMax3D from hullPoints, call passThrough filter, set depth to very high value if outside
                    #for i in range(nr_of_tables):
                    i = goal.id
                    if table_seen_prev:   
                        cv_img_filtered = cv_img.copy()
                        #project 2d depth points in 3D camera frame and then transform in map-frame
                        translation = tf_cam_map[0:3]
                        rot = tf_cam_map[3:]
                        rot_mat = tf3d.quaternions.quat2mat([rot[3], rot[0], rot[1], rot[2]])  # q = w,x,y,z
                        transmat = np.eye(4)
                        transmat[:3, :3] = rot_mat
                        transmat[:3, 3] = translation

                        # find min and max values of convex hull
                        margin = 0.75
                        x_max = max(map(lambda coords: coords[0], hull_points[i])) + margin
                        x_min = min(map(lambda coords: coords[0], hull_points[i])) - margin
                        y_max = max(map(lambda coords: coords[1], hull_points[i])) + margin
                        y_min = min(map(lambda coords: coords[1], hull_points[i])) - margin
                        #z_max = max(map(lambda coords: coords[2], hull_points[i])) + margin
                        #z_min = min(map(lambda coords: coords[2], hull_points[i])) - margin
                        #print("x_min: {}, x_max: {}, y_min: {}, y_max: {}".format(x_min, x_max, y_min, y_max))

                        width = cam.width
                        height = cam.height
                        nx = np.linspace(0, width - 1, width)
                        ny = np.linspace(0, height - 1, height)
                        u, v = np.meshgrid(nx, ny)
                        x = (u.flatten() - cam.cx()) / cam.fx()
                        y = (v.flatten() - cam.cy()) / cam.fy()

                        z = cv_img.flatten() / 1000;
                        x = np.multiply(x, z)
                        y = np.multiply(y, z)

                        w_arr=np.ones(x.shape)
                        img_cam  = np.vstack((x,y,z,w_arr)).T
                        
                        #img_cam = []
                        #for v in range(cam.height):
                        #    for u in range(cam.width):
                        #        d = cv_img[v,u] * 0.001
                        #        x_cam = d * (u - cam.cx()) / cam.fx()
                        #        y_cam = d * (v - cam.cy()) / cam.fy()
                        #        z_cam = d;

                        #       point = np.array([x_cam,y_cam,z_cam, 1.0])
                        #       img_cam.append(point)

                        #transformation in cam frame
                        img_in_map = np.matmul(transmat, np.array(img_cam).T).T #TODO, what is correct here?
                        
                        img_mask = (img_in_map[:,0] < x_min) | (img_in_map[:,0] > x_max) | (img_in_map[:,1] < y_min) | (img_in_map[:,1] > y_max) 
                        
                        temp = cv_imgOrig.copy().flatten()
                        temp[img_mask] = 0

                        temp = np.reshape(temp, (cam.height, cam.width))
                        depth_img_cnt = "depth%05i" % (depth_count / step_size)
                        depth_img_name = depth_img_cnt + "_{0}.png".format(i)
                        cv2.imwrite(os.path.join(depth_folder, depth_img_name), temp)
                        depth_txt_file.write(str(msg.header.stamp.secs) + "." + str(msg.header.stamp.nsecs).zfill(9) + " depth/" + depth_img_name + "\n")
                depth_count += 1

            if topic==rgb_topic:
                if rgb_count % step_size == 0:
                    rgb_img_name = "rgb%05i.png" % (rgb_count/step_size)
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    cv2.imwrite(os.path.join(rgb_folder, rgb_img_name), cv_img)
                    #for i in range(nr_of_tables):
                    i = goal.id
                    if table_seen_prev:
                        rgb_txt_file.write(str(msg.header.stamp.secs) + "." + str(msg.header.stamp.nsecs).zfill(9) + " rgb/" + rgb_img_name + "\n")
                rgb_count += 1


        for file_list in depth_txt_files:
            for file in file_list:
                file.close()
        for file_list in rgb_txt_files:
            for file in file_list:
                file.close()      
        log_txt_file.close()  
        print('Finished')

        self.server.set_succeeded()

    def check_topics(self, t, topics):
        '''
        check if topic t is in list of topics. Also check with t[1:], since the leading string could be missing. 
        return topic if it is in there. 
        exit program if not. 
        '''
        if t in topics:
            pass
        elif t[1:] in topics:
            print('remove leading slash')
            t = t[1:]
        else:
            print('Topic {} not found.'.format(t))
            return None
        return t

if __name__ == '__main__':
    rospy.init_node('read_rosbag')
    server = ReadRosbag()
    print('ReadRosbag Action Server is ready')
    rospy.spin()