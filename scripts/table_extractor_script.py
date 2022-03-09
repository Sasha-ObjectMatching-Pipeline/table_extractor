#!/usr/bin/python

import cv2 as cv
import geometry_msgs
import numpy as np
import open3d as o3d
import ros_numpy
import rospkg
import rospy
import sensor_msgs.point_cloud2 as pc2
import yaml
from mongodb_store.message_store import MessageStoreProxy
from nav_msgs.msg import OccupancyGrid
from obj_det_ppf_matching_msgs.msg import Table, Plane
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from region_growing_by_distance import RegionGrowingByDistanceOnly



class TableExtractor:
    def __init__(self):
        rospack = rospkg.RosPack()
        path = rospack.get_path('table_extractor')
        with open(path+'/config.yaml', 'rb') as f:
            conf = yaml.load(f.read())    # load the config file

        ext_conf = conf['table_extractor']
        self.colors = np.array(ext_conf['colors'], dtype=np.float32)/255
        self.normals_thresh = ext_conf['normals_thresh']
        self.class_labels = ext_conf['class_labels']
        self.class_names = ext_conf['class_names']
        self.reconstruction_file = ext_conf['reconstruction_file']
        self.dwn_vox = ext_conf['downsample_vox_size']
        self.nbpoints = ext_conf['remove_radius_outlier_nbpoints']
        self.radius = ext_conf['remove_radius_outlier_radius']
        self.eps = ext_conf['cluster_dbscan_eps']
        self.minpoints = ext_conf['cluster_dbscan_minpoints']
        #self.min_csp = ext_conf['min_cluster_size_param']
        self.height = ext_conf['map_height']
        self.width = ext_conf['map_width']
        self.delta_x = ext_conf['map_deltax']
        self.delta_y = ext_conf['map_deltay']
        self.res = ext_conf['map_resolution']
        
        self.min_cluster_size = ext_conf['min_cluster_size']
        self.normals_kNN = ext_conf['normals_kNN']
        self.normals_search_radius = ext_conf['normals_search_radius']

    def execute(self):
        msg_store = MessageStoreProxy()

        for msg, meta in msg_store.query(Table._type, {"_meta.inserted_by": "/table_extractor"}):
            msg_store.delete(str(meta.get('_id')))

        #read reconstruction file
        pcd = o3d.io.read_point_cloud(self.reconstruction_file)
        pcd_orig = pcd

        #publish as orig_cloud
        orig_cloud_pub = rospy.Publisher('/orig_cloud', PointCloud2, queue_size=10, latch=True)
        orig_cloud_pub.publish(self.o3dToROS(pcd))

        #downsample cloud
        pcd = pcd.voxel_down_sample(voxel_size=self.dwn_vox)

        #compute normals and filter out any points that are not on horizontal area
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.normals_search_radius, max_nn=self.normals_kNN))
        normals = np.asarray(pcd.normals)
        pcd = pcd.select_down_sample(np.where((abs(normals[:, 0]) < self.normals_thresh) 
                                                        & (abs(normals[:, 1]) < self.normals_thresh))[0])
        scene = pcd

        #select only points from certain class labels by color
        color = self.get_labels_from_colors(pcd, self.class_labels, self.colors)
        print(color)
        # -np.ones((np.asarray(pcd.colors).shape[0]))
        # for point, i in zip(np.asarray(pcd.colors), range(len(pcd.colors))):
        #     for label in class_labels:
        #         if all(np.isclose(point, colors[label])):
        #             color[i] = label
        index_interest = np.where(color > 0)[0]
        pcd = pcd.select_down_sample(index_interest)

        #publish cloud with horizontal areas only
        orig_cloud_interest_pub = rospy.Publisher('/orig_cloud_interest', PointCloud2, queue_size=10, latch=True)
        orig_cloud_interest_pub.publish(self.o3dToROS(pcd.remove_none_finite_points()))

        size_cof = len(pcd.points)
        outlier_cloud = pcd
        h_planes = []  #list of horizontal planes (parameters)
        h_plane_clouds = [] #list of horizontal plane pointclouds, not clustered

        scene_filtered = scene
        big_clusters_left = True
        #get planes with RANSAC
        while len(outlier_cloud.points)>self.minpoints and big_clusters_left:
            plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.03,
                                                    ransac_n=3,
                                                    num_iterations=1000)
            [a, b, c, d] = plane_model
            print("Plane equation: {}x + {}y + {}z + {} = 0".format(a,b,c,d))

            inlier_cloud = outlier_cloud.select_down_sample(inliers)

            # region growing
            rg = RegionGrowingByDistanceOnly(scene_filtered)
            rg.set_plane(inlier_cloud)
            inlier_cloud, rg_inliers = rg.grow_region() #scene indices

            scene_filtered = scene_filtered.select_down_sample(rg_inliers, invert=True)
            color = -np.ones((np.asarray(scene_filtered.colors).shape[0]))
            for point, i in zip(np.asarray(scene_filtered.colors), range(len(scene_filtered.colors))):
               for label in self.class_labels:
                   if all(np.isclose(point, self.colors[label])):
                      color[i] = label
            index_interest = np.where(color > 0)[0]
            outlier_cloud = scene_filtered.select_down_sample(index_interest)
            #o3d.visualization.draw_geometries([inlier_cloud, pcd])
            plane = Plane()
            plane.x = a
            plane.y = b
            plane.z = c
            plane.d = d
            h_planes.append(plane)
            h_plane_clouds.append(inlier_cloud)
            
            if len(rg_inliers) < self.min_cluster_size:
               big_clusters_left = False
               continue

            #break if there are only small clusters left
            idx = outlier_cloud.cluster_dbscan(self.eps, self.minpoints)
            values = np.unique(idx)
            for c in values:
                #select clustered plane cloud
                cluster_idx = np.where(np.asarray(idx) == c)
                if len(cluster_idx[0]) > self.min_cluster_size:
                   big_clusters_left = True
                   break
                else:
                   big_clusters_left = False

        img = np.zeros([self.height, self.width, 1], dtype=np.int8)
        table = Table()
        h_planeclouds_clustered = []
        cloud_pubs = []

        #rg = RegionGrowing(scene)

        i = 0
        #cluster all horizontal planes and store data
        for planecloud, plane in zip(h_plane_clouds, h_planes):
            idx = planecloud.cluster_dbscan(self.eps, self.minpoints)
            values = np.unique(idx)
            values = values[values>=0]
            for c in values:
                #select clustered plane cloud
                cluster_idx = np.where(np.asarray(idx) == c)
                cloud = planecloud.select_down_sample(cluster_idx[0])
                print(len(cloud.points), self.min_cluster_size)
                if len(cloud.points) > self.min_cluster_size:
                    #region growing
                    #rg.set_plane(cloud)
                    #cloud = rg.grow_region()
                    #publishing clouds
                    cloud_pub = rospy.Publisher(
                        '/cloud'+str(i+1), PointCloud2, queue_size=10, latch=True)
                    #cloud = cloud.paint_uniform_color(colors[i])
                    #o3d.io.write_point_cloud('/home/v4r/data/cloud'+str(i+1)+'.pcd', cloud)
                    #remove already selected points from cloud
                    # for d in cluster_idx[0]:
                    #     pcd.points[d] = np.array(
                    #         [np.nan, np.nan, np.nan], dtype=np.float64).reshape(3, 1)
                    color_labels = self.get_labels_from_colors(cloud, self.class_labels, self.colors)
                    ### TODO
                    #find out which class the plane belongs to
                    unique, counts = np.unique(color_labels, return_counts=True)
                    counts_idx = np.where(counts == np.amax(counts))
                    if unique[counts_idx] == -1:
                        continue
                    class_name = self.class_names[np.where(self.class_labels == unique[counts_idx][0])[0][0]]
                    h_planeclouds_clustered.append(cloud)
                    cloud_pubs.append(cloud_pub)
                    #save tables in mongodb
                    table = Table()
                    hull, indx = cloud.compute_convex_hull()
                    points = np.asarray(hull.vertices)
                    center = hull.get_center()
                    table.id = i 
                    table.category = class_name
                    table.center.point.x = center[0]
                    table.center.point.y = center[1]
                    table.center.point.z = center[2]
                    table.center.header.frame_id = '/map'
                    table.plane = plane
                    table_points = []
                    for point in points:
                        pointstamped = geometry_msgs.msg.PointStamped()
                        pointstamped.point = ros_numpy.msgify(geometry_msgs.msg.Point, point.astype(np.float32))
                        pointstamped.header.frame_id = '/map'
                        table_points.append(pointstamped)
                        #table_points.append(ros_numpy.msgify(geometry_msgs.msg.Point, point.astype(np.float32)))
                    table.points = table_points
                    z = np.mean(points[:, 2])
                    table.height = float(z)
                    msg_store.insert(table)
                    #draw a table map
                    points[:, 0] = np.round((points[:, 0]+self.delta_x)/self.res).astype(np.uint32)
                    points[:, 1] = np.round((points[:, 1]+self.delta_y)/self.res).astype(np.uint32)
                    points = np.delete(points, 2, 1)
                    img_copy = np.zeros([self.height, self.width, 1], dtype=np.int8)

                    points = points.reshape(points.shape[0], 1, 2)
                    point = points.astype(np.int32)[:, :, [0, 1]]
                    cv_hull = cv.convexHull(point)

                    img = cv.drawContours(img, [cv_hull], 0, 255, thickness=-1)
                    i = i+1
            pcd = pcd.remove_none_finite_points()

        #publish table map
        map_pub = rospy.Publisher('table_map', OccupancyGrid, queue_size=1, latch=True)
        table_map = ros_numpy.msgify(OccupancyGrid, img.reshape(self.height, self.width))
        table_map.info.resolution = self.res
        table_map.info.origin.position.x = -self.delta_x
        table_map.info.origin.position.y = -self.delta_y
        table_map.header.frame_id = '/map'
        cv.imwrite("table_map.pgm", img)

        map_pub.publish(table_map)
        #store_mongodb(msg_store, table_map)

        # rg = RegionGrowing(scene)
        # print('Starting region growing')
        # i = 1
        # clouds = []
        # for plane in h_planeclouds_clustered:
        #     rg.set_plane(plane)
        #     plane_new = rg.grow_region()
        #     clouds.append(plane_new)
        #     print('Region growing {} of {}'.format(i, len(h_planeclouds_clustered)))
        #     i = i+1

        try:
            p_id = msg_store.update_named('table_map', table_map, upsert=True)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
        print('Table Extractor script finished.')

        # print('Planes extracted')
        # for cloud_pub, i in zip(cloud_pubs, range(len(cloud_pubs))):
        #     cloud_pub.publish(o3dToROS(h_planeclouds_clustered[i]))
        #     #cloud_pub.publish(o3dToROS(clouds[i]))


    # Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
    def o3dToROS(self, open3d_cloud, frame_id="/map"):
        # The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
        FIELDS_XYZ = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        FIELDS_XYZRGB = FIELDS_XYZ + \
            [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

        # Bit operations
        BIT_MOVE_16 = 2**16
        BIT_MOVE_8 = 2**8

        def convert_rgbUint32_to_tuple(rgb_uint32): return (
            (rgb_uint32 & 0x00ff0000) >> 16, (rgb_uint32 & 0x0000ff00) >> 8, (rgb_uint32 & 0x000000ff))

        def convert_rgbFloat_to_tuple(rgb_float): return convert_rgbUint32_to_tuple(
            int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value))
        # Set "header"
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        # Set "fields" and "cloud_data"
        points = np.asarray(open3d_cloud.points)
        if not open3d_cloud.colors:  # XYZ only
            fields = FIELDS_XYZ
            cloud_data = points
        else:  # XYZ + RGB
            fields = FIELDS_XYZRGB
            # -- Change rgb color from "three float" to "one 24-byte int"
            # 0x00FFFFFF is white, 0x00000000 is black.
            colors = np.floor(np.asarray(open3d_cloud.colors)*255)  # nx3 matrix
            colors = colors[:, 0] * BIT_MOVE_16 + \
                colors[:, 1] * BIT_MOVE_8 + colors[:, 2]
            cloud_data = np.c_[points, colors]

        # create ros_cloud
        return pc2.create_cloud(header, fields, cloud_data)

    def get_labels_from_colors(self, pcd, class_labels, colors):
        color_labels = -np.ones((np.asarray(pcd.colors).shape[0])).astype(np.int64)
        for point, i in zip(np.asarray(pcd.colors), range(len(pcd.colors))):
            for label in self.class_labels:
                if all(np.isclose(point, colors[label])):
                    color_labels[i] = label
        return color_labels

if __name__ == '__main__':
    
    rospy.init_node('table_extractor')
    te = TableExtractor()
    te.execute()

    rospy.spin()
