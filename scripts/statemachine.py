#!/usr/bin/env python
# Copyright (C) 2016 Toyota Motor Corporation
import rospy
import smach
import smach_ros

from table_extractor_script import TableExtractor
from table_viewpoint import TableViewpoint
#from stare_at_tables.msg import StareAtTablesAction
from obj_det_ppf_matching_msgs.msg import Table, IdAction, Object, CandidateObject, ObjectMatch, ObjectStateClass
from obj_det_ppf_matching_msgs.srv import Id, SparseCN, SparseCNRequest, det_and_compare_obj, extract_permanent_objects

#from elastic_fusion_ros.msg import ElasticFusionAction
from mongodb_store.message_store import MessageStoreProxy
#from table_extractor.msg import Table 
#from png_to_klg.srv import PngToKlg
from std_srvs.srv import Empty
#from sparseconvnet_ros.srv import execute, executeRequest
import subprocess
import os



class UserInputState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['quit', 'start', 'table_patrolling', 'read_rosbag', 'png_to_klg', 
                                            'generate_mesh', 'clear_mesh', 'object_detection', 'start_visualization', 'clear_object_database', 'print_database'],
                                    input_keys=[],
                                    output_keys=['id'])


    def execute(self, userdata):
        rospy.loginfo('Executing state UserInput')
        userdata.id = 0
        while not rospy.is_shutdown():
            self.print_info()
            while True:
                user_input = raw_input('CMD> ')
                if len(user_input) == 1:
                    break
                print('Please enter only one character')
            char_in = user_input.lower()

            # Quit
            if char_in is None or char_in == 'q':
                rospy.loginfo('Quitting')
                return 'quit'
            # table_extractor
            elif char_in == 's':
                rospy.loginfo('Clear Database and start Pipeline with Table Extractor')
                return 'start'
            elif char_in == 't':
                rospy.loginfo('Start Pipeline with Table Patrolling')
                return 'table_patrolling'
            elif char_in == 'r':
                rospy.loginfo('Start Pipeline with Read Rosbag')
                return 'read_rosbag'
            elif char_in == 'p':
                rospy.loginfo('Start Pipeline with PNG to KLG')
                return 'png_to_klg'
            elif char_in == 'm':
                rospy.loginfo('Voxblox: Generate Mesh + SparseConvNet')
                return 'generate_mesh'
            elif char_in == 'c':
                rospy.loginfo('Voxblox: Clear Mesh')
                return 'clear_mesh'
            elif char_in == 'o':
                rospy.loginfo('Object Detection')
                return 'object_detection'
            elif char_in == 'v':
                rospy.loginfo('Visualization')
                return 'start_visualization'
            elif char_in == 'd':
                rospy.loginfo('Clear Object Database')
                return 'clear_object_database'
            elif char_in == 'w':
                rospy.loginfo('Print Database')
                return 'print_database'
            # Unrecognized command
            else:
                rospy.logwarn('Unrecognized command %s', char_in)
    def print_info(self):
        print('m - Voxblox: Generate Mesh + SparseConvNet')
        print('c - Voxblox: Clear Mesh')
        print('s - Start Pipeline from Table Extractor')
        print('t - Start Pipeline with Table Patrolling')
        print('r - Start Pipeline with Read Rosbag')
        print('p - Start Pipeline with PNG to KLG')
        print('o - Object Detection')
        print('v - Start Visualization')
        print('d - Clear Object Database')
        print('w - Print Database')
        print('q - Quit')
        
class ClearDatabaseState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'])
        self.msg_store = MessageStoreProxy()
    
    def execute(self, ud):
        for msg, meta in self.msg_store.query(Table._type):
            self.msg_store.delete(str(meta.get('_id')))
        for msg, meta in self.msg_store.query(Object._type):
            self.msg_store.delete(str(meta.get('_id')))
        for msg, meta in self.msg_store.query(CandidateObject._type):
            self.msg_store.delete(str(meta.get('_id')))
        return 'succeeded'

class ClearObjectDatabaseState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'])
        self.msg_store = MessageStoreProxy()
    
    def execute(self, ud):
        for msg, meta in self.msg_store.query(Object._type):
            self.msg_store.delete(str(meta.get('_id')))
        for msg, meta in self.msg_store.query(CandidateObject._type):
            self.msg_store.delete(str(meta.get('_id')))
        return 'succeeded'

class PrintDatabase(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded'])
        self.msg_store = MessageStoreProxy()
    
    def execute(self, ud):
        for msg, meta in self.msg_store.query(CandidateObject._type):
            print('State: ', msg.state, 'Match: ', msg.match, 'Object plane id: ', msg.object.plane_id)
            print(msg.object.obj_cloud)
        return 'succeeded'


class FetchReconstructionFile(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['succeeded', 'aborted'])

        self.remote_path = rospy.get_param('/table_extractor/remote_path', 
            'v4r@rocco:/media/v4r/FF64-D891/tidy_up_pipeline/hsrb_result_pred_legend_21.ply')
        self.local_path = rospy.get_param('/table_extractor/local_path', 
            '/home/v4r/Markus_L/reconstruction.ply')
    
    def execute(self, ud):
        cmd_move = ['scp', self.remote_path, self.local_path]
        move = subprocess.Popen(cmd_move, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        move.wait()
        if os.path.exists(self.local_path):
            return 'succeeded'
        else:
            return 'aborted'

class TableExtractorState(smach.State):
    def __init__(self):
        smach.State.__init__(self,input_keys=[], outcomes=['succeeded', 'aborted'])
        
    def execute(self, userdata):
        rospy.loginfo('Executing State TableExtractor')
        te = TableExtractor()
        te.execute()
        rospy.loginfo('Finished State TableExtractor')
        return 'succeeded'

class ViewpointGeneratorState(smach.State):
    def __init__(self):
        smach.State.__init__(self,input_keys=[], outcomes=['succeeded', 'aborted'])
        
    def execute(self, userdata):
        rospy.loginfo('Executing State ViewpointGenerator')
        viewpoint = TableViewpoint()
        viewpoint.search_for_viewpoint()
        rospy.loginfo('Finished State ViewpointGenerator')
        return 'succeeded'

class GenerationFinishedState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes= ['more_tables', 'all_tables'], output_keys=['id'])
        self.msg_store = MessageStoreProxy()
        self.id_counter = 0

    def execute(self, userdata):
        i = 0
#        userdata.id = self.id_counter
#        if self.id_counter < len(self.msg_store.query(Table._type)):
#            self.id_counter += 1
#            if 
#            return 'more_tables'
#        else:
#            self.id_counter = 0
#            return 'all_tables'
        for msg, meta in self.msg_store.query(Table._type):
            #if msg.category == 'table':
                print(msg.id)
                print(msg.category)
                if i >= self.id_counter:
                    userdata.id = msg.id
                    self.id_counter += 1
                    return 'more_tables'
                i += 1
        return 'all_tables'

class ObjectDetection(smach.State):
    def __init__(self):
        smach.State.__init__(self,input_keys=[], outcomes=['succeeded', 'aborted'])

        self.det_perm_obj = rospy.ServiceProxy('obj_det_and_matching/extract_perm_objects', extract_permanent_objects)        
        self.det_and_compare_obj = rospy.ServiceProxy('obj_det_and_matching/det_and_compare_objects', det_and_compare_obj)

        self.msg_store = MessageStoreProxy()

    def execute(self, userdata):
        rospy.loginfo('Executing State ObjectDetection')
        try:
            rospy.wait_for_service('/obj_det_and_matching/extract_perm_objects', timeout=5)
        except:
            print('timeout')
        self.permanent_objects_flag = False

        for msg, meta in self.msg_store.query(Object._type):
            print('Object')
            print (msg.id, msg.plane_id, msg.object_path)
            self.permanent_objects_flag = True


        if not self.permanent_objects_flag:
            try:
                i = 0            
                self.permanent_objects_flag = True    
                rospy.loginfo('Call extract_perm_objects Service')
                for msg, meta in self.msg_store.query(Table._type):
                    self.det_perm_obj([i])
                    i+=1
            except rospy.ServiceException as e:                
                rospy.loginfo('Service call failed: %s'%e)
                return 'aborted'
        else:
            try:
                i = 0                
                rospy.loginfo('Call det_and_compare_obj Service')
                for msg, meta in self.msg_store.query(Table._type):
                    self.det_and_compare_obj([i])
                    i+=1
            except rospy.ServiceException as e:                
                rospy.loginfo('Service call failed: %s'%e)
                return 'aborted'
        rospy.loginfo('Finished State ObjectDetection')
        return 'succeeded'

def main():
    rospy.init_node('tidy_up_statemachine')

    sm = smach.StateMachine(outcomes=['end'])
    with sm:


        smach.StateMachine.add('USER_INPUT_STATE', \
                               UserInputState(), \
                               transitions={'quit':'end', 
                                            'start':'CLEAR_DATABASE_STATE',
                                            'table_patrolling':'GENERATION_FINISHED_STATE',
                                            'read_rosbag':'READ_ROSBAG',
                                            'png_to_klg':'PNG_TO_KLG',
                                            'generate_mesh':'GENERATE_MESH',
                                            'clear_mesh':'CLEAR_MESH',
                                            'object_detection':'OBJECT_DETECTION',
                                            'start_visualization':'START_VISUALIZATION',
                                            'clear_object_database':'CLEAR_OBJECTS',
                                            'print_database':'PRINT_DATABASE'})
        
        smach.StateMachine.add('CLEAR_DATABASE_STATE', \
                               ClearDatabaseState(), \
                               transitions={'succeeded':'FETCH_RECONSTRUCTION_FILE'})
        
        smach.StateMachine.add('FETCH_RECONSTRUCTION_FILE', \
                               FetchReconstructionFile(), \
                               transitions={'succeeded':'TABLE_EXTRACTOR_STATE',
                                            'aborted':'USER_INPUT_STATE'})
        
        smach.StateMachine.add('TABLE_EXTRACTOR_STATE',
                                TableExtractorState(), \
                                transitions={'succeeded':'VIEWPOINT_GENERATOR_STATE',
                                            'aborted':'end'})
        
        smach.StateMachine.add('VIEWPOINT_GENERATOR_STATE',
                                ViewpointGeneratorState(), \
                                transitions={'succeeded':'GENERATION_FINISHED_STATE',
                                            'aborted':'end'})

        smach.StateMachine.add('GENERATION_FINISHED_STATE',
                                GenerationFinishedState(), \
                                transitions={'more_tables' : 'STARE_AT_TABLES',
                                            'all_tables' : 'OBJECT_DETECTION'})

        smach.StateMachine.add('STARE_AT_TABLES', \
                                smach_ros.SimpleActionState('stare_at_tables', IdAction, 
                                                            goal_slots = ['id']),
                                transitions={'succeeded':'READ_ROSBAG', 
                                            'preempted':'USER_INPUT_STATE',
                                            'aborted':'GENERATION_FINISHED_STATE'})

        smach.StateMachine.add('READ_ROSBAG', \
                                smach_ros.SimpleActionState('read_rosbag', IdAction, 
                                                            goal_slots = ['id']),
                                transitions={'succeeded':'PNG_TO_KLG', 
                                            'preempted':'USER_INPUT_STATE',
                                            'aborted':'USER_INPUT_STATE'})
        
        smach.StateMachine.add('PNG_TO_KLG',
                                smach_ros.ServiceState('png_to_klg',
                                                Id,
                                                request_slots = ['id']),
                                transitions={'succeeded':'ELASTIC_FUSION_ROS', 
                                            'preempted':'USER_INPUT_STATE',
                                            'aborted':'USER_INPUT_STATE'})

        smach.StateMachine.add('ELASTIC_FUSION_ROS',
                                smach_ros.SimpleActionState('elastic_fusion_ros', IdAction, 
                                                            goal_slots = ['id']),
                                transitions={'succeeded':'GENERATION_FINISHED_STATE', 
                                            'preempted':'USER_INPUT_STATE',
                                            'aborted':'USER_INPUT_STATE'})

        smach.StateMachine.add('GENERATE_MESH',
                                smach_ros.ServiceState('voxblox_node/generate_mesh',
                                                Empty),
                                transitions={'succeeded':'SPARSE_CONV_NET', 
                                            'preempted':'USER_INPUT_STATE',
                                            'aborted':'USER_INPUT_STATE'})

        smach.StateMachine.add('CLEAR_MESH',
                                smach_ros.ServiceState('voxblox_node/clear_map',
                                                Empty),
                                transitions={'succeeded':'USER_INPUT_STATE', 
                                            'preempted':'USER_INPUT_STATE',
                                            'aborted':'USER_INPUT_STATE'})

        smach.StateMachine.add('SPARSE_CONV_NET',
                                smach_ros.ServiceState('sparseconvnet_ros/sparseconvnet_ros_service/execute',
                                                SparseCN, request=SparseCNRequest('/root/share/hsrb_result.ply')),
                                transitions={'succeeded':'USER_INPUT_STATE', 
                                            'preempted':'USER_INPUT_STATE',
                                            'aborted':'USER_INPUT_STATE'})

        smach.StateMachine.add('OBJECT_DETECTION',
                                ObjectDetection(), \
                                transitions={'succeeded' : 'USER_INPUT_STATE',
                                            'aborted' : 'USER_INPUT_STATE'})
        

        smach.StateMachine.add('START_VISUALIZATION',
                                smach_ros.ServiceState('object_visualization_service/visualize',Empty),
                                transitions={'succeeded':'USER_INPUT_STATE', 
                                            'preempted':'USER_INPUT_STATE',
                                            'aborted':'USER_INPUT_STATE'})
        smach.StateMachine.add('CLEAR_OBJECTS',
                                ClearObjectDatabaseState(),
                                transitions={'succeeded':'USER_INPUT_STATE'})

        smach.StateMachine.add('PRINT_DATABASE',
                                PrintDatabase(),
                                transitions={'succeeded':'USER_INPUT_STATE'})
                                            



    # Create and start the introspection server
    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()

    #Execute state machine
    outcome = sm.execute()

    # Wait for ctrl-c to stop the application
    #rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()
