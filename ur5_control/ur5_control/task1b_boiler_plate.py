#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Logistic coBot (LB) Theme (eYRC 2024-25)
*        		===============================================
*
*  This script should be used to implement Task 1B of Logistic coBot (LB) Theme (eYRC 2024-25).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ 1685]
# Author List:		[ Vedhamsh Bode, Dhananjay Abbot ]
# Filename:		    task1b_boiler_plate.py
# Functions:
#			        [ Comma separated list of functions in this file ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/aligned_depth_to_color/image_raw, /etc... ]


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import math
import rclpy.time
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, Image
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from tf2_ros import TransformException


##################### FUNCTION DEFINITIONS #######################

def calculate_rectangle_area(coordinates):

    area = None
    width = None
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = coordinates.reshape(4, 2)
    width1 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Length between (x1, y1) and (x2, y2)
    width2 = np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)  # Length between (x4, y4) and (x3, y3)
    height1 = np.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2)  # Length between (x1, y1) and (x4, y4)
    height2 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)  # Length between (x2, y2) and (x3, y3)

    width = (width1 + width2) / 2
    height = (height1 + height2) / 2

    area = width * height
    
    return area, width


def detect_aruco(frame):
    '''
    Description:    Function to perform aruco detection and return each detail of aruco detected 
                    such as marker ID, distance, angle, width, center point location, etc.

    Args:
        image                   (Image):    Input image frame received from respective camera topic

    Returns:
        center_aruco_list       (list):     Center points of all aruco markers detected
        distance_from_rgb_list  (list):     Distance value of each aruco markers detected from RGB camera
        angle_aruco_list        (list):     Angle of all pose estimated for aruco marker
        width_aruco_list        (list):     Width of all detected aruco markers
        ids                     (list):     List of all aruco marker IDs detected in a single frame 
    '''

    aruco_area_threshold = 1500
    cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]])
    dist_mat = np.array([0.0,0.0,0.0,0.0,0.0])
    size_of_aruco_m = 0.15

    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_aruco_list = []
    ids = []
    coordinates=[]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    areas = []

    try:
        if frame.shape[0] > 0 and frame.shape[1] > 0:
            frame_np = np.array(frame)
            gray_img = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            corners, ids, rejected = detector.detectMarkers(gray_img)

            try:
                for i in range(len(ids)) :
                    current_marker_corners = [corners[i]]
                    center_coor = [corners[i][0][0][0]+(corners[i][0][2][0]-corners[i][0][0][0])/2, corners[i][0][0][1]-(corners[i][0][0][1]-corners[i][0][2][1])/2]
                    ret = cv2.aruco.estimatePoseSingleMarkers(current_marker_corners, size_of_aruco_m, cameraMatrix=cam_mat, distCoeffs=dist_mat)
                    rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]
                    center_aruco_list.append(center_coor)
                    angle_aruco_list.append(rvec)
                    area,width=calculate_rectangle_area(current_marker_corners[0][0])
                    areas.append(area)
                    width_aruco_list.append(width)

                    if area>aruco_area_threshold:
                        cv2.aruco.drawDetectedMarkers(frame,current_marker_corners)
                        cv2.putText(frame,f"id={ids[i]}",(int(center_coor[0]),int(center_coor[1])),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2,cv2.LINE_AA)
                        cv2.circle(frame,(int(center_coor[0]),int(center_coor[1])), 5, (255,0,0), 5)
                        cv2.drawFrameAxes(frame, cam_mat, dist_mat, rvec, tvec, length =1)
                        cv2.putText(frame,f"center",(int(center_coor[0]),int(center_coor[1])),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2,cv2.LINE_AA)    

            except:
                print("no arucos detected")

            frame = cv2.resize(frame, (0,0), fx=0.5, fy = 0.5)
            cv2.imshow("RGB_camera", frame)
            
        else:
            print("Invalid image dimensions.")



        return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids,areas
    except Exception as e:
        print(e)

class aruco_tf(Node):

    def __init__(self):

        super().__init__('aruco_tf_publisher')                                          # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)
        self.publisher_id = self.create_publisher(Float32MultiArray, 'id_topic', 10)

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)
        self.id = []
        self.id_sort = []
        self.id_unsort = []

        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                        # depth image variable (from depthimagecb())
        self.flag=0

    def depthimagecb(self, data):
        try:
            bgr_depth_camera = self.bridge.imgmsg_to_cv2(data)
            self.depth_image=bgr_depth_camera
            cv2.waitKey(1)
        except:
            print("Not able to get depth image")


    def colorimagecb(self, data):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
            self.flag=1
        except:
             print("Not able to get colour image")
            

    def process_image(self):
        
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 640 
        centerCamY = 360
        focalX = 931.1829833984375
        focalY = 931.1829833984375

        try:
            if self.flag == 1:
                center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids,areas = detect_aruco(self.frame)
                try:
                    self.id =[]
                    print(f"length of ids - {len(ids)}")
                    for i in range(len(ids)):

                        self.id.append(ids[i][0])
                        xp=int(center_aruco_list[i][0])
                        yp=int(center_aruco_list[i][1])

                        depth_value=self.depth_image[yp,xp]
                        c=depth_value
                        c=float(c/1000)
                        distance_from_rgb_list.append(depth_value)
                        #   ->  Use this formula to rectify x, y, z based on focal length, center value and size of image
                    #       x = distance_from_rgb * (sizeCamX - cX - centerCamX) / focalX
                    #       y = distance_from_rgb * (sizeCamY - cY - centerCamY) / focalY
                    #       z = distance_from_rgb
                    #       where, 
                    #               cX, and cY from 'center_aruco_list'
                    #               distance_from_rgb is depth of object calculated in previous step
                    #               sizeCamX, sizeCamY, centerCamX, centerCamY, focalX and focalY are defined above

                        x = c*(sizeCamX - center_aruco_list[i][0] - centerCamX) / focalX 
                        y = c* (sizeCamY - center_aruco_list[i][1] - centerCamY) / focalY
                        z = c
                        # print("c ",c)

                    #   ->  Use this equation to correct the input aruco angle received from cv2 aruco function 'estimatePoseSingleMarkers' here
                    #       It's a correction formula- 
                    #       angle_aruco = (0.788*angle_aruco) - ((angle_aruco**2)/3160)
                        r,p,ya = angle_aruco_list[i]

                        r_1 = 0
                        p_1 = 3.14
                        ya_1 = ya + (0.788*ya) - ((ya**2)/3160) + 1.57

                        print("id,r,p,y",ids[i][0],r,p,ya)

                        # if abs(ya)>1.55:  #rack condition
                        #     r=1.57
                        #     ya=1.57-((0.788*ya) - ((ya**2)/3160))
                        
                        # else:  #table condition
                        #     r=3.14
                        #     ya=1.53+ya

                        x_rot, y_rot, z_rot, w_rot = quaternion_from_euler(r_1,p_1,ya_1)
            
            #   ->  Now, mark the center points on image frame using cX and cY variables with help of 'cv2.cirle' function ########## ALREADY DONE IN DETECT ARUCO FUNCTION

            #   ->  Here, till now you receive coordinates from camera_link to aruco marker center position. 
            #       So, publish this transform w.r.t. camera_link using Geometry Message - TransformStamped 
            #       so that we will collect it's position w.r.t base_link in next step.
            #       Use the following frame_id-
            #           frame_id = 'camera_link'
            #           child_frame_id = 'cam_<marker_id>'          Ex: cam_20, where 20 is aruco marker ID
                        t = TransformStamped()
                        t.header.stamp = self.get_clock().now().to_msg()                        # select transform time stamp as current clock time
                        # frame IDs
                        t.header.frame_id = 'camera_link'                                         # parent frame link with whom to send transform
                        t.child_frame_id = f'cam_{ids[i][0]}'                                              # child frame link from where to send transfrom
                        t.transform.translation.x = z
                        t.transform.translation.y = x                                        # distance offset in Y axis of 2 units
                        t.transform.translation.z = y
                        t.transform.rotation.x = x_rot  
                        t.transform.rotation.y = y_rot 
                        t.transform.rotation.z = z_rot 
                        t.transform.rotation.w = w_rot                                      # rotation 0 degrees

                        self.br.sendTransform(t)                                    # publish transform as defined in 't'

            #   ->  Then finally lookup transform between base_link and obj frame to publish the TF
            #       You may use 'lookup_transform' function to pose of obj frame w.r.t base_link 
                        from_frame_rel = f'cam_{ids[i][0]}'                                                                        # frame from which transfrom has been sent
                        to_frame_rel = 'base_link'                                                                      # frame to which transfrom has been sent

                        try:
                            print("into the try loop")
                            q = self.tf_buffer.lookup_transform( to_frame_rel, from_frame_rel, rclpy.time.Time())       # look up for the transformation between 'obj_1' and 'base_link' frames
                            self.get_logger().info('Successfully received data!')
                            b_tf = TransformStamped()
                            b_tf.header.stamp = self.get_clock().now().to_msg()                        # select transform time stamp as current clock time
                                # frame IDs
                            b_tf.header.frame_id = 'base_link'                                         # parent frame link with whom to send transform
                            b_tf.child_frame_id = f'obj_{ids[i][0]}'                                             # child frame link from where to send transfrom
                            

                            # x_ = q.transform.rotation.x
                            # y_ = q.transform.rotation.y
                            # z_ = q.transform.rotation.z
                            # w_ = q.transform.rotation.w
                            # roll, pitch, yaw = euler_from_quaternion([x_, y_, z_, w_])

                            # pitch =pitch + 1.57
                            # print("ID - roll, pitch, yaw :", ids[i][0], " - ", roll, pitch, yaw)

                            # x_rot_new, y_rot_new, z_rot_new, w_rot_new = quaternion_from_euler(roll, pitch, yaw)

                            b_tf.transform.translation.x = q.transform.translation.x
                            b_tf.transform.translation.y = q.transform.translation.y                                        # distance offset in Y axis of 2 units
                            b_tf.transform.translation.z = q.transform.translation.z
                            b_tf.transform.rotation.x = x_rot
                            b_tf.transform.rotation.y = y_rot
                            b_tf.transform.rotation.z = z_rot  
                            b_tf.transform.rotation.w = w_rot                       # rotation 0 degrees

                            self.br.sendTransform(b_tf)                                    # publish transform as defined in 't'

                            from_frame_rel_ee = 'ee_link'                                                                        # frame from which transfrom has been sent
                            to_frame_rel_b = 'base_link' 

                            #*******EE TO BASE LINK TRANSFROMATION*************
                            ee_transform = self.tf_buffer.lookup_transform( to_frame_rel_b, from_frame_rel_ee, rclpy.time.Time()) 
                            ee_tf = TransformStamped()
                            ee_tf.header.stamp = self.get_clock().now().to_msg()  
                            ee_tf.header.frame_id = 'base_link'                         # parent frame link with whom to send transform
                            ee_tf.child_frame_id = 'ee_link_b'

                            ee_tf.transform.translation.x = ee_transform.transform.translation.x
                            ee_tf.transform.translation.y = ee_transform.transform.translation.y                        # distance offset in Y axis of 2 units
                            ee_tf.transform.translation.z = ee_transform.transform.translation.z
                            ee_tf.transform.rotation.x = ee_transform.transform.rotation.x
                            ee_tf.transform.rotation.y = ee_transform.transform.rotation.y
                            ee_tf.transform.rotation.z = ee_transform.transform.rotation.z
                            ee_tf.transform.rotation.w = ee_transform.transform.rotation.w                       # rotation 0 degrees

                            self.br.sendTransform(ee_tf)    
                            # from_frame_rel_bee = f'obj_{ids[i][0]}'                                                                        # frame from which transfrom has been sent
                            # to_frame_rel_bee = 'ee_link_b'

                            #*******BOX TO EE TRANSFORM************
                            # bee_transform = self.tf_buffer.lookup_transform( to_frame_rel_bee, from_frame_rel_bee, rclpy.time.Time()) 
                            # bee_tf = TransformStamped()
                            # bee_tf.header.stamp = self.get_clock().now().to_msg()                        # select transform time stamp as current clock time
                            #     # frame IDs
                            # bee_tf.header.frame_id = 'ee_link_b'                                         # parent frame link with whom to send transform
                            # bee_tf.child_frame_id = f'bee_{ids[i][0]}'                                              # child frame link from where to send transfrom
                            
                            # bee_tf.transform.translation.x = bee_transform.transform.translation.x
                            # bee_tf.transform.translation.y = bee_transform.transform.translation.y                                        # distance offset in Y axis of 2 units
                            # bee_tf.transform.translation.z = bee_transform.transform.translation.z
                            # bee_tf.transform.rotation.x = bee_transform.transform.rotation.x
                            # bee_tf.transform.rotation.y = bee_transform.transform.rotation.y
                            # bee_tf.transform.rotation.z = bee_transform.transform.rotation.z
                            # bee_tf.transform.rotation.w = bee_transform.transform.rotation.w                       # rotation 0 degrees

                            # self.br.sendTransform(bee_tf)       
                    
                        except TransformException as e:
                            self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {e}')
                except:
                    pass
                self.id_unsort = self.id
                print(f"self.id-unsort - {self.id_unsort}")

                self.id.sort()
                msg = Float32MultiArray()
                msg.data = []
                print(f"self.id - {self.id}")
            
                for i in range(len(self.id)):
                    msg.data.append(float(self.id[i]))
                
                print(msg.data)
                
                angle_aruco_id_dict = dict()
                for i in range(len(self.id_unsort)):
                    print("angle_aruco_list[i][2] ", angle_aruco_list[i][2])
                    angle_aruco_id_dict[angle_aruco_list[i][2]]=self.id_unsort[len(self.id)-i-1]

                print("angle_aruco_id_dict ",angle_aruco_id_dict)

                sorted_aruco_angles = [key for key, value in sorted(angle_aruco_id_dict.items(), key=lambda item: item[1])]

                print("sorted_angle_aruco list ",sorted_aruco_angles)
                
                for i in sorted_aruco_angles:
                    print("angle aruco ", float(i))
                    msg.data.append(float(i))

                
                # print("chutiya data: ",msg.data)
            
                
    
                self.publisher_id.publish(msg)
                print("gand mara")

        except Exception as e:
            print(e)

def main():

    rclpy.init(args=sys.argv)                                       # initialisation
    node = rclpy.create_node('aruco_tf_process')                    # creating ROS node
    node.get_logger().info('Node created: Aruco tf process')        # logging information
    aruco_tf_class = aruco_tf()                                     # creating a new object for class 'aruco_tf'
    rclpy.spin(aruco_tf_class)                                      # spining on the object to make it alive in ROS 2 DDS
    aruco_tf_class.destroy_node()                                   # destroy node after spin ends
    rclpy.shutdown()                                                # shutdown process

if __name__ == '__main__':
    main()