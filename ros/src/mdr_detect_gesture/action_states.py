#!/usr/bin/python
import rospy
import cv2
import numpy as np
import time
from mdr_detect_gesture.inference import MultiPerDetector
from mdr_detect_gesture.person_det_media import Person_Detector
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pyftsm.ftsm import FTSMTransitions
from mas_execution.action_sm_base import ActionSMBase
from mdr_detect_gesture.msg import DetectGestureResult
from mdr_perception_msgs.msg import BodyBoundingBox
from tensorflow.keras.models import load_model
# from mdr_detect_gesture.inference import load_detection_model, detect_faces


class DetectGestureSM(ActionSMBase):
    def __init__(self, timeout=120.,
                 image_topic='gesture_image', #TO_DO : change topic
                 detection_model_path='',   #TO_DO : add model path
                 max_recovery_attempts=1):
        super(DetectGestureSM, self).__init__('DetectGesture', [], max_recovery_attempts)
        self.timeout = timeout
        self.detection_model_path = detection_model_path
        self.bridge = CvBridge()
        self.image_publisher = rospy.Publisher(image_topic, Image, queue_size=1)
        self.model = None
        self.gesture_detection = None
        self.num_person = 1
        self.rp = Person_Detector()
        self.image = Image()

    def init(self):
        try:
            rospy.loginfo('[detect_person] Loading detection model %s', self.detection_model_path)
            self.model = MultiPerDetector(self.rp,model_path='/home/zany/catkin_hsr_ws/src/mas_domestic_robotics/mdr_planning/mdr_actions/mdr_perception_actions/mdr_detect_gesture/model/action30_10_2.tflite',num_person= self.num_person)
        except Exception as exc:
            rospy.logerr('[detect_gesture] Model %s could not be loaded: %s',
                         self.detection_model_path, str(exc))
        return FTSMTransitions.INITIALISED

    def image_callback(self,msg):
        self.image = msg
        # print(self.image)
        input_image = self.__convert_image(self.image)
        # frame_width = int(input_image.get(3))
        # frame_height = int(input_image.get(4))
        try:
            bgr_image = input_image
            # gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            image, gestures = self.model.multi_per_gesture(input_image,sort="area",viz=False)
            print("gestures", gestures)

            for gest in gestures:
                gesture = gesture[0]
                confidence = gesture[1]
                id = gesture[2]
                self.bounding_box.bounding_box_coordinates = gesture[3]
                print("begore cv to ros")
                output_ros_image = self.bridge.cv2_to_imgmsg(rgb_image, 'rgb8')
                print("after cv to ros")
                self.image_publisher.publish(output_ros_image)

            # for face_coordinates in faces:
            #     bounding_box = FaceBoundingBox()
            #     bounding_box.bounding_box_coordinates = face_coordinates.tolist()
            #     bounding_boxes.append(bounding_box)

        except:
            detection_successful = False

    def running(self):
        self.bounding_box = BodyBoundingBox()
        id = 0
        gesture = ''
        confidence = 0.0    
        detection_successful = True
        rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw", Image, self.image_callback)
        print("goal image", self.image)


        self.result = self.set_result(detection_successful, gesture, confidence, self.bounding_box, id)
        return FTSMTransitions.DONE

    def __convert_image(self, ros_image):
        print("begore ros to cv")
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, 'bgr8')
        print("after ros to cv")
        
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        # print(cv_image)
        return cv_image
        # return np.array(cv_image, dtype=np.uint8)

    def set_result(self, detection_successful, gesture, confidence, bounding_box, id):
        result = DetectGestureResult()
        result.success = detection_successful
        result.gesture = gesture
        result.confidence = confidence
        result.id = id
        result.bounding_box = bounding_box
        return result