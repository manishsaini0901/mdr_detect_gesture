import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

class EuclideanDistTracker:
    def __init__(self, min_dist=30, max_disp=50, min_iou=0.5):
        # Store the center positions of the objects
        self.min_dist = min_dist
        self.min_iou = min_iou
        self.max_disp = max_disp
        self.center_points = {}
        self.center_points_iou = {}
        self.disappeared = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def create_mask_from_bbox(self,bboxs, img_shape=(480,640)):
        masks=[]
        for bbox in bboxs:
            # Create a binary mask with the same shape as the image
            mask = np.zeros(img_shape, dtype=np.uint8)
            # Extract the x, y, width, and height values from the bounding box
            x, y, x2, y2 = bbox
            # Set the region defined by the bounding box to white (255) in the mask
            mask[y:y2, x:x2] = 255
            masks.append(mask)
        return masks

    def calculate_iou(self, mask1s, mask2s):
        ious=[]
        for mask1 in mask1s:
            mask1_ious = []
            for mask2 in mask2s:
                intersection = np.logical_and(mask1, mask2)
                union = np.logical_or(mask1, mask2)
                iou = np.sum(intersection) / np.sum(union)
                mask1_ious.append(iou)
                '''intersection = cv2.countNonZero(cv2.bitwise_and(mask1, mask2))
                union = cv2.countNonZero(cv2.bitwise_or(mask1, mask2))
                iou = intersection / union'''
            ious.append(mask1_ious)
        return ious
            
    def centroid(self, rect):
        x1, y1, x2, y2 = rect
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return np.array([cx, cy])
    
    def update_iou(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []
        # Get center point of new object
        masks = self.create_mask_from_bbox(objects_rect)
        obs = np.array(list(self.center_points_iou.values()))
        ids = np.array(list(self.center_points_iou.keys()))
        if len(obs)>0:
            ious = np.array(self.calculate_iou(masks,obs))
            assigned = []
            for i in range(len(masks)):
                closest_index = np.unravel_index(np.argmax(ious, axis=None),ious.shape)
                print(closest_index)
                if np.max(ious) > self.min_iou:
                    assigned.append(closest_index[0])
                    id = ids[closest_index[1]]
                    self.center_points_iou[id] = masks[closest_index[0]]
                    #print(self.center_points)
                    rect = objects_rect[closest_index[0]]
                    area = (rect[1]-rect[3])*(rect[0]-rect[2])
                    objects_bbs_ids.append([rect, id, area])
                    same_object_detected = True
                    self.disappeared[id] = 0
                    ious[closest_index[0]] = (ious[closest_index[0]]*0) -1

            for i in range(len(masks)):
                if i not in assigned:
                    self.center_points_iou[self.id_count] = masks[i]
                    rect = objects_rect[i]
                    area = (rect[1]-rect[3])*(rect[0]-rect[2])
                    objects_bbs_ids.append([rect, self.id_count, area])
                    self.disappeared[self.id_count] = 0
                    self.id_count += 1
        else:
            for i in range(len(masks)):
                self.center_points_iou[self.id_count] = masks[i]
                rect = objects_rect[i]
                area = (rect[1]-rect[3])*(rect[0]-rect[2])
                objects_bbs_ids.append([rect, self.id_count, area])
                self.disappeared[self.id_count] = 0
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            re, object_id, ar = obj_bb_id
            mask = self.center_points_iou[object_id]
            new_center_points[object_id] = mask
        
        disap = [x for x in list(self.center_points_iou.keys()) if x not in list(new_center_points.keys())]
        for i in disap:
            self.disappeared[i] += 1
            if self.disappeared[i]<self.max_disp:
                new_center_points[i] = self.center_points_iou[i]
        # Update dictionary with IDs not used removed
        self.center_points_iou = new_center_points.copy()
        return objects_bbs_ids

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []
        # Get center point of new object
        for rect in objects_rect:
            area = (rect[1]-rect[3])*(rect[0]-rect[2])
            center = self.centroid(rect)
            # Find out if that object was detected already
            same_object_detected = False
            obs = np.array(list(self.center_points.values()))
            ids = np.array(list(self.center_points.keys()))
            if len(obs)>0:
                dists = np.linalg.norm(center-obs, axis=1)
                closest_indx = np.argmin(dists)
                if dists[closest_indx] < self.min_dist:
                    id = ids[closest_indx]
                    self.center_points[id] = center
                    #print(self.center_points)
                    objects_bbs_ids.append([rect, id, area])
                    same_object_detected = True
                    self.disappeared[id] = 0

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = center
                objects_bbs_ids.append([rect, self.id_count, area])
                self.disappeared[self.id_count] = 0
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            re, object_id, ar = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        
        disap = [x for x in list(self.center_points.keys()) if x not in list(new_center_points.keys())]
        for i in disap:
            self.disappeared[i] += 1
            if self.disappeared[i]<self.max_disp:
                new_center_points[i] = self.center_points[i]
        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

class PoseDetector():
    def __init__(self, model_path, seq_len=30, facemesh_det=1, pose_det=1, hand_det=1, face_det=1):
        if not model_path=="":
            classifiername, classifier_extension = os.path.splitext(model_path)
            if classifier_extension==".tflite":
                self.interpreter = tf.lite.Interpreter(model_path=model_path,num_threads=1)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.lite=True
            else:
                self.model = load_model(model_path)
                self.lite=False
        
        self.seq_len = seq_len
        self.gestures = {0: "Nodding", 1: "Stop sign", 2: "Thumbs down", 3: "Waving", 4: "Pointing",
                         5: "Calling someone", 6: "Thumbs up", 7: "Wave someone away", 8: "Shaking head",
                         9: "Talking", 10: "Idle"}
        self.pTime = 0
        self.d = deque(maxlen=60)
        self.fps=0
        self.keys = deque(maxlen=self.seq_len)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.facemesh_det = facemesh_det
        self.face_det= face_det
        self.pose_det = pose_det
        self.hand_det= hand_det
        if self.facemesh_det==True:
            self.mp_face = mp.solutions.face_mesh
            self.face =  self.mp_face.FaceMesh(max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.5,refine_landmarks=False)
        if self.pose_det==True:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)
        if self.hand_det==True:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5)
        if self.face_det==1:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def update_keys(self, image, results1, results2, results3):
        face, lh, rh, pose = self.data2array(image, results1, results2, results3)
        #print(pose.shape,lh.shape,rh.shape)
        key = np.concatenate([lh.flatten(),face.flatten(),pose.flatten(),rh.flatten()])
        self.keys.append(key)
    
    def classify(self):
        X = np.expand_dims(self.keys, axis=0)
        X = X.reshape((X.shape[0], 1, self.seq_len, X.shape[2]))
        if self.lite:
            X = np.array(X, dtype=np.float32)
            input_details_tensor_index = self.input_details[0]['index']
            self.interpreter.set_tensor(input_details_tensor_index,X)
            self.interpreter.invoke()
            output_details_tensor_index = self.output_details[0]['index']
            pred1 = self.interpreter.get_tensor(output_details_tensor_index)
        else:
            pred1 = self.model.predict(X,verbose=0)
        prob = np.max(pred1)
        pred = np.argmax(pred1,axis=1)
        return self.gestures[pred[0]], prob
    
    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results3 = self.pose.process(image)
        results1 = self.face.process(image)
        results2 = self.hands.process(image)
        return results1, results2, results3
    
    def lm2img(self, image, results1, results2, results3):
        h, w = image.shape[:2]
        pose2 = np.array([[int(res.x*w), int(res.y*h)] for res in results3.pose_landmarks.landmark]) if results3.pose_landmarks else np.zeros((33, 2))
        face = np.zeros((468, 3))
        if results1.multi_face_landmarks:
          for face_landmarks in results1.multi_face_landmarks:
              face = np.array([[int(res.x*w), int(res.y*h)] for res in face_landmarks.landmark])
        face = face[:468]
        lh= np.zeros((21, 2))
        rh= np.zeros((21, 2))
        if results2.multi_hand_landmarks:
          for hand_landmarks, handedness in zip(results2.multi_hand_landmarks,results2.multi_handedness):
              side = str(handedness.classification[0].label[0:]).lower()
              if side=='left':
                  lh = np.array([[int(res.x*w), int(res.y*h)] for res in hand_landmarks.landmark])#.flatten()
              else:
                  rh = np.array([[int(res.x*w), int(res.y*h)] for res in hand_landmarks.landmark])#.flatten()
        return face, lh, rh, pose2
    
    def draw(self, image, pose):
        #pose1 = pose[0:15]#np.delete(pose, (0,21), axis = 0)
        for i in pose:
            cv2.circle(image, (int(i[0]),int(i[1])), 2, (255,255,0), -1)
        return image    
        
    def draw_landmarks(self, image, results1, results2, results3):
        # Draw face connections
        if results1.multi_face_landmarks:
          for face_landmarks in results1.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(image=image,landmark_list=face_landmarks,
                                               connections=self.mp_face.FACEMESH_CONTOURS,landmark_drawing_spec=None,
                                               connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image,results3.pose_landmarks,self.mp_pose.POSE_CONNECTIONS,
                                       landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        # Draw hand connections
        if results2.multi_hand_landmarks:
            for hand_landmarks in results2.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image,hand_landmarks,self.mp_hands.HAND_CONNECTIONS,
                                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                        self.mp_drawing_styles.get_default_hand_connections_style())
        return image
        
    def get_fps(self):
        self.cTime = time.time()
        self.fps = 1 / (self.cTime - self.pTime)
        self.d.append(self.fps)
        self.fps = sum(self.d)/len(self.d)
        self.pTime = self.cTime
        return self.fps
    
    def data2array(self, image, results1, results2, results3):
        pose1 = np.array([[res.x, res.y] for res in results3.pose_landmarks.landmark]) if results3.pose_landmarks else np.zeros((33, 2))
        pose1 = pose1-pose1[0]
        scaler = MinMaxScaler()
        pose1 = scaler.fit_transform(pose1)
        face = np.zeros((468, 2))
        if results1.multi_face_landmarks:
          for face_landmarks in results1.multi_face_landmarks:
              face = np.array([[res.x, res.y] for res in face_landmarks.landmark])
        face = face[:468]
        face = face-face[0]
        scaler = MinMaxScaler()
        face = scaler.fit_transform(face)
        lh= np.zeros((21, 2))
        rh= np.zeros((21, 2))
        if results2.multi_hand_landmarks:
          for hand_landmarks, handedness in zip(results2.multi_hand_landmarks,results2.multi_handedness):
              side = str(handedness.classification[0].label[0:]).lower()
              if side=='left':
                  lh = np.array([[res.x, res.y] for res in hand_landmarks.landmark])#.flatten()
              else:
                  rh = np.array([[res.x, res.y] for res in hand_landmarks.landmark])#.flatten()
        lh = lh-lh[0]
        scaler = MinMaxScaler()
        lh = scaler.fit_transform(lh)
        rh = rh-rh[0]
        scaler = MinMaxScaler()
        rh = scaler.fit_transform(rh)
        return face, lh, rh, pose1
    
    def face_detect(self, image, draw=True):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_detection.process(image)
        h, w = image.shape[:2]
        face_boxes = []
        if results.detections:
          for detection in results.detections:
              bboxC = detection.location_data.relative_bounding_box
              xmin, ymin, hw, hh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
              xmax, ymax = xmin+hw, ymin+hh
              face_boxes.append([xmin, ymin, xmax, ymax])
        return face_boxes

class MultiPerDetector():
    def __init__(self,person_detector, num_person, model_path="", seq_len=30):
        self.seq_len = seq_len
        self.num_person = num_person
        self.tracker = EuclideanDistTracker()
        for i in range(self.num_person):
            globals()["per"+str(i)] = PoseDetector(model_path=model_path, seq_len=seq_len)
        self.person_detector = person_detector
        
    def det_per(self,frame):
        bboxs = self.person_detector.detect_person(frame)
        objects_bbs_ids = self.tracker.update_iou(bboxs)
        #print(bboxs,objects_bbs_ids)
        return objects_bbs_ids
    
    def multi_per_gesture(self,image,sort="id",viz=False):
        gestures = []
        objects_bbs_ids = self.det_per(image)
        #print(objects_bbs_ids)
        if sort =="id":
            sorted_objects_bbs_ids = sorted(objects_bbs_ids,key=lambda x: x[1])
        elif sort=="area":
            sorted_objects_bbs_ids = sorted(objects_bbs_ids,key=lambda x: x[2])[::-1]
        for i, bbox in enumerate(sorted_objects_bbs_ids):
            if i<self.num_person:
                person = image[bbox[0][1]:bbox[0][3],bbox[0][0]:bbox[0][2]]
                results1, results2, results3 = globals()["per"+str(i)].mediapipe_detection(person)
                globals()["per"+str(i)].update_keys(image, results1, results2, results3)
                if len(globals()["per"+str(i)].keys)==self.seq_len: #i%10==0 and 
                    gest, prob = globals()["per"+str(i)].classify()
                    gestures.append([gest, prob, bbox[1], bbox[0]]) #gest, prob, id, bbox
                if viz:
                    per = globals()["per"+str(i)].draw_landmarks(person, results1, results2, results3)
                    image[bbox[0][1]:bbox[0][3],bbox[0][0]:bbox[0][2]] = per
            if viz:
                cv2.rectangle(image, (bbox[0][0], bbox[0][1]), (bbox[0][2], bbox[0][3]), (255,255,255), thickness=2)
                cv2.putText(image, str(bbox[1]), (bbox[0][0], int(bbox[0][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return image, gestures
    
    def multi_per_poses(self,image,sort="id",viz=False):
        poses = []
        objects_bbs_ids = self.det_per(image)
        if sort =="id":
            sorted_objects_bbs_ids = sorted(objects_bbs_ids,key=lambda x: x[1])
        elif sort=="area":
            sorted_objects_bbs_ids = sorted(objects_bbs_ids,key=lambda x: x[2])
        for i, bbox in enumerate(sorted_objects_bbs_ids):
            if i<self.num_person:
                person = image[bbox[0][1]:bbox[0][3],bbox[0][0]:bbox[0][2]]
                results1, results2, results3 = globals()["per"+str(i)].mediapipe_detection(person)
                poses.append(results1, results2, results3, bbox[1], bbox[0])
            if viz:
                cv2.rectangle(image, (bbox[0][0], bbox[0][1]), (bbox[0][2], bbox[0][3]), (255,255,255), thickness=2)
                cv2.putText(image, str(bbox[1]), (bbox[0][0], int(bbox[0][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                per = globals()["per"+str(i)].draw_landmarks(person, results1, results2, results3)
                image[bbox[0][1]:bbox[0][3],bbox[0][0]:bbox[0][2]] = per
        return image, poses