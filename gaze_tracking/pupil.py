import numpy as np
import cv2
from PIL import Image

class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    def __init__(self, eye_frame, threshold,side):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        #self.detect_iris(eye_frame,side)
        # preprocess_iris
        pre_eys_frame = self.preprocess_iris(eye_frame,side)
        
        self.detect_iris(pre_eys_frame,side)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Performs operations on the eye frame to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame

        Returns:
            A frame with a single element representing the iris
        """
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame
        
    def preprocess_iris(self,eye_frame,side):
        
        result = None
        self.iris_frame = self.image_processing(eye_frame, self.threshold)
        if side == 0:
            cv2.imwrite("left_eye_frame0.jpg",eye_frame)
            cv2.imwrite("left_iris_frame0.jpg",self.iris_frame)
        elif side == 1:
            cv2.imwrite("right_eye_frame0.jpg",eye_frame)
            cv2.imwrite("right_iris_frame0.jpg",self.iris_frame)
            
        #contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        contours, hierarchy = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        # 查找轮廓
        #contours, hierarchy = cv2.findContours(thresh, mode, method)
        iris_frame_mask = np.array(self.iris_frame, np.uint8) / 255
        iris_image_face = np.ones_like(np.array(self.iris_frame)) * 255
        eye_image_face = Image.fromarray(np.uint8( (1-iris_frame_mask)*np.array(eye_frame) +  iris_image_face * iris_frame_mask))
        image = np.array(eye_image_face)
        if side == 0:
            cv2.imwrite("left_ipa_image_face.jpg",image)
            _,new_frame = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
            cv2.imwrite("left_new_frame.jpg", new_frame)
            result = cv2.bitwise_and(eye_frame, new_frame)
            cv2.imwrite(str(side)+"preprocess_iris.jpg",result)
        elif side == 1:
            cv2.imwrite("right_ipa_image_face.jpg",image)
            _,new_frame = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
            cv2.imwrite("right_new_frame.jpg", new_frame)
            result = cv2.bitwise_and(eye_frame, new_frame)
            cv2.imwrite(str(side)+"preprocess_iris.jpg",result)        
        
        return result
        
    def detect_iris(self, eye_frame,side):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """
        
        #scale_factor = 0.8
        # 获取原始图像的尺寸
        #height, width = eye_frame.shape[:2]
        # 计算调整后的目标尺寸
        #new_width = int(width * scale_factor)
        #new_height = int(height * scale_factor)
        #new_size = (new_width, new_height)
        
        # 调整图像大小
        #eye_frame = cv2.resize(eye_frame, new_size)
        # 将图像转换为 NumPy 数组
        #image_array = np.array(eye_frame, np.uint8) / 255
        #height, width,c = eye_frame.shape
        #print(f"height:{height},width:{width},c:{c}")
        #new_frame = cv2.threshold(eye_frame, 0, 255, cv2.THRESH_BINARY)
        #print(f"new_frame:{new_frame}")
        #image_mask = np.array(new_frame, np.uint8) / 255
        
        
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

            
        #contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        contours, hierarchy = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        # 查找轮廓
        #contours, hierarchy = cv2.findContours(thresh, mode, method)
        iris_frame_mask = np.array(self.iris_frame, np.uint8) / 255
        iris_image_face = np.ones_like(np.array(self.iris_frame)) * 255
        eye_image_face = Image.fromarray(np.uint8( (1-iris_frame_mask)*np.array(eye_frame) +  iris_image_face * iris_frame_mask))
        image = np.array(eye_image_face)
        if side == 0:
            cv2.imwrite("left_ipa_image_face.jpg",image)
            _,new_frame = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
            cv2.imwrite("left_new_frame.jpg", new_frame)
        elif side == 1:
            cv2.imwrite("right_ipa_image_face.jpg",image)
            _,new_frame = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
            cv2.imwrite("right_new_frame.jpg", new_frame)
        
        # 绘制轮廓
        color = (0, 255, 0) 
        #image_with_contours = cv2.drawContours(self.iris_frame, contours, -1, color, 1)
        #if side == 0:
            #cv2.imwrite("left_contours.jpg",image_with_contours)
            #cv2.imwrite("left_eye_frame.jpg",eye_frame)
            #cv2.imwrite("left_iris_frame.jpg",self.iris_frame)
        #elif side == 1:
            #cv2.imwrite("right_contours.jpg",image_with_contours)
            #cv2.imwrite("right_eye_frame.jpg",eye_frame)
            #cv2.imwrite("right_iris_frame.jpg",self.iris_frame)
        
        contours = sorted(contours, key=cv2.contourArea)
        # 计算轮廓面积
        tmp = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            print(f"tmp:{tmp},Contour area:{area}")
            tmp = tmp + 1
        contourtmp = cv2.drawContours(self.iris_frame, contours[-2], -2, color, 1)
        cv2.imwrite(str(side)+str(tmp)+"contourtmp.jpg",contourtmp)  
        try:
            moments = cv2.moments(contours[-2])
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
            if side == 0:
                print(f"left,self.x:{self.x},self.y:{self.y}")
            elif side == 1:
                print(f"right,self.x:{self.x},self.y:{self.y}")
        except (IndexError, ZeroDivisionError):
            pass
