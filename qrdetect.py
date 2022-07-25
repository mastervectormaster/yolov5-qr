"""
    QR Code Detection Module based on Yolov5
    2022.7.24
    @vectorgeek
"""
from os.path import exists
import torch
import cv2


class QRDetect:
    """
        QR Code Detection Class based on Yolov5
    """
    def __init__(self, model_path='qr-yolov5.pt'):
        if not exists(model_path):
            print("Can not find the model.")
            return None
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.initialized = True
        print("Model initialized.")
    
    def detect(self, image_path, need_draw=False):
        """
            Detect QR Code from specified image.
            @param image_path: image containing (or not) QR Code.
            @return box: box coordinates (xmin, ymin, xmax, ymax) if detected, None otherwise
        """

        if not self.initialized:
            print("Model is not initialized.")
            return None
        if not exists(image_path):
            print("Can not find the image.")
            return None
        detect_results = self.model(image_path)
        sorted_by_confidence_results = detect_results.pandas().xyxy[0].sort_values(by=['confidence'], ascending=False)
        if len(sorted_by_confidence_results) == 0:
            print("No QR Code detected")
            return None
        most_confident_box = sorted_by_confidence_results.iloc[0]
        if need_draw:
            self.draw_box(image_path, most_confident_box)
        return most_confident_box

    def draw_box(self, image_path, box):
        """
            Draw bounding box on the image.
            @param image_path: image containing (or not) QR Code.
            @param box: box coordinates (xmin, ymin, xmax, ymax)
            @output: new image with bounding box drawn on the original one
        """
        image = cv2.imread(image_path)
        start_point = (int(box['xmin']), int(box['ymin']))
        end_point = (int(box['xmax']), int(box['ymax']))
        color = (255, 0, 0)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.imwrite('output.jpg', image)