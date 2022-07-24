"""
    QR Code Detection Module based on Yolov5
    2022.7.24
    @vectorgeek
"""
from os.path import exists
import torch


class QRDetect:
    """
        QR Code Detection Class based on Yolov5
    """
    def __init__(self, model_path='qr-yolov5.pt'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.initialized = True
        print("Model initialized.")
    
    def detect(self, image_path):
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
        print(most_confident_box)
        return most_confident_box
