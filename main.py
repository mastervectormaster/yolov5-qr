"""
    QR Code Detection test based on Yolov5
    2022.7.24
    @vectorgeek
"""

from qrdetect import QRDetect

IMAGE_PATH = 'images/2.jpg'
MODEL_PATH = 'models/qr-yolov5.pt'

if __name__ == "__main__":
    qrdetect = QRDetect(MODEL_PATH)
    box = qrdetect.detect(IMAGE_PATH, True)
    print(box)