from darkflow.net.build import TFNet
import cv2
import numpy as np

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

# 動画の読み込み
cap = cv2.VideoCapture("/content/darkflow/sample_movie/デモ.mp4")

# アウトプットの準備
output_file = "/content/darkflow/sample_movie/デモ_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
size = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
vw = cv2.VideoWriter(output_file, fourcc, fps,  size)

class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor']

num_classes = len(class_names)
class_colors = []
for i in range(0, num_classes):
    hue = 255*i/num_classes
    col = np.zeros((1,1,3)).astype("uint8")
    col[0][0][0] = hue
    col[0][0][1] = 128
    col[0][0][2] = 255
    cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
    col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
    class_colors.append(col)

ret, frame = cap.read()
print(len(frame))
#if len(frame)':
#  print('true')

print(ret)
count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(count)
print(size)

