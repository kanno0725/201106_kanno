from darkflow.net.build import TFNet
import cv2
import numpy as np

options1 = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.1}
options2 = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}


tfnet = TFNet(options2)

# 動画の読み込み
cap = cv2.VideoCapture("/content/darkflow/sample_movie/demo.mp4")

# 危険区域の設定
danger_tl = (10,270)
danger_br = (950,540)

# アウトプットの準備
output_file = "/content/darkflow/sample_movie/demo_output.mp4"
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
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
class_names1 = ['person']
#
num_classes = len(class_names1)
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

def main():
    frame_no = 0
    while(True):

        # 動画ストリームからフレームを取得
        ret, frame = cap.read()
        if not ret:
          break
      
        result = tfnet.return_predict(frame)

        person_num = 0
        for item in result: 
            tlx = item['topleft']['x']
            tly = item['topleft']['y']
            brx = item['bottomright']['x']
            bry = item['bottomright']['y']
            label = item['label']
            conf = item['confidence']
            area = (brx-tlx)*(bry-tly)


            if label == 'person' and conf > 0.3 and danger_tl[0] < (tlx+brx)/2 < danger_br[0] and danger_tl[1] < bry < danger_br[1] and area < 10000:
                person_num += 1
                #
                #for i in class_names1:
                    #if label == i:
                        #
                        #class_num = class_names1.index(i)
                        #break

                #枠の作成
                frame = cv2.rectangle(frame, (tlx, tly), (brx, bry), (255,255,255), 2)

                #ラベルの作成
                text = label + " " + ('%.2f' % conf)
                frame = cv2.rectangle(frame, (tlx, tly - 15), (tlx + 100, tly + 5), (255,255,255), -1)
                frame = cv2.putText(frame, text, (tlx, tly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        person_num_text = 'Number of person in exclusion area:' + str(person_num)
        frame = cv2.putText(frame, str(person_num), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
        frame = cv2.rectangle(frame, danger_tl, danger_br, (0,0,255), 3)

        vw.write(frame)
        print('end = '+str(frame_no))
        frame_no += 1
        # 表示
        #cv2.imshow("Show FLAME Image", frame)

        # escを押したら終了。
        #k = cv2.waitKey(10);
        #if k == ord('q'):  break;

    vw.release()
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()