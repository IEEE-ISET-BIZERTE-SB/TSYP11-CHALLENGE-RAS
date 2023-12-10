import cv2

def plot_boxes(results, frame,classes):

    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")


    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55:
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape),
            int(row[1]*y_shape),
            int(row[2]*x_shape),
            int(row[3]*y_shape) 

            text_d = classes[int(labels[i])]

            emotion_colors =  {
                'angry':        (0, 0, 255),    #? Red,
                'disgusted':    (255, 0, 255),  #? Magenta
                'fearful':      (0, 255, 255),  #? Yellow
                'happy':        (0, 255, 0),    #? Green
                'neutral':      (255, 255, 0),  #? Cyan
                'sad':          (255, 0, 0),    #? Blue
                'surprised':    (255, 165, 0),  #? Orange
            }

            if text_d in emotion_colors:
                color = emotion_colors[text_d]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), color, -1)
                cv2.putText(frame, f'{text_d} {round (float(row[4]), 2)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

