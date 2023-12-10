import  torch
import  cv2  
from detectx    import detectx
from plot_boxes import plot_boxes

def main(img_path=None, vid_path=None, vid_out=None):
    print(f"[INFO] Loading model... ")
    model = torch.hub.load('C:/Users/Med Hedi/Desktop/sem3/yolov5_deploy-main/yolov5', 'custom', source='local', path='last.pt', force_reload=True)
    classes = model.names  

    if img_path is not None:
        print(f"[INFO] Working with image: {img_path}")
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model=model)
        frame = plot_boxes(results, frame, classes=classes)

        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL)

        while True:
            cv2.imshow("img_only", frame)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # Touche Échap
                print("[INFO] Exiting. . . ")
                cv2.destroyAllWindows()  
                break  

    elif vid_path is not None:
        print(f"[INFO] Working with video: {vid_path}")
        cap = cv2.VideoCapture(vid_path)

        if vid_out:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        frame_no = 1

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if ret:
                print(f"[INFO] Working with frame {frame_no} ")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detectx(frame, model=model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame, classes=classes)
                cv2.imshow("vid_out", frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

            key = ((cv2.waitKey(5)) & (0xFF))
            if (key == 27):
                out.release()
                cv2.destroyAllWindows()
                break
                
            frame_no += 1

        print(f"[INFO] Cleaning up. . . ")
        out.release()
        cv2.destroyAllWindows()


