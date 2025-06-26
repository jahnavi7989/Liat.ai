# Player Re-Identification in Sports Video using YOLOv11 + Deep SORT

This project solves a real-world **sports analytics** problem: **player re-identification**. The goal is to assign consistent player IDs to each player throughout a video — even when they leave and re-enter the frame.

We use a **custom-trained YOLOv11** model for detecting players and a **Deep SORT tracker** to re-identify players using visual and motion-based features.

---

##  Objective

> Identify and track players in a 15-second sports video.  
> Ensure that players who go out of the frame and come back retain their original ID.

---

##  Project Structure

| File/Folder | Description |
|-------------|-------------|
| `mod.pt` | YOLOv11 detection model (trained to detect players and ball) |
| `vedio.mp4` | Input test video |
| `player_detections.csv` | YOLO detection outputs (bounding boxes per frame) |
| `player_tracks.csv` | Tracking results: frame-wise IDs and coordinates |
| `player_reid_output.mp4` | Final output video with IDs and boxes |
| `Untitled7.ipynb` | Google Colab notebook with full pipeline |
| `/deep_sort` | Deep SORT tracking code |
| `ckpt.t7` | Deep SORT appearance model checkpoint |

---

##  Setup Instructions (Google Colab)

### 1. Upload Required Files

Upload the following to your Colab environment or Drive:
- `vedio.mp4`
- `mod.pt` (YOLO model)
- `ckpt.t7` (Deep SORT checkpoint)

---

### 2. Install Required Libraries

```bash
!pip install ultralytics opencv-python pandas numpy tqdm filterpy lap scikit-image

Clone and Set Up Deep SORT
!git clone https://github.com/ZQPei/deep_sort_pytorch.git
%cd deep_sort_pytorch
!cp -r deep_sort /content/deep_sort
%cd /content
!mkdir -p /content/deep_sort/deep/checkpoint
# Upload ckpt.t7 manually or use:
!mv ckpt.t7 /content/deep_sort/deep/checkpoint/ckpt.t7

### How to Run the Pipeline
Step 1: Player Detection using YOLO
from ultralytics import YOLO
model = YOLO("/content/drive/MyDrive/player-reid/mod.pt")

results = model(frame)  

import pandas as pd
df = pd.DataFrame(detections, columns=["frame", "x1", "y1", "x2", "y2", "confidence"])
df.to_csv("/content/player_detections.csv", index=False)

Step 2: Player Tracking with Deep SORT
from deep_sort.deep_sort.deep_sort import DeepSort
deepsort = DeepSort(model_path="/content/deep_sort/deep/checkpoint/ckpt.t7")
outputs = deepsort.update(bbox_xywh, confs, [0]*len(bbox_xywh), frame)
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

Save final outputs:
player_reid_output.mp4   
player_tracks.csv        

Output Samples
| File                     | Description                              |
| ------------------------ | ---------------------------------------- |
| `player_reid_output.mp4` | Output video with tracked player IDs     |
| `player_tracks.csv`      | Frame-by-frame coordinates and player ID |

