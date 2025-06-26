#  Player Re-Identification in Sports Video using YOLOv11 + Deep SORT

This project tackles a real-world **sports analytics** problem — **player re-identification**. The objective is to assign consistent player IDs to each player throughout a short sports video, even when they leave and re-enter the frame.

We achieve this using a **custom-trained YOLOv11 model** for player detection and **Deep SORT** for tracking and re-identification using visual and motion-based cues.

---

##  Objective

> Track all players in a 15-second video.  
> Ensure players who exit and re-enter the frame retain the same ID.  

---

##  Project Structure

| File/Folder             | Description                                       |
|-------------------------|---------------------------------------------------|
| `mod.pt`                | YOLOv11 detection model (trained to detect players and ball) |
| `vedio.mp4`             | Input test video                                  |
| `player_detections.csv` | YOLO detection outputs (bounding boxes per frame) |
| `player_tracks.csv`     | Tracking results: frame-wise IDs and coordinates  |
| `player_reid_output.mp4`| Final output video with IDs and boxes             |
| `Untitled7.ipynb`       | Google Colab notebook with full pipeline          |
| `/deep_sort`            | Deep SORT tracking code                           |
| `ckpt.t7`               | Deep SORT appearance model checkpoint             |

---

## ⚙️ Setup Instructions (Google Colab)

### 1. Upload Required Files
Make sure the following files are uploaded to your Colab workspace or Google Drive:
- `vedio.mp4` (input video)
- `mod.pt` (YOLOv11 detection model)
- `ckpt.t7` (Deep SORT appearance model)

### 2. Install Required Libraries
Install dependencies including `ultralytics`, `opencv-python`, `pandas`, `tqdm`, `filterpy`, `lap`, `scikit-image`.

### 3. Clone and Set Up Deep SORT
Clone the Deep SORT repository, move necessary folders, and place the `ckpt.t7` file under `/content/deep_sort/deep/checkpoint/`.

---

##  How to Run the Pipeline

**STEP 1: Player Detection using YOLO**  
Detect players and save the bounding boxes for each frame to `player_detections.csv`.

**STEP 2: Player Tracking with Deep SORT**  
Use Deep SORT to assign consistent tracking IDs and draw bounding boxes and IDs across frames.

**Save final outputs:**
- `player_reid_output.mp4` (output video with IDs and boxes)
- `player_tracks.csv` (CSV with frame, player_id, and coordinates)

---

##  Output Samples

| File                     | Description                              |
|--------------------------|------------------------------------------|
| `player_reid_output.mp4` | Output video with tracked player IDs     |
| `player_tracks.csv`      | Frame-by-frame coordinates and player ID |

---

##  Notes

- Player IDs may not appear if detections are very small or confidence is low.
- This project uses visual features from Deep SORT for player identity consistency.
- A fully working solution was not mandatory — the primary aim is to demonstrate **consistent tracking** using YOLO and Deep SORT.


