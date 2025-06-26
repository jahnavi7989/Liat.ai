# 🏟️ Player Re-Identification in Sports Video using YOLOv11 + Deep SORT

This project solves a real-world **sports analytics** problem: **player re-identification**. The goal is to assign consistent player IDs to each player throughout a video — even when they leave and re-enter the frame.

We use a **custom-trained YOLOv11** model for detecting players and a **Deep SORT tracker** to re-identify players using visual and motion-based features.

---

## 🎯 Objective

> Identify and track players in a 15-second sports video.  
> Ensure that players who go out of the frame and come back retain their original ID.

---

## 📁 Project Structure

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

## ⚙️ Setup Instructions (Google Colab)

### 1. Upload Required Files

Upload the following to your Colab environment or Drive:
- `vedio.mp4`
- `mod.pt` (YOLO model)
- `ckpt.t7` (Deep SORT checkpoint)

---

### 2. Install Required Libraries

```bash
!pip install ultralytics opencv-python pandas numpy tqdm filterpy lap scikit-image
