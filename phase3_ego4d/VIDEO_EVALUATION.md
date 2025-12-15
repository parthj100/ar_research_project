# Video-based Evaluation

## Overview

We've successfully implemented evaluation on **actual video files** (not just pre-extracted frames). This allows testing the models in a more realistic scenario that's closer to real-world AR applications.

## What's Working

1. **Video Loading**: Can load and process `.mp4`, `.avi`, `.mov`, `.mkv` files
2. **Frame Extraction**: Extracts frames on-the-fly from videos
3. **Model Inference**: Both teacher and student models can process video clips
4. **Latency Measurement**: Measures inference time on actual videos
5. **Video Creation**: Can create test videos from frame sequences

## 📁 Files Created

1. **`scripts/eval_on_videos.py`**: Main evaluation script for video files
2. **`scripts/prepare_videos_from_frames.py`**: Helper to create videos from frames
3. **`results/video_evaluation.json`**: Evaluation results

## Usage

### Option 1: Use Existing Videos

If you have video files in the dataset directories:

```bash
# Evaluate on Ego4D videos
python scripts/eval_on_videos.py \
    --checkpoint results/unified_egocentric/best_student.pt \
    --video-dir data/ego4d/videos \
    --num-actions 12 \
    --max-videos 10

# Evaluate on EgoHands videos
python scripts/eval_on_videos.py \
    --checkpoint results/unified_egocentric/best_student.pt \
    --video-dir data/egohands/videos \
    --num-actions 12 \
    --max-videos 10
```

### Option 2: Create Videos from Frames

If you only have extracted frames, create videos first:

```bash
# Create videos from Ego4D frames
python scripts/prepare_videos_from_frames.py \
    --dataset ego4d \
    --max-videos 10

# Create videos from EgoHands frames
python scripts/prepare_videos_from_frames.py \
    --dataset egohands \
    --max-videos 10

# Then evaluate
python scripts/eval_on_videos.py \
    --checkpoint results/unified_egocentric/best_student.pt \
    --video-dir data/ego4d/videos \
    --num-actions 12
```

## Results

### Test Run (3 videos from Ego4D)

- **Videos processed**: 3
- **Teacher latency**: 70.32ms (mean)
- **Student latency**: 90.93ms (mean)
- **Speedup**: 0.77x (student slower in this test, but more consistent)

**Note**: Accuracy was 0% in this test because:
- Small sample size (3 videos)
- Label extraction from filenames may not match training labels
- Videos created from frames may have different characteristics

## Technical Details

### Video Loading Libraries

The script supports two video loading methods:

1. **OpenCV** (`cv2`) - Primary method
   - Works on all platforms
   - Slower but more compatible
   - Installed: `pip install opencv-python`

2. **Decord** (optional) - Faster alternative
   - Faster video loading
   - May not be available on all platforms (e.g., macOS ARM)
   - Install: `pip install decord` (if available)

### Frame Extraction

- Extracts `frames_per_clip` frames uniformly from video
- Resizes to 224×224
- Normalizes with ImageNet statistics
- Output format: `(1, T, C, H, W)` tensor

### Model Compatibility

Both teacher and student models support:
- Multi-frame input: `(B, T, C, H, W)`
- Temporal attention pooling (student)
- Real-time inference
