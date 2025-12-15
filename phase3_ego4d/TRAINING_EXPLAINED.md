# How Training Works with Video Frames

## Overview

This document explains how the teacher-student distillation training process works with video frames, from data loading to model updates.

## Complete Training Pipeline

```
Video Files → Frame Extraction → DataLoader → Batch → Models → Loss → Backward → Update
```

Let's break this down step by step:

---

## 1. Data Loading Phase

### Video-based Training (New Approach)

**Step 1: Video File Selection**
```python
# From video_loader.py
video_files = ['video_001.mp4', 'video_002.mp4', ...]
labels = [0, 1, ...]  # Action class indices
```

**Step 2: Frame Extraction (On-the-fly)**
```python
# When __getitem__ is called:
def __getitem__(self, idx):
    video_path = self.video_paths[idx]  # e.g., 'video_001.mp4'
    label = self.labels[idx]  # e.g., 0 (walking)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # e.g., 1000 frames
    
    # Sample 8 frames uniformly or randomly
    if training:
        indices = random.sample(range(total_frames), 8)  # Random for augmentation
    else:
        indices = np.linspace(0, total_frames-1, 8)  # Uniform for validation
    
    # Extract and process frames
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()  # Read frame from video
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB
        frame = cv2.resize(frame, (224, 224))  # Resize to model input size
        frame = apply_augmentation(frame)  # Random flips, rotations, etc.
        frames.append(frame)
    
    # Convert to tensor: (8, 3, 224, 224) = (T, C, H, W)
    frames_tensor = torch.stack([to_tensor(f) for f in frames])
    return {'frames': frames_tensor, 'label': label}
```

**Result**: Each video clip becomes a tensor of shape `(T, C, H, W)` where:
- `T = 8` frames per clip
- `C = 3` RGB channels
- `H, W = 224, 224` image size

### Frame-based Training (Original Approach)

**Step 1: Pre-extracted Frames**
```python
# Frames already extracted and saved as images
frame_paths = [
    'frames/video_001_frame_00.jpg',
    'frames/video_001_frame_01.jpg',
    ...
    'frames/video_001_frame_07.jpg',
]
```

**Step 2: Load Frames**
```python
def __getitem__(self, idx):
    frame_paths = self.clips[idx]['frames']  # List of 8 frame paths
    label = self.clips[idx]['label']
    
    frames = []
    for path in frame_paths:
        frame = Image.open(path)  # Load pre-extracted frame
        frame = transform(frame)  # Resize, normalize, augment
        frames.append(frame)
    
    frames_tensor = torch.stack(frames)  # (8, 3, 224, 224)
    return {'frames': frames_tensor, 'label': label}
```

**Key Difference**: 
- **Video-based**: Extracts frames on-the-fly from video files
- **Frame-based**: Loads pre-extracted frame images

---

## 2. Batch Creation

### DataLoader Batching

```python
# DataLoader collects multiple samples into batches
batch = [
    {'frames': (8, 3, 224, 224), 'label': 0},  # Video 1: walking
    {'frames': (8, 3, 224, 224), 'label': 1},  # Video 2: turning
    {'frames': (8, 3, 224, 224), 'label': 0},  # Video 3: walking
    ...
]

# Collate function stacks them
def collate_fn(batch):
    frames = torch.stack([item['frames'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {'frames': frames, 'label': labels}

# Final batch shape: (B, T, C, H, W)
# B = batch_size (e.g., 8)
# T = frames_per_clip (e.g., 8)
# Result: (8, 8, 3, 224, 224)
```

**Batch Structure**:
```
Batch Shape: (B, T, C, H, W) = (8, 8, 3, 224, 224)
├── 8 videos (batch size)
│   ├── Each video has 8 frames
│   │   ├── Each frame is 3×224×224 (RGB image)
```

---

## 3. Forward Pass

### Teacher Model (CLIP)

```python
# Input: frames = (8, 8, 3, 224, 224) - batch of 8 videos, each with 8 frames
frames = batch['frames'].to(device)

# Teacher processes each frame independently
with torch.no_grad():  # Teacher is frozen (no gradients)
    teacher_outputs = []
    
    # Process each video in batch
    for video in frames:  # video shape: (8, 3, 224, 224)
        frame_features = []
        
        # Process each frame
        for frame in video:  # frame shape: (3, 224, 224)
            # CLIP Vision Encoder
            frame_embedding = clip_vision_encoder(frame)  # (512,)
            frame_features.append(frame_embedding)
        
        # Stack frame features: (8, 512)
        frame_features = torch.stack(frame_features)
        
        # Temporal aggregation (average pooling)
        video_embedding = frame_features.mean(dim=0)  # (512,)
        
        # Action classifier
        logits = action_classifier(video_embedding)  # (num_actions,)
        
        teacher_outputs.append({
            'logits': logits,
            'features': video_embedding,
            'frame_features': frame_features,
        })
    
    # Stack all videos
    teacher_logits = torch.stack([out['logits'] for out in teacher_outputs])  # (8, num_actions)
    teacher_features = torch.stack([out['features'] for out in teacher_outputs])  # (8, 512)
```

**Teacher Output**:
- `logits`: (B, num_actions) - Action predictions
- `features`: (B, 512) - Video-level embeddings
- `frame_features`: (B, T, 512) - Frame-level embeddings

### Student Model (MobileViT)

```python
# Input: frames = (8, 8, 3, 224, 224)
frames = batch['frames'].to(device)

# Student processes frames with temporal attention
student_outputs = []

for video in frames:  # video shape: (8, 3, 224, 224)
    frame_features = []
    
    # Process each frame with MobileViT
    for frame in video:  # frame shape: (3, 224, 224)
        # MobileViT backbone
        frame_embedding = mobilevit_backbone(frame)  # (320,)
        frame_features.append(frame_embedding)
    
    # Stack: (8, 320)
    frame_features = torch.stack(frame_features)
    
    # Project to match teacher dimension
    projected = projection_layer(frame_features)  # (8, 512)
    
    # Temporal Attention Pooling (learns which frames are important)
    attention_weights = temporal_attention(projected)  # (8,)
    video_embedding = (projected * attention_weights.unsqueeze(-1)).sum(dim=0)  # (512,)
    
    # Action classifier
    logits = action_classifier(video_embedding)  # (num_actions,)
    
    student_outputs.append({
        'logits': logits,
        'features': video_embedding,
        'pooled_features': video_embedding,  # Same as features for student
    })

# Stack all videos
student_logits = torch.stack([out['logits'] for out in student_outputs])  # (8, num_actions)
student_features = torch.stack([out['features'] for out in student_outputs])  # (8, 512)
```

**Student Output**:
- `logits`: (B, num_actions) - Action predictions
- `features`: (B, 512) - Video-level embeddings (projected to match teacher)
- Uses **temporal attention** to weight important frames

**Key Difference**:
- **Teacher**: Simple average pooling of frame features
- **Student**: Learned temporal attention to focus on important frames

---

## 4. Loss Computation

### Distillation Loss Components

```python
# Get labels
labels = batch['label'].to(device)  # (8,) - Ground truth action classes

# Compute three loss components
losses = {}

# 1. Feature Distillation Loss (α)
# Match student features to teacher features
teacher_feat_norm = F.normalize(teacher_features, dim=-1)  # (8, 512)
student_feat_norm = F.normalize(student_features, dim=-1)  # (8, 512)
losses['feature'] = MSE(student_feat_norm, teacher_feat_norm)
# Goal: Student learns same visual representations as teacher

# 2. Response Distillation Loss (β)
# Match soft predictions (knowledge transfer)
temperature = 3.0
teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)  # (8, num_actions)
student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)  # (8, num_actions)
losses['response'] = KL_divergence(student_log_soft, teacher_soft) * (temperature ** 2)
# Goal: Student learns teacher's "soft" knowledge (uncertainty, relationships)

# 3. Task Loss (γ)
# Standard classification loss
losses['task'] = CrossEntropy(student_logits, labels)
# Goal: Student learns to predict correct actions

# Combined Loss
total_loss = α * losses['feature'] + β * losses['response'] + γ * losses['task']
# Default: α=0.5, β=1.0, γ=0.1
```

**Loss Breakdown**:
```
Total Loss = 0.5 × Feature Loss + 1.0 × Response Loss + 0.1 × Task Loss
            ↓                    ↓                      ↓
    Match embeddings    Match predictions    Match ground truth
```

---

## 5. Backward Pass & Optimization

```python
# Zero gradients
optimizer.zero_grad()

# Backward pass (compute gradients)
total_loss.backward()

# Gradient clipping (prevent exploding gradients)
torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

# Update student model weights
optimizer.step()

# Update learning rate
scheduler.step()
```

**What Happens**:
1. Gradients flow backward through student model
2. Teacher model is frozen (no gradients)
3. Student weights are updated to minimize loss
4. Learning rate may decrease (scheduler)

---

## 6. Complete Training Loop

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # 1. Load batch
        frames = batch['frames'].to(device)  # (B, T, C, H, W)
        labels = batch['label'].to(device)    # (B,)
        
        # 2. Teacher forward (frozen)
        with torch.no_grad():
            teacher_out = teacher(frames, return_features=True)
        
        # 3. Student forward (trainable)
        student_out = student(frames, return_features=True)
        
        # 4. Compute loss
        losses = criterion(
            student_logits=student_out['logits'],
            student_features=student_out['features'],
            teacher_logits=teacher_out['logits'],
            teacher_features=teacher_out['features'],
            labels=labels,
        )
        
        # 5. Backward & update
        optimizer.zero_grad()
        losses['total'].backward()
        clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        
        # 6. Track metrics
        accuracy = (student_out['logits'].argmax(-1) == labels).float().mean()
```

---

## Data Flow Visualization

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO FILES                               │
│  video_001.mp4, video_002.mp4, ...                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              FRAME EXTRACTION (On-the-fly)                 │
│  • Open video file                                          │
│  • Sample 8 frames (random or uniform)                     │
│  • Resize to 224×224                                        │
│  • Apply augmentation (flips, rotations, color jitter)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    BATCH CREATION                           │
│  Shape: (B, T, C, H, W) = (8, 8, 3, 224, 224)            │
│  • 8 videos per batch                                        │
│  • 8 frames per video                                        │
│  • 3 RGB channels                                            │
│  • 224×224 image size                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌───────────────┐            ┌───────────────┐
│   TEACHER     │            │   STUDENT     │
│   (CLIP)      │            │  (MobileViT)  │
│               │            │               │
│ • Process     │            │ • Process     │
│   each frame  │            │   each frame  │
│ • Average     │            │ • Temporal    │
│   pooling     │            │   attention   │
│ • Classify    │            │ • Classify     │
│               │            │               │
│ Output:       │            │ Output:       │
│ • logits      │            │ • logits      │
│ • features    │            │ • features    │
└───────┬───────┘            └───────┬───────┘
        │                            │
        └──────────────┬─────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    LOSS COMPUTATION                          │
│  • Feature Loss: Match embeddings (α=0.5)                 │
│  • Response Loss: Match soft predictions (β=1.0)            │
│  • Task Loss: Match ground truth (γ=0.1)                    │
│  • Total = α×Feature + β×Response + γ×Task                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              BACKWARD PASS & OPTIMIZATION                    │
│  • Compute gradients (only for student)                    │
│  • Clip gradients (max_norm=1.0)                            │
│  • Update student weights                                   │
│  • Update learning rate                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### 1. **Temporal Processing**
- Videos have **temporal dimension** (8 frames)
- Models process all frames, then aggregate
- Student uses **attention** to focus on important frames

### 2. **Knowledge Distillation**
- Teacher provides "soft" knowledge (uncertainty, relationships)
- Student learns from both teacher and ground truth
- Three loss components balance different learning objectives

### 3. **Data Augmentation**
- **Training**: Random frame sampling, flips, rotations
- **Validation**: Uniform frame sampling, no augmentation
- Helps model generalize to unseen data

### 4. **Batch Processing**
- Multiple videos processed simultaneously
- Efficient GPU utilization
- Batch size limited by GPU memory

---

## 🔍 Example: Single Video Training Step

Let's trace one video through the pipeline:

```
1. Video File: "walking_video.mp4" (1000 frames, 30 FPS)

2. Frame Sampling (Training):
   - Randomly select: frames [45, 123, 234, 456, 567, 678, 789, 890]
   - Extract and resize each to 224×224
   - Apply random horizontal flip to some frames
   - Result: 8 frames, each (3, 224, 224)

3. Teacher Processing:
   - Frame 1 → CLIP → (512,) embedding
   - Frame 2 → CLIP → (512,) embedding
   - ...
   - Frame 8 → CLIP → (512,) embedding
   - Average all 8 embeddings → (512,) video embedding
   - Classify → (num_actions,) logits

4. Student Processing:
   - Frame 1 → MobileViT → (320,) → Project → (512,)
   - Frame 2 → MobileViT → (320,) → Project → (512,)
   - ...
   - Frame 8 → MobileViT → (320,) → Project → (512,)
   - Temporal Attention → Weighted sum → (512,) video embedding
   - Classify → (num_actions,) logits

5. Loss Computation:
   - Feature Loss: ||teacher_embedding - student_embedding||²
   - Response Loss: KL(teacher_soft, student_soft)
   - Task Loss: CrossEntropy(student_logits, label="walking")
   - Total = 0.5×Feature + 1.0×Response + 0.1×Task

6. Update:
   - Compute gradients
   - Update student weights
   - Student learns to match teacher while predicting correctly
```

---

