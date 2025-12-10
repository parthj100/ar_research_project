# Training Walkthrough: Step-by-Step Code Example

This is a simplified walkthrough showing exactly what happens during training with actual code.

## Complete Training Step

```python
# ============================================
# STEP 1: Load a batch of videos
# ============================================

# DataLoader returns a batch
batch = {
    'frames': torch.Tensor,  # Shape: (8, 8, 3, 224, 224)
    'label': torch.Tensor,   # Shape: (8,)
}

# Example batch:
# - 8 videos (batch_size=8)
# - Each video has 8 frames
# - Each frame is 224×224 RGB image
# - Labels: [0, 1, 0, 2, 1, 0, 2, 1] (action class indices)

frames = batch['frames'].to(device)  # Move to GPU/MPS
labels = batch['label'].to(device)

print(f"Batch shape: {frames.shape}")  # (8, 8, 3, 224, 224)
print(f"Labels: {labels}")             # [0, 1, 0, 2, 1, 0, 2, 1]


# ============================================
# STEP 2: Teacher Forward Pass (Frozen)
# ============================================

with torch.no_grad():  # No gradients needed for teacher
    teacher_output = teacher(frames, return_features=True)
    
    # What happens inside teacher.forward():
    # 
    # Input: frames = (8, 8, 3, 224, 224)
    # 
    # For each video in batch:
    #   For each frame in video:
    #     frame → CLIP Vision Encoder → (512,) embedding
    #   
    #   Stack frame embeddings: (8, 512)
    #   Average pool: (512,)  # Simple temporal aggregation
    #   Classify: (num_actions,) logits
    #
    # Output:
    teacher_logits = teacher_output['logits']      # (8, num_actions)
    teacher_features = teacher_output['features']  # (8, 512)
    
    print(f"Teacher logits shape: {teacher_logits.shape}")    # (8, 12)
    print(f"Teacher features shape: {teacher_features.shape}") # (8, 512)


# ============================================
# STEP 3: Student Forward Pass (Trainable)
# ============================================

student_output = student(frames, return_features=True)

# What happens inside student.forward():
#
# Input: frames = (8, 8, 3, 224, 224)
#
# For each video in batch:
#   For each frame in video:
#     frame → MobileViT Backbone → (320,) features
#   
#   Stack frame features: (8, 320)
#   Project to match teacher: (8, 512)
#   Temporal Attention:
#     - Learn which frames are important
#     - Weight frames by importance
#     - Weighted sum: (512,)
#   Classify: (num_actions,) logits
#
# Output:
student_logits = student_output['logits']      # (8, num_actions)
student_features = student_output['features']  # (8, 512)

print(f"Student logits shape: {student_logits.shape}")    # (8, 12)
print(f"Student features shape: {student_features.shape}") # (8, 512)


# ============================================
# STEP 4: Compute Loss
# ============================================

# Component 1: Feature Distillation Loss
# Goal: Student embeddings should match teacher embeddings
teacher_feat_norm = F.normalize(teacher_features, dim=-1)  # (8, 512)
student_feat_norm = F.normalize(student_features, dim=-1)  # (8, 512)
feature_loss = F.mse_loss(student_feat_norm, teacher_feat_norm)

print(f"Feature Loss: {feature_loss.item():.4f}")
# Example: 0.0038
# Meaning: Student embeddings are close to teacher embeddings


# Component 2: Response Distillation Loss
# Goal: Student predictions should match teacher's "soft" predictions
temperature = 3.0
teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)  # (8, 12)
student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)  # (8, 12)
response_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

print(f"Response Loss: {response_loss.item():.4f}")
# Example: 0.0260
# Meaning: Student predictions match teacher's uncertainty/confidence


# Component 3: Task Loss
# Goal: Student should predict correct action classes
task_loss = F.cross_entropy(student_logits, labels)

print(f"Task Loss: {task_loss.item():.4f}")
# Example: 1.3524
# Meaning: Student is learning to predict correct actions


# Combined Loss
alpha, beta, gamma = 0.5, 1.0, 0.1
total_loss = alpha * feature_loss + beta * response_loss + gamma * task_loss

print(f"Total Loss: {total_loss.item():.4f}")
# Example: 0.1631
# = 0.5 × 0.0038 + 1.0 × 0.0260 + 0.1 × 1.3524


# ============================================
# STEP 5: Backward Pass
# ============================================

# Zero previous gradients
optimizer.zero_grad()

# Compute gradients (only for student, teacher is frozen)
total_loss.backward()

# Clip gradients to prevent explosion
torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

# Update student weights
optimizer.step()

# Update learning rate
scheduler.step()


# ============================================
# STEP 6: Compute Metrics
# ============================================

# Accuracy
predictions = student_logits.argmax(dim=-1)  # (8,)
correct = (predictions == labels).sum().item()
accuracy = 100 * correct / len(labels)

print(f"Accuracy: {accuracy:.1f}% ({correct}/{len(labels)})")
# Example: 50.0% (4/8)
```

---

## Visual Representation of One Video

Let's trace a single video through the pipeline:

```
Video: "walking_video.mp4"
Label: 0 (walking)
```

### Frame Extraction

```
Video File (1000 frames)
    ↓
Sample 8 frames: [45, 123, 234, 456, 567, 678, 789, 890]
    ↓
Extract & Process:
  Frame 45  → Resize(224×224) → Augment → (3, 224, 224)
  Frame 123 → Resize(224×224) → Augment → (3, 224, 224)
  ...
  Frame 890 → Resize(224×224) → Augment → (3, 224, 224)
    ↓
Stack: (8, 3, 224, 224)
```

### Teacher Processing

```
Input: (8, 3, 224, 224) - 8 frames
    ↓
For each frame:
  Frame 1 → CLIP ViT-B/32 → (512,) embedding
  Frame 2 → CLIP ViT-B/32 → (512,) embedding
  ...
  Frame 8 → CLIP ViT-B/32 → (512,) embedding
    ↓
Stack: (8, 512) - 8 frame embeddings
    ↓
Average Pooling: (512,) - video-level embedding
    ↓
Action Classifier: (12,) - logits for 12 action classes
    ↓
Output:
  logits: [2.1, 0.3, 0.8, ..., 0.1]  # 12 values
  features: (512,) embedding
```

### Student Processing

```
Input: (8, 3, 224, 224) - 8 frames
    ↓
For each frame:
  Frame 1 → MobileViT-XXS → (320,) features
  Frame 2 → MobileViT-XXS → (320,) features
  ...
  Frame 8 → MobileViT-XXS → (320,) features
    ↓
Stack: (8, 320) - 8 frame features
    ↓
Project to match teacher: (8, 512)
    ↓
Temporal Attention:
  - Compute attention weights: [0.15, 0.12, 0.18, 0.10, 0.14, 0.11, 0.13, 0.07]
  - Weighted sum: (512,) - video-level embedding
    ↓
Action Classifier: (12,) - logits for 12 action classes
    ↓
Output:
  logits: [1.8, 0.5, 0.6, ..., 0.2]  # 12 values (different from teacher)
  features: (512,) embedding (should match teacher)
```

### Loss Computation

```
Teacher Output:
  logits: [2.1, 0.3, 0.8, ..., 0.1]
  features: (512,) embedding

Student Output:
  logits: [1.8, 0.5, 0.6, ..., 0.2]
  features: (512,) embedding

Ground Truth:
  label: 0 (walking)

Losses:
  1. Feature Loss: ||teacher_features - student_features||²
     = ||(512,) - (512,)||²
     = 0.0038
     
  2. Response Loss: KL(softmax(student_logits/T), softmax(teacher_logits/T))
     = KL divergence between soft predictions
     = 0.0260
     
  3. Task Loss: CrossEntropy(student_logits, label=0)
     = -log(P(walking | student_logits))
     = 1.3524

Total Loss = 0.5×0.0038 + 1.0×0.0260 + 0.1×1.3524 = 0.1631
```

### Weight Update

```
Gradients flow backward:
  total_loss.backward()
    ↓
Compute gradients for student parameters:
  - MobileViT backbone weights
  - Projection layer weights
  - Temporal attention weights
  - Classifier weights
    ↓
Clip gradients (max_norm=1.0)
    ↓
Update weights:
  optimizer.step()
    ↓
Student model is now slightly better at:
  - Matching teacher embeddings (Feature Loss)
  - Matching teacher predictions (Response Loss)
  - Predicting correct actions (Task Loss)
```

---

## Key Insights

### 1. **Why Three Losses?**

- **Feature Loss**: Ensures student learns same visual representations
- **Response Loss**: Transfers teacher's "soft knowledge" (uncertainty, relationships)
- **Task Loss**: Ensures student can actually classify correctly

### 2. **Why Temporal Attention?**

- Not all frames are equally important
- Some frames show the action clearly, others are transitions
- Attention learns to focus on informative frames
- More efficient than simple averaging

### 3. **Why Teacher is Frozen?**

- Teacher (CLIP) is already well-trained
- We only want to transfer knowledge, not update teacher
- Saves computation and memory
- Teacher provides stable "ground truth" for student to learn from

### 4. **How Does Student Learn?**

- Student sees both:
  - **Teacher's knowledge**: "This looks like walking (confidence: 0.8)"
  - **Ground truth**: "This IS walking (label: 0)"
- Student learns to match teacher while being correct
- Over time, student becomes both accurate and knowledgeable

---

## Training Progress Example

```
Epoch 1:
  Feature Loss: 0.0038  → Student embeddings don't match teacher yet
  Response Loss: 0.0260 → Student predictions don't match teacher
  Task Loss: 1.3524     → Student can't predict correctly
  Accuracy: 0.0%        → Random predictions

Epoch 10:
  Feature Loss: 0.0021  → Getting closer to teacher embeddings
  Response Loss: 0.0123 → Predictions more similar to teacher
  Task Loss: 0.8543     → Better at predicting actions
  Accuracy: 45.0%       → Getting better

Epoch 30:
  Feature Loss: 0.0015  → Very close to teacher embeddings
  Response Loss: 0.0074 → Predictions match teacher well
  Task Loss: 0.5234     → Good at predicting actions
  Accuracy: 87.5%       → High accuracy!
```

The student learns to be both:
- **Knowledgeable** (matches teacher's understanding)
- **Accurate** (predicts correct actions)

