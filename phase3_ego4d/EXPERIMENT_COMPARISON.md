# Experiment Comparison: Teacher-Student Distillation

## Overview

This document provides a comprehensive comparison of all distillation experiments conducted in Phase 3.

---

## Table 1: Experiment Summary

| Experiment | Dataset | Task Type | Classes | Train Samples | Val Samples | Epochs | Training Time |
|------------|---------|-----------|---------|---------------|-------------|--------|---------------|
| **Human Action v2** | Human Action Recognition | Static Image Classification | 15 | 10,080 | 2,520 | 15 | ~19 min |
| **Human Action v3** | Human Action Recognition | Static Image Classification | 15 | 10,080 | 2,520 | 20 | ~23 min |
| **Ego4D** | Ego4D Egocentric Video | Video Action Recognition | 8 | 80 | 20 | 30 | ~7 min |

---

## Table 2: Model Architecture

| Experiment | Teacher Model | Teacher Params | Student Model | Student Params | Compression Ratio |
|------------|---------------|---------------|---------------|----------------|-------------------|
| **Human Action v2** | CLIP ViT-B/32 | ~86M | MobileViT-XXS | ~2.3M | **37.4x** |
| **Human Action v3** | CLIP ViT-B/32 | ~86M | MobileViT-XXS | ~2.3M | **37.4x** |
| **Ego4D** | CLIP ViT-B/32 | ~86M | MobileViT-XXS | ~2.3M | **37.4x** |

**Note:** All experiments use the same model architecture. CLIP teacher is frozen (not fine-tuned).

---

## Table 3: Training Configuration

| Experiment | Learning Rate | Batch Size | α (Feature) | β (Response) | γ (Task) | Temperature | Optimizer |
|------------|---------------|------------|-------------|--------------|----------|-------------|-----------|
| **Human Action v2** | 3e-4 | 32 | 1.0 | 1.0 | 0.5 | 4.0 | AdamW |
| **Human Action v3** | 5e-5 | 32 | 0.5 | 1.0 | 1.0 | 3.0 | AdamW |
| **Ego4D** | 5e-5 | 16 | 0.5 | 1.0 | 1.0 | 3.0 | AdamW |

**Hyperparameter Changes:**
- **v2 → v3:** Lower LR (3e-4 → 5e-5), reduced feature weight (1.0 → 0.5), increased task weight (0.5 → 1.0), lower temperature (4.0 → 3.0)
- **v3 → Ego4D:** Same hyperparameters, smaller batch size (32 → 16) due to video frames

---

## Table 4: Training Performance

| Experiment | Final Train Loss | Final Train Acc | Best Val Loss | Best Val Acc | Final Val Acc | Overfitting Gap |
|------------|------------------|-----------------|---------------|--------------|---------------|-----------------|
| **Human Action v2** | 0.749 | 88.15% | 1.206 | **38.25%** | 35.91% | **52.24%** |
| **Human Action v3** | 1.090 | 93.00% | 1.502 | **75.28%** | 60.71% | **32.29%** |
| **Ego4D** | 1.345 | 85.00% | 2.041 | **35.00%** | 0.00% | **85.00%** |

**Key Observations:**
- **v3** achieved best validation accuracy (75.28%) with improved hyperparameters
- **Ego4D** shows severe overfitting due to very small dataset (80 train samples)
- **v2** had unstable validation performance (8-38% range)

---

## Table 5: Dataset Characteristics

| Experiment | Dataset Source | Data Type | Input Format | Frames/Clip | Image Size | Augmentation |
|------------|----------------|-----------|--------------|-------------|------------|--------------|
| **Human Action v2** | HuggingFace (Bingsu/Human_Action_Recognition) | Static Images | (B, 1, C, H, W) | 1 | 224×224 | Yes |
| **Human Action v3** | HuggingFace (Bingsu/Human_Action_Recognition) | Static Images | (B, 1, C, H, W) | 1 | 224×224 | Yes |
| **Ego4D** | Ego4D (pre-extracted frames) | Egocentric Video | (B, 8, C, H, W) | 8 | 224×224 | Yes |

**Action Classes:**

**Human Action (15 classes):**
- calling, clapping, cycling, dancing, drinking, eating, fighting, hugging, laughing, listening_to_music, running, sitting, sleeping, texting, using_laptop

**Ego4D (8 classes):**
- walking, turning, looking_around, manipulating, picking_up, putting_down, reaching, standing

---

## Table 6: Training Progress (Selected Epochs)

### Human Action v2
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 1 | 50.27% | 8.13% | 1.108 | 1.432 |
| 5 | 76.06% | 38.25% | 0.870 | 1.205 |
| 10 | 83.98% | 33.10% | 0.792 | 1.310 |
| 15 | 88.15% | 35.91% | 0.749 | 1.289 |

### Human Action v3
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 1 | 80.81% | 68.97% | 1.380 | 1.624 |
| 5 | 85.84% | 72.62% | 1.248 | 1.539 |
| 10 | 88.79% | 68.29% | 1.191 | 1.677 |
| 15 | 91.43% | 75.00% | 1.160 | 1.502 |
| 20 | 93.00% | 60.71% | 1.090 | 1.864 |

### Ego4D
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 1 | 10.00% | 35.00% | 2.147 | 2.041 |
| 10 | 42.50% | 5.00% | 1.857 | 2.150 |
| 20 | 72.50% | 0.00% | 1.604 | 2.209 |
| 30 | 85.00% | 0.00% | 1.345 | 2.629 |

---

## Table 7: Model Size & Deployment Metrics

| Experiment | Teacher Size (MB) | Student Size (MB) | Size Reduction | Model Format |
|------------|-------------------|-------------------|----------------|--------------|
| **Human Action v2** | ~330 MB | ~9 MB | **36.7x** | PyTorch (.pt) |
| **Human Action v3** | ~330 MB | ~9 MB | **36.7x** | PyTorch (.pt) |
| **Ego4D** | ~330 MB | ~27 MB | **12.2x** | PyTorch (.pt) |

**Note:** Ego4D checkpoint is larger (27MB) because it includes optimizer state in some checkpoints.

---

## Table 8: Research Validity Assessment

| Experiment | AR Relevance | Real-World Data | Egocentric | Temporal | Valid for Research? |
|------------|--------------|-----------------|------------|----------|---------------------|
| **Human Action v2** | ❌ Low | ✅ Yes | ❌ Third-person | ❌ Single frame | ⚠️ Partial |
| **Human Action v3** | ❌ Low | ✅ Yes | ❌ Third-person | ❌ Single frame | ⚠️ Partial |
| **Ego4D** | ✅ **High** | ✅ Yes | ✅ **First-person** | ✅ **Multi-frame** | ✅ **Yes** |

**Conclusion:** Ego4D experiment is the most valid for AR research question as it uses:
- Egocentric (first-person) perspective
- Real-world video data
- AR-relevant actions (walking, turning, object manipulation)
- Temporal sequences (8 frames per clip)

---

## Table 9: Key Findings

| Metric | Human Action v2 | Human Action v3 | Ego4D |
|--------|----------------|----------------|-------|
| **Best Val Accuracy** | 38.25% | **75.28%** | 35.00% |
| **Training Stability** | ⚠️ Unstable | ✅ Stable | ⚠️ Overfitting |
| **Hyperparameter Quality** | Baseline | ✅ Optimized | Same as v3 |
| **Dataset Size** | ✅ Large (12.6K) | ✅ Large (12.6K) | ❌ Small (100) |
| **Generalization** | ⚠️ Poor | ✅ Good | ❌ Poor (overfitting) |

---

## Table 10: Distillation Loss Components

### Human Action v3 (Best Performing)
| Epoch | Feature Loss | Response Loss | Task Loss | Total Loss |
|-------|--------------|--------------|-----------|------------|
| 1 | ~0.004 | ~0.45 | ~0.93 | 1.380 |
| 10 | ~0.004 | ~0.40 | ~0.79 | 1.191 |
| 20 | ~0.004 | ~0.45 | ~0.64 | 1.090 |

**Observations:**
- Feature loss stays very low (~0.004) - student learns to match teacher features well
- Response loss dominates (~0.4-0.45) - soft label matching is important
- Task loss decreases over time - hard label learning improves

---

## Recommendations

### For Best Results:
1. **Use Human Action v3 configuration** for similar static image tasks
2. **Collect more Ego4D data** (need 1000+ samples) to reduce overfitting
3. **Apply v3 hyperparameters** to Ego4D when dataset is larger

### For AR Research:
1. **Ego4D experiment is most relevant** despite overfitting
2. **Need larger dataset** - current 100 clips insufficient
3. **Consider data augmentation** strategies for small datasets
4. **Measure latency** on actual mobile devices for deployment metrics

---

## File Locations

| Experiment | Checkpoints | Training History | Visualizations |
|------------|-------------|------------------|----------------|
| **Human Action v2** | `results/human_action_v2/` | `training_history.json` | N/A |
| **Human Action v3** | `results/human_action_v3/` | `training_history.json` | `results/visualizations/` |
| **Ego4D** | `results/ego4d_distill/` | `training_history.json` | N/A |

---

## Next Steps

1. ✅ **Completed:** Three distillation experiments
2. ⬜ **Pending:** Latency measurements (teacher vs student)
3. ⬜ **Pending:** Mobile device deployment testing
4. ⬜ **Pending:** More Ego4D data collection
5. ⬜ **Pending:** Evaluation on separate test set

---

*Generated: November 29, 2024*

