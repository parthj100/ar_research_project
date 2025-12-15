# Phase 3: CLIP Teacher → MobileViT Student + Ego4D

**Real Egocentric Vision with Knowledge Distillation**

This phase implements teacher-student distillation using:
- **Teacher:** CLIP (ViT-B/32) - ~86M parameters
- **Student:** MobileViT-XXS - ~1.3M parameters  
- **Data:** Ego4D egocentric video (or synthetic for testing)

---

## Key Improvements over Phase 1-2

| Aspect | Phase 1-2 | Phase 3 |
|--------|-----------|---------|
| **Learning** | Reinforcement Learning | **Supervised Learning** |
| **Data** | Synthetic simulation | **Real egocentric video** |
| **Teacher** | Custom CNN | **Pre-trained CLIP** |
| **Student** | Simple CNN | **MobileViT (ViT+CNN hybrid)** |
| **Compression** | ~16x | **~66x** |

---

## Quick Start

### 1. Setup Environment

```bash
cd phase3_ego4d
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Test with Synthetic Data (No Ego4D Required)

```bash
# Train with synthetic data
python scripts/train_distill.py --use_synthetic --epochs 20 --num_samples 1000

# Evaluate
python scripts/eval_compare.py --num_samples 500
```

### 3. Use Real Ego4D Data (Optional)

```bash
# Step 1: Sign up at https://ego4d-data.org/ (free for research)

# Step 2: Install and authenticate Ego4D CLI
pip install ego4d
ego4d --auth

# Step 3: Download a subset of videos
python -c "from data.ego4d_subset import download_ego4d_subset; download_ego4d_subset('./data/ego4d', num_videos=10)"

# Step 4: Train with real data
python scripts/train_distill.py --data_dir ./data/ego4d --epochs 50
```

---

## 📁 Project Structure

```
phase3_ego4d/
├── data/
│   ├── __init__.py
│   └── ego4d_subset.py      # Data loading & Ego4D subset download
├── models/
│   ├── __init__.py
│   ├── clip_teacher.py      # CLIP-based teacher (~86M params)
│   └── mobilevit_student.py # MobileViT student (~1.3M params)
├── scripts/
│   ├── __init__.py
│   ├── train_distill.py     # Main training script
│   └── eval_compare.py      # Evaluation & comparison
├── results/                  # Checkpoints & plots
├── requirements.txt
└── README.md
```

---

## 🏗️ Architecture

### CLIP Teacher

```
CLIP Vision Encoder (ViT-B/32)
├── Patch Embedding: 3×224×224 → 196×768
├── 12 Transformer Blocks
├── Layer Norm
└── Projection: 768 → 512

+ Action Classification Head
├── Linear(512 → 512) + ReLU + Dropout
├── Linear(512 → 256) + ReLU + Dropout  
└── Linear(256 → num_actions)

Total: ~86M parameters (512-dim embeddings)
```

### MobileViT Student

```
MobileViT-XXS
├── Stem: Conv 3×3
├── Stage 1: MobileNetV2 blocks
├── Stage 2: MobileNetV2 + MobileViT block
├── Stage 3: MobileNetV2 + MobileViT block
├── Stage 4: MobileNetV2 + MobileViT block
└── Global Pool → 320-dim features

+ Feature Projection (to match CLIP)
├── Linear(320 → 512) + LayerNorm + GELU

+ Temporal Attention Pooling
├── Multi-head self-attention (4 heads)
└── CLS token aggregation

+ Action Head
├── Linear(512 → 256) + ReLU + Dropout
└── Linear(256 → num_actions)

Total: ~1.3M parameters
```

---

## Distillation Strategy

We use a **combined distillation loss**:

```
L_total = α × L_feature + β × L_response + γ × L_task

Where:
- L_feature: MSE between normalized student/teacher embeddings
- L_response: KL divergence on soft labels (temperature=4.0)
- L_task: Cross-entropy on hard labels

Default weights: α=1.0, β=1.0, γ=0.5
```

This combines:
1. **Feature distillation** - Student learns CLIP's rich visual representations
2. **Response distillation** - Student mimics teacher's output distribution
3. **Task supervision** - Direct learning signal from labels

---

## Expected Results

| Metric | CLIP Teacher | MobileViT Student |
|--------|--------------|-------------------|
| **Parameters** | ~86M | ~1.3M |
| **Size on Disk** | ~330 MB | ~5 MB |
| **Latency (CPU)** | ~200ms | ~15ms |
| **Latency (GPU)** | ~20ms | ~5ms |
| **Accuracy** | 70-80% | 60-75% |

**Key Achievements:**
-  **66x model compression**
- **10-13x latency reduction**
- **Deployable on mobile devices**
- **<15% accuracy drop** (with good distillation)

---

## 🔧 Training Options

### Basic Training
```bash
python scripts/train_distill.py \
    --use_synthetic \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4
```

### Full Options
```bash
python scripts/train_distill.py \
    --data_dir ./data/ego4d \
    --teacher_size base \
    --student_size xxs \
    --num_actions 10 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --alpha 1.0 \
    --beta 1.0 \
    --gamma 0.5 \
    --temperature 4.0 \
    --output_dir results
```

### Student Size Options
| Size | Params | Use Case |
|------|--------|----------|
| `tiny` | ~500K | Extreme compression |
| `xxs` | ~1.3M | **Recommended** |
| `xs` | ~2.3M | Better accuracy |
| `s` | ~5.6M | Highest accuracy |

---

## 📈 Evaluation

```bash
# Basic evaluation
python scripts/eval_compare.py

# With trained checkpoint
python scripts/eval_compare.py \
    --student_checkpoint results/best_student.pt \
    --num_samples 1000

# Full options
python scripts/eval_compare.py \
    --teacher_size base \
    --student_size xxs \
    --num_actions 10 \
    --batch_size 32 \
    --output_dir results
```

**Outputs:**
- `results/evaluation_results.json` - Metrics
- `results/comparison_plot.png` - Visual comparison

---

## 🔬 Technical Notes

### Why CLIP as Teacher?

1. **Rich visual representations** - Trained on 400M image-text pairs
2. **Zero-shot transfer** - Good features without task-specific training
3. **Well-studied** - Extensive research on CLIP distillation
4. **Open weights** - Freely available via OpenAI/HuggingFace

### Why MobileViT as Student?

1. **Hybrid architecture** - Combines CNN efficiency with ViT expressivity
2. **Mobile-optimized** - Designed for on-device deployment
3. **Good accuracy/size tradeoff** - State-of-the-art for mobile vision
4. **Pre-trained available** - ImageNet weights via timm

### Ego4D Subset Selection

We focus on clips relevant to AR navigation:
- **FHO (Forecasting Hands & Objects)** - Action anticipation
- **Moments** - Episodic memory queries
- **Narrations** - Timestamped activity descriptions

Download ~10-50 videos (not the full 3,670 hours!) for efficient testing.

---

## Troubleshooting

### "No CLIP backend found"
```bash
pip install open-clip-torch
# or
pip install transformers
```

### "timm not found"
```bash
pip install timm>=0.9.12
```

### "Ego4D CLI not found"
```bash
pip install ego4d
ego4d --auth  # Follow authentication prompts
```

### Training too slow?
- Use `--student_size tiny` for faster iteration
- Reduce `--num_samples` for synthetic data
- Use GPU/MPS if available

### Out of memory?
- Reduce `--batch_size`
- Use `--student_size tiny`
- Reduce `frames_per_clip` in data loader

---

## References

### Knowledge Distillation
- Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
- Romero et al. "FitNets: Hints for Thin Deep Nets" (2015)

### CLIP
- Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (2021)
- OpenCLIP: https://github.com/mlfoundations/open_clip

### MobileViT
- Mehta & Rastegari "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer" (2022)
- timm implementation: https://github.com/huggingface/pytorch-image-models
