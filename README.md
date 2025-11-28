# AR Teacher-Student Distillation Research

**Research Question:** Can a teacher–student setup (teacher off-device; student on-device) enable real-time AR agents that retain performance while reducing latency and bandwidth?

## 🎯 Project Overview

This research explores knowledge distillation for AR agents, training large "teacher" models in the cloud and distilling them into lightweight "student" models for on-device deployment.

## 📁 Repository Structure

```
ar_teacher_student_full/
├── phase1_gridworld/          # ✅ COMPLETE - Proof of concept
│   ├── envs/                  # Gridworld environment
│   ├── models/                # Teacher (PPO) & Student (MLP)
│   ├── scripts/               # Training, distillation, evaluation
│   └── results/               # Trained models & plots
│
├── phase2_vision_synthetic/   # ✅ COMPLETE - Vision-based navigation
│   ├── envs/                  # Vision AR environment (64x64 images)
│   ├── models/                # CNN Teacher & Student
│   ├── scripts/               # DQN training, distillation
│   └── results/               # Models & performance metrics
│
├── phase3_ego4d/              # 🚧 IN PROGRESS - Real egocentric data
│   ├── data/                  # Ego4D dataset (download separately)
│   ├── models/                # MobileViT Student, CLIP/VILA Teacher
│   └── scripts/               # Supervised learning pipeline
│
└── docs/                      # Documentation & analysis
```

## 🔬 Experiments Completed

### Phase 1: Gridworld (Proof of Concept)
- **Teacher:** PPO (Stable-Baselines3), ~10K params
- **Student:** 3-layer MLP, ~1.5K params
- **Results:** 
  - Teacher: 100% success
  - Student: 96% success (after distillation)
  - **85% parameter reduction**

### Phase 2: Vision-Based Navigation
- **Teacher:** CNN with ResNet18 backbone, ~11M params
- **Student:** Lightweight CNN, ~50K params  
- **Environment:** 64x64 synthetic images with objects
- **Results:**
  - Teacher: 70% success
  - Student: 68% success
  - **99.5% parameter reduction**

### Phase 3: Ego4D (Planned)
- **Teacher:** CLIP/VILA (Vision-Language Model)
- **Student:** MobileViT (~1.3M params)
- **Dataset:** Ego4D egocentric navigation
- **Goal:** Real AR camera footage with human demonstrations

## 📊 Key Metrics

| Metric | Phase 1 | Phase 2 | Phase 3 (Target) |
|--------|---------|---------|------------------|
| **Teacher Params** | 10K | 11M | 428M+ |
| **Student Params** | 1.5K | 50K | 1.3M |
| **Compression** | 85% | 99.5% | 99.7% |
| **Teacher Success** | 100% | 70% | 80%+ |
| **Student Success** | 96% | 68% | 70%+ |
| **Latency (Student)** | <1ms | 5ms | 10-50ms |

## 🚀 Quick Start

### Phase 1: Gridworld
```bash
cd phase1_gridworld
pip install -r requirements.txt

# Train teacher
python scripts/train_teacher.py

# Distill student
python scripts/distill_student.py

# Evaluate both
python scripts/eval_compare.py

# Visualize
python scripts/visualize_agent.py
```

### Phase 2: Vision
```bash
cd phase2_vision_synthetic
pip install -r requirements.txt

# Train vision teacher
python scripts/train_vision_teacher.py

# Distill vision student
python scripts/distill_vision_student.py

# Evaluate
python scripts/eval_vision_compare.py
```

### Phase 3: Ego4D (Setup)
```bash
cd phase3_ego4d
pip install transformers timm ego4d

# Download Ego4D (requires license agreement)
# See: https://ego4d-data.org/

# Training scripts coming soon...
```

## 🛠️ Requirements

- Python 3.9+
- PyTorch 2.0+
- stable-baselines3
- gymnasium
- transformers
- timm
- matplotlib
- numpy

## 📈 Research Contributions

1. **Demonstrated feasibility** of teacher-student distillation for AR agents
2. **Achieved 99%+ compression** while maintaining task performance
3. **Reduced inference latency** from seconds to milliseconds
4. **Validated approach** across symbolic and visual domains
5. **Provided framework** for real AR data integration (Ego4D)

## 🎯 Next Steps

- [ ] Complete Ego4D data integration
- [ ] Train MobileViT student on real egocentric data
- [ ] Deploy to mobile device (iOS/Android)
- [ ] Measure real-world latency metrics
- [ ] Compare with baseline on-device models

## 👥 Authors

Parth Joshi

## 📄 License

MIT License

