# Teacher–Student Latency Mini (Gridworld)

A tiny proof-of-concept for your teacher–student research question using a 2D gridworld:
- Train a larger **teacher** (PPO).
- Distill to a smaller **student**.
- Compare latency and bandwidth across modes:
  1) Off-device teacher (simulated network),
  2) On-device student,
  3) Student + sparse teacher hints.

## Setup (macOS, Python 3.11)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

> Optional: PyTorch MPS on Apple Silicon is automatically used if available.

## Train Teacher
```bash
python scripts/train_teacher.py
```

## Distill Student
```bash
python scripts/distill_student.py
```

## Evaluate & Simulate Latency/Bandwidth
```bash
python scripts/eval_compare.py
```

You can tweak simulated network delay (`net_delay_ms`) and hint frequency (`k`) inside `eval_compare.py`.

## Project Layout
```text
envs/
  gridworld.py
models/
  student.py
scripts/
  train_teacher.py
  distill_student.py
  eval_compare.py
results/
  logs/
  plots/
requirements.txt
README.md
```