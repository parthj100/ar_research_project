"""
Run EGTEA Gaze+ official 3-split evaluation in a "from scratch" pipeline:
1) Train teacher (CLIP + head) on each split
2) Distill student from the trained teacher on each split
3) Aggregate best validation accuracy across splits (mean)

Default configuration is intentionally "quick" but meaningful:
- label_type=verb (19 classes)
- teacher=large (ViT-L/14)
- student=mobilenetv3_large
- max_train=2000, max_test=500
- teacher_epochs=2, student_epochs=3
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[str], log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        code = p.wait()
    if code != 0:
        raise RuntimeError(f"Command failed ({code}). See log: {log_path}")


def read_best_acc_from_log(log_path: Path) -> float:
    """
    Parse 'Best validation accuracy: XX.XX%' from DistillationTrainer output.
    """
    best = None
    with open(log_path, "r") as f:
        for line in f:
            if "Best validation accuracy:" in line:
                # line: Best validation accuracy: 40.20%
                try:
                    best = float(line.strip().split(":")[1].strip().replace("%", ""))
                except Exception:
                    pass
    if best is None:
        raise RuntimeError(f"Could not parse best accuracy from log: {log_path}")
    return best


def read_teacher_best_acc_from_log(log_path: Path) -> float:
    """
    Parse 'Teacher training complete. Best val acc: XX.XX%' from teacher script output.
    """
    best = None
    with open(log_path, "r") as f:
        for line in f:
            if "Teacher training complete. Best val acc:" in line:
                # line: Teacher training complete. Best val acc: 22.20%
                try:
                    best = float(line.strip().split(":")[-1].strip().replace("%", ""))
                except Exception:
                    pass
    if best is None:
        raise RuntimeError(f"Could not parse teacher best acc from log: {log_path}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-type", default="verb", choices=["verb", "action"])
    parser.add_argument("--teacher", default="large", choices=["base", "large"])
    parser.add_argument("--student", default="mobilenetv3_large",
                        choices=["mobilenetv3", "mobilenetv3_large", "efficientnet_b0", "efficientnet_b1", "mobilevit_xxs"])
    parser.add_argument("--teacher-epochs", type=int, default=2)
    parser.add_argument("--student-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument("--freeze-clip", action="store_true", default=True)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    py = str(root / ".venv" / "bin" / "python")
    if not Path(py).exists():
        py = "python"

    results_root = root / "results" / "egtea_3split_from_scratch"
    results_root.mkdir(parents=True, exist_ok=True)

    split_results = []
    for split_id in [1, 2, 3]:
        print(f"\n=== EGTEA Split {split_id} ===")
        split_dir = results_root / f"split{split_id}_{args.label_type}"
        split_dir.mkdir(parents=True, exist_ok=True)

        teacher_log = split_dir / "teacher.log"
        student_log = split_dir / "student.log"

        # 1) Train teacher
        teacher_cmd = [
            py, str(root / "scripts" / "train_teacher_egtea_gaze.py"),
            "--split-id", str(split_id),
            "--label-type", args.label_type,
            "--teacher", args.teacher,
            "--epochs", str(args.teacher_epochs),
            "--batch-size", str(args.batch_size),
            "--frames-per-clip", str(args.frames_per_clip),
            "--max-train", str(args.max_train),
            "--max-test", str(args.max_test),
            "--lr", "1e-4",
        ]
        if args.freeze_clip:
            teacher_cmd.append("--freeze-clip")

        run(teacher_cmd, teacher_log)

        # Teacher checkpoint path (known pattern)
        teacher_ckpt = root / "results" / "egtea_teacher" / f"split{split_id}_{args.label_type}_teacher-{args.teacher}_freezeclip-1" / "best_teacher.pt"
        if not teacher_ckpt.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")

        # 2) Distill student from trained teacher
        student_cmd = [
            py, str(root / "scripts" / "train_student_from_teacher_egtea_gaze.py"),
            "--teacher-ckpt", str(teacher_ckpt.relative_to(root)),
            "--split-id", str(split_id),
            "--label-type", args.label_type,
            "--teacher", args.teacher,
            "--student", args.student,
            "--epochs", str(args.student_epochs),
            "--batch-size", str(args.batch_size),
            "--frames-per-clip", str(args.frames_per_clip),
            "--max-train", str(args.max_train),
            "--max-test", str(args.max_test),
        ]
        run(student_cmd, student_log)

        teacher_best = read_teacher_best_acc_from_log(teacher_log)
        student_best = read_best_acc_from_log(student_log)

        split_results.append({
            "split": split_id,
            "teacher_best_val_acc": teacher_best,
            "student_best_val_acc": student_best,
            "teacher_ckpt": str(teacher_ckpt),
            "teacher_log": str(teacher_log),
            "student_log": str(student_log),
        })

        print(f"Split {split_id} teacher best: {teacher_best:.2f}% | student best: {student_best:.2f}%")

    teacher_mean = sum(r["teacher_best_val_acc"] for r in split_results) / 3.0
    student_mean = sum(r["student_best_val_acc"] for r in split_results) / 3.0

    summary = {
        "label_type": args.label_type,
        "teacher": args.teacher,
        "student": args.student,
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "max_train": args.max_train,
        "max_test": args.max_test,
        "split_results": split_results,
        "teacher_mean_best_val_acc": teacher_mean,
        "student_mean_best_val_acc": student_mean,
    }
    with open(results_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EGTEA 3-split summary ===")
    print(f"Teacher mean best val acc: {teacher_mean:.2f}%")
    print(f"Student mean best val acc: {student_mean:.2f}%")
    print(f"Saved summary: {results_root / 'summary.json'}")


if __name__ == "__main__":
    main()


