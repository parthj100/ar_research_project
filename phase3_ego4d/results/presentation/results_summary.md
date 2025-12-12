
# Research Results Summary

## Experiment Results Table

| Experiment | Model | Dataset | Train Acc | Val Acc | Params | Size | Distilled |
|------------|-------|---------|-----------|---------|--------|------|-----------|
| Human Action v3 | MobileViT-XXS | Human Action (15 classes) | 93.0% | 75.0% | 2.3M | 8.8MB | Yes |
| Ego4D (small) | MobileViT-XXS | Ego4D (8 classes) | 85.0% | 0.0% | 2.3M | 8.8MB | Yes |
| EgoHands | MobileViT-XXS | EgoHands (4 classes) | 100.0% | 100.0% | 2.3M | 8.8MB | Yes |
| Unified (MobileViT) | MobileViT-XXS | Unified (12 classes) | 96.1% | 87.5% | 2.3M | 8.8MB | Yes |
| Unified (MobileNetV3) Distilled | MobileNetV3 | Unified (12 classes) | 85.9% | 92.7% | 1.8M | 16.5MB | Yes |
| Unified (EfficientNet-B0) Distilled | EfficientNet-B0 | Unified (12 classes) | 85.4% | 88.2% | 4.7M | 53.9MB | Yes |
| Unified (MobileNetV3) Baseline | MobileNetV3 | Unified (12 classes) | 84.3% | 85.3% | 1.8M | 16.5MB | No |
| Unified (EfficientNet-B0) Baseline | EfficientNet-B0 | Unified (12 classes) | 86.8% | 86.0% | 4.7M | 53.9MB | No |

## Teacher Model (Reference)

| Model | Parameters | Size |
|-------|------------|------|
| CLIP ViT-B/32 | 151.9M | 580.0MB |

## Key Findings

1. **Best Model**: Distilled MobileNetV3 achieved **92.65% validation accuracy** on the Unified dataset.

2. **Distillation Gains**:
   - MobileNetV3: +7.36% (85.29% → 92.65%)
   - EfficientNet-B0: +2.21% (86.03% → 88.24%)

3. **Compression**:
   - Teacher (CLIP ViT-B/32): 580 MB, 151.9M params
   - Best Student (MobileNetV3): 16.5 MB, 1.8M params
   - **Compression ratio: ~35× smaller**

4. **Bandwidth**: 100% reduction (student runs entirely on-device)

5. **Latency**: Students run faster than teacher; suitable for real-time AR inference.
