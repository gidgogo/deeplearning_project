# SIDL Dirty-Lens Restoration — Course Project

NAFNet 기반 **all-in-one** dirty-lens 영상 복원. 무료 Colab(T4) 5계정에서
2시간 청크로 끊어 학습하며, 다음 3요소를 통제 실험(ablation)으로 분석한다.

- **Frequency-domain loss** (FFT/SSIM)
- **Synthetic dirty augmentation**
- **Degradation-aware conditioning** (FiLM, type embedding)

## 실험 매트릭스
| # | Config | Loss | Aug | Deg-Cond |
|---|--------|------|-----|----------|
| A | Baseline | PSNRLoss | ✗ | ✗ |
| B | +Aug | PSNRLoss | ✓ | ✗ |
| C | +Freq | PSNR+FFT | ✗ | ✗ |
| D | +DegCond | PSNRLoss | ✗ | ✓ |
| E | All | PSNR+FFT | ✓ | ✓ |

- 학습: `finger+dust+water+scratch` 통합 (NAFNet-width32)
- 평가: **val set** 타입별 × easy/medium/hard (test는 GT 없음 → online 제출용)
- `mixed`/`clean`: 학습 제외, mixed는 held-out 일반화 테스트

## 구성
```
ext/sidl_multitask_dataset.py   # all-in-one 통합 dataset (basicsr/data 에 주입)
configs/SIDL_allinone_A_baseline.yml
COLAB_GUIDE.md                  # 2시간 청크 + 자동 resume 실행 가이드
```

## 사용
`COLAB_GUIDE.md` 참고. 노트북 1개를 5계정에 공유하고 `EXP` 변수만 A~E로 변경.
