# GAC-Net: Geometry-Anchored Cross-Modal Attention for Sparse Point Cloud Completion

Official code for the **Pattern Recognition** submission (Manuscript ID **PR-D-26-16422**).

## Paper

> GAC-Net: Geometry-Anchored Cross-Modal Attention for Sparse Point Cloud Completion from a Single RGB-D View

## Overview

GAC-Net is a cross-modal completion network built around **GACA (Geometry-Anchored Cross-Modal Attention)**, which injects the known camera projection into cross-modal attention as a soft, learnable Gaussian logit bias.

- Code: `image_guided_pc_completion/`
- Baselines: `tools/baselines/`
- Re-rendered PCN-8 protocol generator: `tools/render_shapenet.py`
- Mechanism probes: `tools/significance.py`, `tools/camera_utils.py`

## Directory layout

```
image_guided_pc_completion/
├── pcic/                 # GAC-Net model, losses, metrics, data
│   ├── models/           # network.py (GACA), fusion.py, decoder.py, pointnet2.py, image_encoder.py
│   ├── data/             # shapenet.py, synthetic.py (re-rendered PCN-8)
│   └── losses.py, metrics.py, utils.py
├── tools/
│   ├── camera_utils.py   # calibration handling, attention probe helpers
│   ├── render_shapenet.py# re-render PCN-8 from ShapeNetCore.v2 with one calibrated view
│   ├── significance.py   # paired bootstrap CI for CD-L1 / F@0.01
│   ├── verify_alignment.py
│   └── baselines/        # scripts to run PoinTr / SnowflakeNet / export PCN predictions
├── configs/              # default, ablation, smoke, point_only, ablation_no_anchor, ...
├── train.py              # 50-epoch cosine schedule
├── evaluate.py           # unified evaluator (CD-L1, CD-L2, F@0.01, F@0.02)
├── predict.py            # single-checkpoint prediction dump
├── smoke_test.py         # end-to-end smoke on chair category
└── requirements.txt
repro/                    # server-side pipeline scripts used in the paper
env_*.txt                 # conda explicit spec + pip freeze (two envs used)
snowflakenet_built.tar.gz # pre-built SnowflakeNet CUDA ext for Windows
```

## Environment

```bash
# recommended: conda + pip, see env_snowflake_conda_explicit.txt for pinned versions
pip install -r image_guided_pc_completion/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Tested with Python 3.10 / PyTorch 2.3 / CUDA 12.1.

## Data

The paper uses a **re-rendered PCN-8 protocol**: each ShapeNetCore.v2 model is rendered from a single calibrated camera (azimuth 0°, elevation 25° ± 5° jitter) producing one 2048-point partial cloud and one 16384-point ground-truth cloud.

```bash
python tools/render_shapenet.py --root /path/to/ShapeNetCore.v2 --out data/shapenet8
```

## Train / Test

```bash
# smoke test on chair category
python train.py --config configs/smoke_pcn8.json

# full 50-epoch run
python train.py --config configs/default.json
python evaluate.py --ckpt logs/default/best.pt --split test

# run PoinTr / SnowflakeNet / AnchorFormer through the same evaluator
bash tools/baselines/run_pointr.sh
bash tools/baselines/run_snowflakenet.sh
```

## Notes

- The view-consistency objective (λ = 0.05) and fine-tuning probe configs live in `configs/ablation_no_view.json` / `configs/ablation_concat.json`.
- Mechanism probes — attention centroid distance (px), projected-patch mass, image-shuffle CD-L1 — are reproduced by `tools/camera_utils.py` and `tools/significance.py`.
- This release excludes the full ShapeNet render dump (~200 GB) and checkpoints. Upon acceptance, we will host the full PCN-8 re-rendered dataset and pretrained weights on a persistent data share.

## Citation

```bibtex
@article{huang2026gacnet,
  title={GAC-Net: Geometry-Anchored Cross-Modal Attention for Sparse Point Cloud Completion from a Single RGB-D View},
  author={Huang, Wenquan and ...},
  journal={Pattern Recognition},
  year={2026}
}
```

## License

Non-commercial academic use only. Corresponding author: hwqwlsu@163.com.
