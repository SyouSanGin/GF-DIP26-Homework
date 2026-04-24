# Assignment 3 - Bundle Adjustment
本仓库为高凡(SA25001019) DIP HW3 Bundle Adjustment 作业代码仓

## Requirements

You can create the environment with the following command (conda is required!):

```bash
conda env create -f environment.yml
```

## Task 1: Bundle Adjustment via PyTorch

### Evaluation
1. Make sure your current working directory is `Assignments/03_BundleAdjustment`.
2. Run the following command:
```bash
python ./bundle_adjustment.py --data_dir data --output_dir output
```
3. After running, check the `output` directory. The expected structure is:
```
output/
├── ba_loss.png           # Loss curve
├── ba_params.pt          # optimized result ckpt
├── ba_scene.html         # visualization of cameras & points
└── points3d_recon.obj    # pointcloud of the face
```

### Results
