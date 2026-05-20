# Assignment 4 - Implement Simplified 3D Gaussian Splatting

### In this assignment, you will implement a simplified version of 3D Gaussian Splatting (3DGS) in pure PyTorch — a complete pipeline that reconstructs a 3D scene from multi-view images via differentiable rasterization of 3D Gaussians.

### Resources:
- [Paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf)
- [3DGS Official Implementation](https://github.com/graphdeco-inria/gaussian-splatting)
- [COLMAP — Structure-from-Motion](https://colmap.github.io/)
- [Teaching Slides](https://pan.ustc.edu.cn/share/index/66294554e01948acaf78)

---

### Background

3D Gaussian Splatting 将场景表示为一组带颜色和不透明度的 3D 高斯，通过将其投影到图像平面做 α-blending 实现可微体渲染。本作业将带你从零实现一个**简化版** 3DGS（不含 tile-based rasterizer 和 adaptive densification），完整体验 pipeline：相机参数恢复 → 3D 高斯参数化 → 投影 → α-blending。

### Data

```
data/
├── chair/images/   # 100 张 multi-view 渲染图像
└── lego/images/    # 100 张 multi-view 渲染图像
```

两个场景任选其一，下面以 `chair` 为例（你也可以用自己的多视角图像，放入 `<scene>/images/` 即可）。

---

## Task 1: Structure-from-Motion with COLMAP

使用 COLMAP 恢复相机内外参，并得到一组稀疏 3D 点作为 3DGS 的初始化：

```bash
python mvs_with_colmap.py --data_dir data/chair
```

将恢复的 3D 点重投影回各视角进行验证：

```bash
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

---

## Task 2: Simplified 3D Gaussian Splatting (主要部分)

观察 Task 1 的输出可以发现，COLMAP 恢复的 3D 点对于稠密渲染来说过于稀疏。我们将每个点扩展为一个 3D 高斯，使其覆盖周围空间。

### 2.1 3D Gaussian Initialization

参考 paper 公式 (6)：协方差矩阵由缩放矩阵 *S* 和旋转矩阵 *R* 构造。每个高斯需要以下可优化参数：

| 参数 | 说明 |
|------|------|
| Position μ | 初始化为 SfM 3D 点 |
| Rotation R | 用单位四元数参数化 |
| Scaling S | 3 维向量 |
| Opacity o | 标量 |
| Color c | RGB 三通道 |

[gaussian_model.py#L32](gaussian_model.py#L32) 已实现这些参数的初始化。

> **TODO**：在 [gaussian_model.py#L103](gaussian_model.py#L103) 中由四元数和缩放参数构造 **3D 协方差矩阵**。

### 2.2 Project 3D Gaussians to 2D

参考 paper 公式 (5)，将 3D 高斯投影到图像平面需要：

- 世界到相机的变换矩阵 *W*
- 投影变换的雅可比矩阵 *J*

投影后的 2D 协方差为 $\Sigma' = J W \Sigma W^T J^T$。

> **TODO**：在 [gaussian_renderer.py#L26](gaussian_renderer.py#L26) 中实现 3D → 2D 投影。

### 2.3 Compute 2D Gaussian Values

2D Gaussian 在像素 $\mathbf{x}$ 处的取值：

$$
f(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) = \frac{1}{2\pi\sqrt{|\boldsymbol{\Sigma}_i|}} \exp\left(P_{(\mathbf{x},i)}\right), \quad P_{(\mathbf{x},i)} = -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x} - \boldsymbol{\mu}_i)
$$

其中 **μᵢ** 与 **Σᵢ** 为投影后的 2D 高斯中心与协方差。

> **TODO**：在 [gaussian_renderer.py#L61](gaussian_renderer.py#L61) 中计算 Gaussian 取值。

### 2.4 Volume Rendering via α-blending

给定 *N* 个按深度排序的 2D 高斯，每个高斯在像素 $\mathbf{x}$ 处的 alpha 与透射率为：

$$
\alpha_{(\mathbf{x}, i)} = o_i \cdot f(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i), \qquad T_{(\mathbf{x}, i)} = \prod_{j<i} (1 - \alpha_{(\mathbf{x}, j)})
$$

最终像素颜色由各高斯按 α-blending 累加（paper 公式 1-3）。

> **TODO**：在 [gaussian_renderer.py#L83](gaussian_renderer.py#L83) 中实现最终渲染。

### Train your 3DGS

完成上述代码后，启动训练：

```bash
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints
```

### Render a Multi-view Video (Optional)

训练完成后，可用 [render_3dgs_mv.py](render_3dgs_mv.py) 沿一个绕场景中心的**水平圆轨迹**渲染一段连续视角视频，便于直观检查重建质量：

```bash
python render_3dgs_mv.py \
    --colmap_dir data/chair \
    --checkpoint data/chair/checkpoints/checkpoint_000060.pt \
    --num_frames 240 --fps 30
# 默认输出: <colmap_dir>/render_mv.mp4
```

up 轴由训练相机的 y 轴平均自动估计（NeRF 合成数据图像均为正放），orbit 半径与高度取训练相机的均值。

---

## Task 3: Compare with the Official 3DGS Implementation

本作业为纯 PyTorch 实现，训练速度与显存效率远不如官方实现，且未实现 adaptive Gaussian densification 等关键模块。请使用相同数据集运行 [官方 3DGS](https://github.com/graphdeco-inria/gaussian-splatting)，从**渲染质量、训练速度、显存占用**三方面进行对比，并在报告中讨论差异来源。

---

### Requirements:
- 请自行环境配置，推荐使用 [conda 环境](https://docs.anaconda.com/miniconda/)
- 代码框架已提供，按 TODO 标注完成核心实现
- 按照模板要求写 Markdown 版作业报告，包含 Task 1/2/3 的结果与分析
