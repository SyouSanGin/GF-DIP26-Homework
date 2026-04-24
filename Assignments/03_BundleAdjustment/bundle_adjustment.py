import argparse
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from utils import rotvec_to_matrix
import matplotlib.pyplot as plt

def load_points2d(npz_path: str):
    data = np.load(npz_path)
    keys = sorted(list(data.keys()))
    views = [data[k] for k in keys]
    obs = np.stack(views, axis=0)
    return torch.from_numpy(obs).float(), keys

# 初始化焦距，基于图像高度和视场角计算
def init_focal(height: int, fov_deg: float):
    fov = math.radians(fov_deg)
    return height / (2.0 * math.tan(0.5 * fov))


def project_points(
    points: torch.Tensor,
    rotvecs: torch.Tensor,
    trans: torch.Tensor,
    focal: torch.Tensor,
    cx: float,
    cy: float,
    clamp_z = 1e-6,
):
    R = rotvec_to_matrix(rotvecs)  # (V, 3, 3)
    cam = torch.einsum("vij,nj->vni", R, points) + trans[:, None, :]
    z = cam[..., 2].clamp_max(-clamp_z) # no to negative z
    u = -focal * cam[..., 0] / z + cx
    v = focal * cam[..., 1] / z + cy
    return torch.stack([u, v], dim=-1)


def calc_rep_loss( # 投影误差
    points: torch.Tensor,
    rotvecs: torch.Tensor,
    trans: torch.Tensor,
    focal: torch.Tensor,
    obs_xy: torch.Tensor,
    vis_mask: torch.Tensor,
    cx: float,
    cy: float,
    chunk_points: int, # calc batch
):
    V, N = obs_xy.shape[:2]
    if chunk_points <= 0:
        chunk_points = N

    total = torch.zeros((), device=points.device)
    denom = torch.zeros((), device=points.device)
    for start in range(0, N, chunk_points):
        end = min(start + chunk_points, N)
        pts = points[start:end]
        pred = project_points(pts, rotvecs, trans, focal, cx, cy)
        obs_chunk = obs_xy[:, start:end]
        vis_chunk = vis_mask[:, start:end]
        per = F.smooth_l1_loss(pred, obs_chunk, reduction="none").sum(dim=-1)
        total = total + (per * vis_chunk).sum()
        denom = denom + vis_chunk.sum()
    return total / denom.clamp_min(1.0)


def save_obj(path: str, points: np.ndarray, colors: np.ndarray):
    if colors.max() > 1.0:
        colors = colors / 255.0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p, c in zip(points, colors):
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")


def save_all_via_plotly(
    path: str,
    points: np.ndarray,
    colors: np.ndarray,
    rotvecs: torch.Tensor,
    trans: torch.Tensor,
    max_points: int,
    cam_scale: float,
):
    pts = points
    cols = colors
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points, dtype=np.int64)
        pts = pts[idx]
        cols = cols[idx]

    if cols.max() > 1.0:
        cols = cols / 255.0

    R = rotvec_to_matrix(rotvecs).detach().cpu()  # (V, 3, 3)
    t = trans.detach().cpu()  # (V, 3)
    cam_centers = -torch.einsum("vij,vj->vi", R.transpose(1, 2), t).numpy()

    axis = np.eye(3, dtype=np.float32)
    axis_colors = ["red", "green", "blue"]
    line_traces = []
    for k in range(3):
        segments_x = []
        segments_y = []
        segments_z = []
        for i in range(cam_centers.shape[0]):
            R_i = R[i].numpy()
            dir_world = R_i.T @ axis[k]
            p0 = cam_centers[i]
            p1 = p0 + cam_scale * dir_world
            segments_x += [p0[0], p1[0], None]
            segments_y += [p0[1], p1[1], None]
            segments_z += [p0[2], p1[2], None]
        line_traces.append(
            go.Scatter3d(
                x=segments_x,
                y=segments_y,
                z=segments_z,
                mode="lines",
                line=dict(color=axis_colors[k], width=3),
                name=f"cam_{'xyz'[k]}",
                showlegend=False,
            )
        )

    point_trace = go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker=dict(size=2, color=cols, opacity=0.9),
        name="points",
    )

    cam_trace = go.Scatter3d(
        x=cam_centers[:, 0],
        y=cam_centers[:, 1],
        z=cam_centers[:, 2],
        mode="markers",
        marker=dict(size=4, color="black"),
        name="cameras",
    )

    fig = go.Figure(data=[point_trace, cam_trace] + line_traces)
    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Bundle Adjustment: Cameras + Point Cloud",
    )
    fig.write_html(path)


def parse_args():
    parser = argparse.ArgumentParser(description="BA")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--steps", type=int, default=2000) # 2000 epoches
    parser.add_argument("--lr_points", type=float, default=1e-2)
    parser.add_argument("--lr_pose", type=float, default=5e-3)
    parser.add_argument("--lr_focal", type=float, default=1e-3)
    parser.add_argument("--fov_deg", type=float, default=60.0)
    parser.add_argument("--init_depth", type=float, default=2.5)
    parser.add_argument("--chunk_points", type=int, default=5000)
    parser.add_argument("--vis_max_points", type=int, default=12000)
    parser.add_argument("--vis_cam_scale", type=float, default=0.2)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=114514)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    obs_path = os.path.join(args.data_dir, "points2d.npz")
    colors_path = os.path.join(args.data_dir, "points3d_colors.npy")

    obs, keys = load_points2d(obs_path)
    obs = obs.to(device)
    obs_xy = obs[..., :2]
    vis_mask = (obs[..., 2] > 0.5).float()

    V, N = obs_xy.shape[:2]
    # contant intrinsics
    H = 1024 
    W = 1024 
    cx = W / 2.0
    cy = H / 2.0
    # for opt
    points3d = (0.05 * torch.randn(N, 3, device=device)).requires_grad_(True)
    rotvecs = torch.zeros(V, 3, device=device, requires_grad=True)
    trans = torch.zeros(V, 3, device=device, requires_grad=True)
    trans.data[:, 2] = -abs(args.init_depth)

    f_init = init_focal(int(H), args.fov_deg)
    f_log = torch.tensor(math.log(f_init), device=device, requires_grad=True)

    optimizer = torch.optim.Adam(
        [
            {"params": [points3d], "lr": args.lr_points},
            {"params": [rotvecs, trans], "lr": args.lr_pose},
            {"params": [f_log], "lr": args.lr_focal},
        ]
    )

    loss_rec = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        focal = torch.exp(f_log)
        loss = calc_rep_loss(
            points3d,
            rotvecs,
            trans,
            focal,
            obs_xy,
            vis_mask,
            cx,
            cy,
            args.chunk_points,
        )
        reg = 1e-4 * (points3d.pow(2).mean() + trans.pow(2).mean()) # 整体正则化，避免消除尺度&刚体歧义
        total = loss + reg
        total.backward()
        torch.nn.utils.clip_grad_norm_([points3d, rotvecs, trans, f_log], max_norm=10.0) # stable optimization
        optimizer.step()

        loss_rec.append(float(loss.detach().cpu()))
        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step:04d} | loss {loss_rec[-1]:.6f} | f {float(focal.detach().cpu()):.2f}"
            )
    # save ckpts
    os.makedirs(args.output_dir, exist_ok=True)
    params_path = os.path.join(args.output_dir, "ba_params.pt")
    torch.save(
        {
            "points3d": points3d.detach().cpu(),
            "rotvecs": rotvecs.detach().cpu(),
            "trans": trans.detach().cpu(),
            "focal": float(torch.exp(f_log).detach().cpu()),
            "view_keys": keys,
        },
        params_path,
    )

    # save obj
    colors = np.load(colors_path)
    obj_path = os.path.join(args.output_dir, "points3d_recon.obj")
    save_obj(obj_path, points3d.detach().cpu().numpy(), colors)

    # save all
    html_path = os.path.join(args.output_dir, "ba_scene.html")
    save_all_via_plotly(
        html_path,
        points3d.detach().cpu().numpy(),
        colors,
        rotvecs,
        trans,
        args.vis_max_points,
        args.vis_cam_scale,
    )
    plt.figure(figsize=(6, 4))
    plt.plot(loss_rec)
    plt.xlabel("Step")
    plt.ylabel("Reprojection Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "ba_loss.png"))
    plt.close()





if __name__ == "__main__":
    main()
