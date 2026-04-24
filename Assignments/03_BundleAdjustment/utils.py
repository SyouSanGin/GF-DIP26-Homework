import torch


def _hat(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(dim=-1)
    O = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([O, -z, y], dim=-1),
            torch.stack([z, O, -x], dim=-1),
            torch.stack([-y, x, O], dim=-1),
        ],
        dim=-2,
    )


def quat_normalize(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (..., 4), got {tuple(quat.shape)}")
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(eps)
    return quat / norm


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(quat)
    w, x, y, z = q.unbind(dim=-1)

    wx = w * x
    wy = w * y
    wz = w * z
    xx = x * x
    xy = x * y
    xz = x * z
    yy = y * y
    yz = y * z
    zz = z * z

    one = torch.ones_like(w)
    R00 = one - 2.0 * (yy + zz)
    R01 = 2.0 * (xy - wz)
    R02 = 2.0 * (xz + wy)
    R10 = 2.0 * (xy + wz)
    R11 = one - 2.0 * (xx + zz)
    R12 = 2.0 * (yz - wx)
    R20 = 2.0 * (xz - wy)
    R21 = 2.0 * (yz + wx)
    R22 = one - 2.0 * (xx + yy)

    return torch.stack(
        [
            torch.stack([R00, R01, R02], dim=-1),
            torch.stack([R10, R11, R12], dim=-1),
            torch.stack([R20, R21, R22], dim=-1),
        ],
        dim=-2,
    )


def matrix_to_quat(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected matrix shape (..., 3, 3), got {tuple(R.shape)}")

    R00 = R[..., 0, 0]
    R11 = R[..., 1, 1]
    R22 = R[..., 2, 2]

    qw = 0.5 * torch.sqrt((1.0 + R00 + R11 + R22).clamp_min(eps))
    qx = 0.5 * torch.sqrt((1.0 + R00 - R11 - R22).clamp_min(eps))
    qy = 0.5 * torch.sqrt((1.0 - R00 + R11 - R22).clamp_min(eps))
    qz = 0.5 * torch.sqrt((1.0 - R00 - R11 + R22).clamp_min(eps))

    qx = torch.copysign(qx, R[..., 2, 1] - R[..., 1, 2])
    qy = torch.copysign(qy, R[..., 0, 2] - R[..., 2, 0])
    qz = torch.copysign(qz, R[..., 1, 0] - R[..., 0, 1])

    quat = torch.stack([qw, qx, qy, qz], dim=-1)
    return quat_normalize(quat)


def rotvec_to_quat(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if rotvec.shape[-1] != 3:
        raise ValueError(f"Expected rotvec shape (..., 3), got {tuple(rotvec.shape)}")

    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * theta

    w = torch.cos(half)
    s = torch.sin(half) / theta.clamp_min(eps)
    xyz = s * rotvec

    # small-angle: sin(theta/2)/theta ~= 1/2 - theta^2/48
    theta2 = theta * theta
    s_small = 0.5 - theta2 / 48.0
    xyz_small = s_small * rotvec
    small = theta < 1e-4
    xyz = torch.where(small.expand_as(xyz), xyz_small, xyz)

    quat = torch.cat([w, xyz], dim=-1)
    return quat_normalize(quat)


def quat_to_rotvec(quat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q = quat_normalize(quat)
    w = q[..., :1]
    xyz = q[..., 1:]
    vnorm = torch.linalg.norm(xyz, dim=-1, keepdim=True)

    angle = 2.0 * torch.atan2(vnorm, w.clamp_min(eps))
    scale = angle / vnorm.clamp_min(eps)
    rotvec = scale * xyz

    # small-angle: rotvec ~= 2 * xyz
    small = vnorm < 1e-6
    rotvec_small = 2.0 * xyz
    return torch.where(small.expand_as(rotvec), rotvec_small, rotvec)


def rotvec_to_matrix(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return quat_to_matrix(rotvec_to_quat(rotvec, eps=eps))


def matrix_to_rotvec(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return quat_to_rotvec(matrix_to_quat(R, eps=eps), eps=eps)

