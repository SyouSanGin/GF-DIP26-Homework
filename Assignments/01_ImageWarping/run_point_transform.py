import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    warped_image = np.array(image)
    ### FILL: Implement MLS or RBF based image warping

    if image is None:
        return None

    img = np.array(image)
    h, w = img.shape[:2]

    if source_pts is None or target_pts is None:
        return warped_image
    if len(source_pts) == 0 or len(target_pts) == 0:
        return warped_image

    num_pairs = min(len(source_pts), len(target_pts))
    source_pts = np.asarray(source_pts[:num_pairs], dtype=np.float32)
    target_pts = np.asarray(target_pts[:num_pairs], dtype=np.float32)

    if num_pairs == 1:
        offset = source_pts[0] - target_pts[0]
        map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = map_x + offset[0]
        map_y = map_y + offset[1]
        warped_image = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return warped_image

    # Inverse warping: compute source coordinate for every pixel in target image.
    # Given user pairs source -> target, inverse mapping uses target as p and source as q.
    p = target_pts
    q = source_pts

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    identity = np.eye(2, dtype=np.float32)
    block_rows = 64
    x_coords = np.arange(w, dtype=np.float32)

    for y0 in range(0, h, block_rows):
        y1 = min(y0 + block_rows, h)
        y_coords = np.arange(y0, y1, dtype=np.float32)

        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        v = np.stack([grid_x, grid_y], axis=-1)  # (bh, w, 2)

        # (bh, w, n, 2)
        diff = p[None, None, :, :] - v[:, :, None, :]
        dist2 = np.sum(diff * diff, axis=-1)  # (bh, w, n)
        weights = 1.0 / (np.power(dist2, alpha) + eps)

        w_sum = np.sum(weights, axis=-1, keepdims=True)
        w_sum = np.maximum(w_sum, eps)

        p_star = np.sum(weights[:, :, :, None] * p[None, None, :, :], axis=2) / w_sum
        q_star = np.sum(weights[:, :, :, None] * q[None, None, :, :], axis=2) / w_sum

        p_hat = p[None, None, :, :] - p_star[:, :, None, :]
        q_hat = q[None, None, :, :] - q_star[:, :, None, :]

        weighted_p_hat = weights[:, :, :, None] * p_hat

        # m = Σ_i w_i * p_hat_i * p_hat_i^T, c = Σ_i w_i * q_hat_i * p_hat_i^T
        m = np.einsum("...ni,...nj->...ij", p_hat, weighted_p_hat)
        c = np.einsum("...ni,...nj->...ij", q_hat, weighted_p_hat)

        m_reg = m + eps * identity[None, None, :, :]
        inv_m = np.linalg.inv(m_reg)
        a = c @ inv_m

        src = (a @ (v - p_star)[..., None])[..., 0] + q_star

        map_x[y0:y1, :] = src[..., 0]
        map_y[y0:y1, :] = src[..., 1]

    warped_image = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
