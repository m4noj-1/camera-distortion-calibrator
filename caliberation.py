

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import json
from pathlib import Path

# ==================== CONFIGURATION ====================
IMAGE_PATH = "whoami.png"  # or "whoami.jpg"
OUTPUT_DIR = "calibration_results"

# ==================== NORMALIZED DISTORTION MODEL ====================
class BrownConradyNorm:
    """
    Brown-Conrady radial distortion performed in normalized coordinates:
      x_n = (x - cx) / fx
      y_n = (y - cy) / fy

    radial = 1 + k1 * r^2 + k2 * r^4 + k3 * r^6
    pixel-distorted coords returned (in pixels):
      x_d = fx * x_dn + cx
      y_d = fy * y_dn + cy
    """
    def __init__(self, k1=0.0, k2=0.0, k3=0.0, fx=1.0, fy=None, cx=0.0, cy=0.0):
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.k3 = float(k3)
        self.fx = float(fx)
        self.fy = float(fx) if fy is None else float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    def distort_pixels(self, pts):
        """pts: (N,2) pixel coordinates -> returns (N,2) distorted pixel coordinates"""
        pts = np.atleast_2d(pts).astype(np.float64)
        x = pts[:, 0]
        y = pts[:, 1]

        xn = (x - self.cx) / self.fx
        yn = (y - self.cy) / self.fy
        r2 = xn * xn + yn * yn
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6

        xdn = xn * radial
        ydn = yn * radial

        xd = xdn * self.fx + self.cx
        yd = ydn * self.fy + self.cy
        return np.column_stack([xd, yd])

    def undistort_pixels(self, pts, iterations=8):
        """Iteratively invert distortion. pts in pixel coordinates -> returns undistorted pixel coords"""
        pts = np.atleast_2d(pts).astype(np.float64)
        x = pts[:, 0]
        y = pts[:, 1]

        # initial guess (normalized)
        xu = (x - self.cx) / self.fx
        yu = (y - self.cy) / self.fy

        for _ in range(iterations):
            r2 = xu * xu + yu * yu
            r4 = r2 * r2
            r6 = r4 * r2
            radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
            # avoid divide by zero
            inv = 1.0 / np.maximum(radial, 1e-12)
            xu = ((x - self.cx) / self.fx) * inv
            yu = ((y - self.cy) / self.fy) * inv

        xd = xu * self.fx + self.cx
        yd = yu * self.fy + self.cy
        return np.column_stack([xd, yd])

    def set_from_vector(self, vec):
        # expects [k1,k2,k3,fx,cx,cy] or [k1,k2,k3,fx,fy,cx,cy]
        v = np.asarray(vec, dtype=np.float64)
        if v.size == 6:
            self.k1, self.k2, self.k3, self.fx, self.cx, self.cy = v
            self.fy = self.fx
        elif v.size == 7:
            self.k1, self.k2, self.k3, self.fx, self.fy, self.cx, self.cy = v
        else:
            raise ValueError("Unsupported parameter vector length")

    def to_vector(self):
        return np.array([self.k1, self.k2, self.k3, self.fx, self.cx, self.cy], dtype=np.float64)

# ==================== CORNER DETECTION ====================
def detect_grid_corners(image, max_corners=300):
    print("🔍 Step 1: Detecting corners...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # Use Harris and then cv2.goodFeaturesToTrack fallback if too few
    harris = cv2.cornerHarris(gray, blockSize=5, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    threshold = 0.01 * harris.max()
    corners = np.argwhere(harris > threshold)
    if corners.size == 0:
        # fallback to Shi-Tomasi
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=8)
        if pts is None:
            return np.zeros((0, 2), dtype=np.float32)
        corners = pts.reshape(-1, 2)
        print(f"⚠ Harris failed; fallback to Shi-Tomasi: {len(corners)} corners")
        return corners

    corners = np.column_stack([corners[:, 1], corners[:, 0]])  # (x,y)
    corners = non_max_suppression(corners, harris, radius=8)

    # Sub-pixel refinement (requires float32 Nx1x2)
    if len(corners) > 0:
        corners32 = corners.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners32.reshape(-1, 1, 2), (5, 5), (-1, -1), criteria)
        if corners_refined is not None:
            corners = corners_refined.reshape(-1, 2)

    if len(corners) > max_corners:
        # keep strongest via Harris response
        responses = [harris[int(c[1]), int(c[0])] for c in corners]
        idx = np.argsort(responses)[-max_corners:]
        corners = corners[idx]

    print(f"✅ Detected {len(corners)} corners")
    return corners

def non_max_suppression(corners, response_map, radius=8):
    if len(corners) == 0:
        return corners
    corners_with_response = []
    for corner in corners:
        x, y = int(round(corner[0])), int(round(corner[1]))
        if 0 <= y < response_map.shape[0] and 0 <= x < response_map.shape[1]:
            corners_with_response.append((corner, response_map[y, x]))
    corners_with_response.sort(key=lambda x: x[1], reverse=True)
    filtered = []
    for corner, _ in corners_with_response:
        too_close = False
        for existing in filtered:
            if np.linalg.norm(corner - existing) < radius:
                too_close = True
                break
        if not too_close:
            filtered.append(corner)
    return np.array(filtered, dtype=np.float32)

# ==================== GRID STRUCTURE FITTING ====================
def fit_grid_structure(corners, tolerance_horizontal=18, tolerance_vertical=18):
    print("📐 Step 2: Fitting grid structure...")
    if len(corners) == 0:
        return [], []
    corners = np.array(corners)
    corners_sorted_y = corners[corners[:, 1].argsort()]
    corners_sorted_x = corners[corners[:, 0].argsort()]

    horizontal_lines = []
    used = set()
    for i, corner in enumerate(corners_sorted_y):
        if i in used:
            continue
        line = [corner]
        used.add(i)
        for j in range(i + 1, len(corners_sorted_y)):
            if j in used:
                continue
            if abs(corners_sorted_y[j, 1] - corner[1]) < tolerance_horizontal:
                line.append(corners_sorted_y[j])
                used.add(j)
        if len(line) >= 3:
            line = sorted(line, key=lambda p: p[0])
            horizontal_lines.append(np.array(line))

    vertical_lines = []
    used = set()
    for i, corner in enumerate(corners_sorted_x):
        if i in used:
            continue
        line = [corner]
        used.add(i)
        for j in range(i + 1, len(corners_sorted_x)):
            if j in used:
                continue
            if abs(corners_sorted_x[j, 0] - corner[0]) < tolerance_vertical:
                line.append(corners_sorted_x[j])
                used.add(j)
        if len(line) >= 3:
            line = sorted(line, key=lambda p: p[1])
            vertical_lines.append(np.array(line))

    print(f"✅ Found {len(horizontal_lines)} horizontal lines, {len(vertical_lines)} vertical lines")
    return horizontal_lines, vertical_lines

# ==================== RANSAC OUTLIER REMOVAL ====================
def ransac_line_fit(points, iterations=200, threshold=5.0):
    if len(points) < 2:
        return points
    best_inliers = np.array([], dtype=np.float64).reshape(0,2)
    pts = np.array(points)
    for _ in range(iterations):
        idx = np.random.choice(len(pts), 2, replace=False)
        p1, p2 = pts[idx]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.hypot(dx, dy)
        if length < 1e-6:
            continue
        a = -dy / length
        b = dx / length
        c = -(a * p1[0] + b * p1[1])
        distances = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)
        inliers = pts[distances < threshold]
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers
    return best_inliers if best_inliers.shape[0] > 0 else pts

def ransac_grid_outlier_removal(horizontal_lines, vertical_lines):
    print("🎯 Step 3: RANSAC outlier removal...")
    robust_horizontal = [ransac_line_fit(line) for line in horizontal_lines]
    robust_vertical = [ransac_line_fit(line) for line in vertical_lines]
    robust_horizontal = [line for line in robust_horizontal if len(line) >= 3]
    robust_vertical = [line for line in robust_vertical if len(line) >= 3]
    total_points = sum(len(line) for line in robust_horizontal) + sum(len(line) for line in robust_vertical)
    print(f"✅ {total_points} inlier points after RANSAC")
    return robust_horizontal, robust_vertical

# ==================== COST FUNCTION (normalized) ====================
def straightness_residuals_norm(params, horizontal_lines, vertical_lines, image_shape):
    """
    params: vector [k1, k2, k3, fx, cx, cy]
    Straightness residuals measured in normalized coordinates (so they are numerically stable)
    """
    k1, k2, k3, fx, cx, cy = params
    fx = float(fx)
    fy = fx  # keep fx==fy for stability in single-image case

    distort = BrownConradyNorm(k1=k1, k2=k2, k3=k3, fx=fx, fy=fy, cx=cx, cy=cy)
    residuals = []

    # horizontal lines: after undistortion, points on a line should have roughly constant normalized y
    for line in horizontal_lines:
        if len(line) < 3:
            continue
        und = distort.undistort_pixels(line)  # pixels
        yn = (und[:, 1] - cy) / fy
        dev = yn - np.median(yn)
        residuals.extend(dev.tolist())

    # vertical lines: after undistortion, points on a line should have roughly constant normalized x
    for line in vertical_lines:
        if len(line) < 3:
            continue
        und = distort.undistort_pixels(line)
        xn = (und[:, 0] - cx) / fx
        dev = xn - np.median(xn)
        residuals.extend(dev.tolist())

    # small regularizers to keep k small (in normalized units)
    residuals.append(1e-4 * k1)
    residuals.append(1e-5 * k2)
    residuals.append(1e-6 * k3)

    return np.array(residuals, dtype=np.float64)

# ==================== OPTIMIZATION ====================
def optimize_distortion_parameters(horizontal_lines, vertical_lines, img_width, img_height):
    print("⚙ Step 4: Optimizing distortion parameters (TRF with robust loss)...")

    # initial guesses
    fx0 = 0.9 * max(img_width, img_height)
    cx0 = img_width / 2.0
    cy0 = img_height / 2.0
    p0 = np.array([0.0, 0.0, 0.0, fx0, cx0, cy0], dtype=np.float64)  # k1,k2,k3,fx,cx,cy

    # bounds
    lower = np.array([-0.5, -0.5, -1.0, 0.2 * max(img_width, img_height), 0.0, 0.0], dtype=np.float64)
    upper = np.array([ 0.5,  0.5,  1.0, 5.0 * max(img_width, img_height), img_width, img_height], dtype=np.float64)

    result = least_squares(
        straightness_residuals_norm,
        p0,
        args=(horizontal_lines, vertical_lines, (img_width, img_height)),
        method='trf',       # trust-region reflective, supports bounds
        loss='soft_l1',     # robust loss
        f_scale=1.0,
        bounds=(lower, upper),
        max_nfev=5000,
        verbose=2
    )

    optimized = result.x
    distortion = BrownConradyNorm()
    distortion.set_from_vector(optimized)
    print("✅ Optimization complete!")
    print(f"   k1 = {distortion.k1:.6e}")
    print(f"   k2 = {distortion.k2:.6e}")
    print(f"   k3 = {distortion.k3:.6e}")
    print(f"   fx = {distortion.fx:.2f}")
    print(f"   cx = {distortion.cx:.2f}")
    print(f"   cy = {distortion.cy:.2f}")

    return distortion, result

# ==================== IMAGE UNDISTORTION (using the normalized model) ====================
def undistort_image(image, distortion, dst_shape=None):
    print("🖼 Step 5: Undistorting image...")
    h, w = image.shape[:2]
    if dst_shape is None:
        dst_w, dst_h = w, h
    else:
        dst_w, dst_h = dst_shape

    # For every pixel in the undistorted target image, compute where to sample from source:
    # map_x,map_y = distortion.distort_pixels([u,v])  (mapping from undistorted->distorted)
    grid_x, grid_y = np.meshgrid(np.arange(dst_w), np.arange(dst_h))
    tgt_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # We want mapping: for undistorted pixel (u,v) -> find distorted source (xs,ys)
    src_points = distortion.distort_pixels(tgt_points)  # pix -> pix

    map_x = src_points[:, 0].reshape(dst_h, dst_w).astype(np.float32)
    map_y = src_points[:, 1].reshape(dst_h, dst_w).astype(np.float32)

    undistorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    print("✅ Image undistorted")
    return undistorted

# ==================== REPROJECTION ERROR ====================
def compute_reprojection_error(horizontal_lines, vertical_lines, distortion):
    print("📊 Step 6: Computing reprojection error...")
    all_points = []
    for line in horizontal_lines:
        all_points.extend(line)
    for line in vertical_lines:
        all_points.extend(line)
    if len(all_points) == 0:
        return 0.0, 0.0, []

    all_points = np.array(all_points, dtype=np.float64)

    # Undistort (pixels -> pixels)
    und = distortion.undistort_pixels(all_points)

    # Distort back
    reproj = distortion.distort_pixels(und)

    errs = np.linalg.norm(all_points - reproj, axis=1)
    mean_e = float(np.mean(errs))
    max_e = float(np.max(errs))
    rms = float(np.sqrt(np.mean(errs * errs)))

    print(f"✅ Mean reprojection error: {mean_e:.4f} pixels")
    print(f"   Max reprojection error: {max_e:.4f} pixels")
    print(f"   RMS error: {rms:.4f} pixels")
    return mean_e, max_e, errs

# ==================== VISUALIZATION ====================
def visualize_results(image, corners, horizontal_lines, vertical_lines, distortion, undistorted_image):
    print("📈 Step 7: Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Original image with detected corners
    ax = axes[0, 0]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if corners is not None and len(corners) > 0:
        ax.plot(corners[:, 0], corners[:, 1], 'g.', markersize=4, alpha=0.6)
    ax.set_title(f'Detected Corners ({0 if corners is None else len(corners)} points)', fontsize=14, fontweight='bold')
    ax.axis('off')

    # 2. Grid structure with RANSAC lines
    ax = axes[0, 1]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for line in horizontal_lines:
        ax.plot(line[:, 0], line[:, 1], 'r-', linewidth=2, alpha=0.7)
        ax.plot(line[:, 0], line[:, 1], 'ro', markersize=3)
    for line in vertical_lines:
        ax.plot(line[:, 0], line[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax.plot(line[:, 0], line[:, 1], 'bo', markersize=3)
    ax.plot(distortion.cx, distortion.cy, 'y*', markersize=16, markeredgecolor='black')
    ax.set_title(f'Grid Structure (H:{len(horizontal_lines)}, V:{len(vertical_lines)})', fontsize=14, fontweight='bold')
    ax.axis('off')

    # 3. Undistorted image
    ax = axes[1, 0]
    ax.imshow(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
    ax.set_title('Undistorted Image', fontsize=14, fontweight='bold')
    ax.axis('off')

    # 4. Distortion model overlay (plot distorted grid lines from undistorted space)
    ax = axes[1, 1]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    h, w = image.shape[:2]
    grid_x = np.linspace(w * 0.05, w * 0.95, 12)
    grid_y = np.linspace(h * 0.05, h * 0.95, 12)
    for y in grid_y:
        line_points = np.column_stack([grid_x, np.full_like(grid_x, y)])
        distorted_line = distortion.distort_pixels(line_points)
        ax.plot(distorted_line[:, 0], distorted_line[:, 1], 'r-', linewidth=1.2, alpha=0.7)
    for x in grid_x:
        line_points = np.column_stack([np.full_like(grid_y, x), grid_y])
        distorted_line = distortion.distort_pixels(line_points)
        ax.plot(distorted_line[:, 0], distorted_line[:, 1], 'b-', linewidth=1.2, alpha=0.7)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
    ax.set_title('Distortion Model Overlay', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = Path(OUTPUT_DIR) / "visualization.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_path}")
    plt.show()

# ==================== SAVE RESULTS ====================
def save_results(distortion, mean_error, max_error, errors, horizontal_lines, vertical_lines):
    print("💾 Saving results...")
    results = {
        "distortion_model": "Brown-Conrady (normalized)",
        "formula": "x_d = x_u * (1 + k1*r^2 + k2*r^4 + k3*r^6)  (applied in normalized coordinates)",
        "parameters": {
            "k1": float(distortion.k1),
            "k2": float(distortion.k2),
            "k3": float(distortion.k3),
            "fx": float(distortion.fx),
            "cx": float(distortion.cx),
            "cy": float(distortion.cy)
        },
        "reprojection_error": {
            "mean_pixels": float(mean_error),
            "max_pixels": float(max_error),
            "rms_pixels": float(np.sqrt(np.mean(np.array(errors)**2))) if len(errors) > 0 else 0.0
        },
        "grid_statistics": {
            "num_horizontal_lines": len(horizontal_lines),
            "num_vertical_lines": len(vertical_lines),
            "total_points": sum(len(line) for line in horizontal_lines) + sum(len(line) for line in vertical_lines)
        }
    }
    output_path = Path(OUTPUT_DIR) / "calibration_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {output_path}")
    return output_path

# ==================== MAIN PIPELINE ====================
def main():
    print("="*70)
    print("🎯 CAMERA RADIAL DISTORTION CALIBRATION PIPELINE (fixed)")
    print("="*70)
    print(f"📸 Processing image: {IMAGE_PATH}\n")

    # Load image
    path = Path(IMAGE_PATH)
    if not path.exists():
        alt = IMAGE_PATH.replace('.png', '.jpg') if '.png' in IMAGE_PATH else IMAGE_PATH.replace('.jpg', '.png')
        if Path(alt).exists():
            path = Path(alt)
            print(f"⚠ Using alternative image {alt}")
        else:
            print(f"❌ ERROR: image not found: {IMAGE_PATH}")
            return

    image = cv2.imread(str(path))
    if image is None:
        print("❌ ERROR: Could not load image")
        return
    h, w = image.shape[:2]
    print(f"✅ Image loaded: {w}x{h} pixels\n")

    # Step 1: detect corners
    corners = detect_grid_corners(image)
    print()
    # Step 2: fit grid
    horizontal_lines, vertical_lines = fit_grid_structure(corners)
    print()
    # Step 3: RANSAC outlier removal
    horizontal_lines, vertical_lines = ransac_grid_outlier_removal(horizontal_lines, vertical_lines)
    print()
    # Step 4: optimize
    distortion, opt_result = optimize_distortion_parameters(horizontal_lines, vertical_lines, w, h)
    print()
    # Step 5: undistort image
    undistorted_image = undistort_image(image, distortion, dst_shape=(w, h))
    undistorted_path = Path(OUTPUT_DIR) / "undistorted_image.png"
    undistorted_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(undistorted_path), undistorted_image)
    print(f"   Saved to {undistorted_path}\n")
    # Step 6: compute reprojection error
    mean_error, max_error, errors = compute_reprojection_error(horizontal_lines, vertical_lines, distortion)
    print()
    # Step 7: visualize
    visualize_results(image, corners, horizontal_lines, vertical_lines, distortion, undistorted_image)
    print()
    # save json
    save_results(distortion, mean_error, max_error, errors, horizontal_lines, vertical_lines)
    print()
    print("="*70)
    print("🎉 CALIBRATION COMPLETE!")
    print("="*70)
    print(f"📁 Results saved in '{OUTPUT_DIR}/' directory\n")

if __name__ == "__main__":
    main()