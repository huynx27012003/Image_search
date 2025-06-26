from contour import extract_main_contour_bounding_box, extract_hsv_feature_from_broken_edges
import cv2
import numpy as np
from skimage.feature import hog
from config import IMAGE_SIZE, HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, LBP_RADIUS, LBP_POINTS, DEFAULT_WEIGHTS
from numpy.linalg import norm

def get_bird_bbox_and_mask(image):
    """
    Tìm bounding box nhỏ nhất bao quanh đối tượng chính (chim) và mask vùng đó.
    Giả sử đối tượng có độ tương phản cao, dùng threshold Otsu.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image.shape[:2]
        return (0, 0, w, h), np.ones((h, w), dtype=np.uint8)
    main_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_cnt)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [main_cnt], -1, 255, cv2.FILLED)
    return (x, y, w, h), mask

def extract_hsv_grid(image, bbox, mask=None, grid_x=5, grid_y=5, h_bins=8, s_bins=8):
    """
    Trích xuất đặc trưng màu HSV từ ảnh chim theo quy trình 7 bước:
    1. Phát hiện biên
    2. Khép kín các đoạn biên bị hở
    3. Lấp đầy vùng bên trong đường bao
    4. Tạo mặt nạ ROI
    5. Lấy giá trị HSV trong ROI
    6. Tính histogram và chuẩn hóa
    7. Hợp thành vector đặc trưng 32 chiều
    """
    # Bước 1: Phát hiện biên
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Bước 2: Khép kín các đoạn biên
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    dilated_edges = cv2.dilate(closed_edges, kernel, iterations=1)

    # Bước 3: Lấp đầy vùng bên trong
    inv_edges = cv2.bitwise_not(dilated_edges)
    h_roi, w_roi = inv_edges.shape
    mask_ff = np.zeros((h_roi+2, w_roi+2), np.uint8)
    filled = inv_edges.copy()
    cv2.floodFill(filled, mask_ff, (0, 0), 255)
    filled = cv2.bitwise_not(filled)
    filled_region = cv2.bitwise_or(filled, dilated_edges)

    # Bước 4: Tạo mặt nạ ROI
    contours, _ = cv2.findContours(filled_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if mask is not None:
        roi_mask = mask[y:y+h, x:x+w]
    else:
        roi_mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(roi_mask, [largest_contour], 0, 255, cv2.FILLED)

    # Bước 5: Lấy giá trị HSV trong ROI
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Bước 6: Tính histogram H-S tổ hợp (2D histogram)
    h_bins_new, s_bins_new = 16, 8
    hs_hist = cv2.calcHist(
        [hsv_roi], [0, 1], roi_mask,
        [h_bins_new, s_bins_new],
        [0, 180, 0, 256]
    )

    # Chuẩn hóa theo tổng số pixel trong vùng ROI
    total_pixels = cv2.countNonZero(roi_mask)
    if total_pixels == 0:
        return np.zeros(h_bins_new * s_bins_new, dtype=np.float32)

    hs_hist /= total_pixels

    # Bước 7: Hợp thành vector đặc trưng 128 chiều
    feature_vector = hs_hist.flatten()
    return feature_vector.astype(np.float32)

def extract_lbp_feature(image, bbox, mask=None, P=8, R=1):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi, (128, 128))
    if len(roi_resized.shape) == 3:
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_resized
    height, width = gray.shape
    lbp = np.zeros((height, width), dtype=np.uint8)

    # Tạo ánh xạ ROR-LBP
    def rotate_right_min(value, bits=8):
        min_val = value
        for i in range(1, bits):
            rotated = ((value >> i) | (value << (bits - i))) & ((1 << bits) - 1)
            if rotated < min_val:
                min_val = rotated
        return min_val

    def generate_ror_lbp_mapping():
        mapping = {}
        labels = {}
        next_label = 0
        for i in range(256):
            min_code = rotate_right_min(i)
            if min_code not in labels:
                labels[min_code] = next_label
                next_label += 1
            mapping[i] = labels[min_code]
        return mapping, next_label

    ror_mapping, num_bins = generate_ror_lbp_mapping()

    for yy in range(R, height - R):
        for xx in range(R, width - R):
            center = gray[yy, xx]
            lbp_value = 0
            for p in range(P):
                angle = 2 * np.pi * p / P
                dx = int(round(R * np.cos(angle)))
                dy = int(round(-R * np.sin(angle)))
                neighbor = gray[yy + dy, xx + dx]
                if neighbor >= center:
                    lbp_value |= (1 << p)
            lbp[yy, xx] = ror_mapping[lbp_value]

    if mask is not None:
        mask_resized = cv2.resize(mask[y:y+h, x:x+w], (128, 128), interpolation=cv2.INTER_NEAREST)
        lbp_vals = lbp[mask_resized > 0]
    else:
        lbp_vals = lbp[R:height-R, R:width-R].ravel()
    lbp_hist, _ = np.histogram(lbp_vals, bins=np.arange(0, num_bins + 1), range=(0, num_bins))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist

def extract_silhouette_ratio(mask, bbox):
    """
    Trích xuất đặc trưng silhouette ratio.
    - mask: ảnh nhị phân vùng chim (255: chim, 0: nền)
    - bbox: (x, y, w, h) bounding box quanh chim
    Trả về: np.array([silhouette_ratio, background_ratio], dtype=np.float32)
    """
    x, y, w, h = bbox
    roi_mask = mask[y:y+h, x:x+w]
    S_silhouette = np.count_nonzero(roi_mask)
    S_bbox = w * h if w > 0 and h > 0 else 1
    silhouette_ratio = S_silhouette / S_bbox
    background_ratio = 1.0 - silhouette_ratio
    return np.array([silhouette_ratio, background_ratio], dtype=np.float32)

def extract_elongation(contour):
    """
    Trích xuất đặc trưng elongation (độ dẹt).
    - contour: contour lớn nhất của chim
    Trả về: np.array([v1, v2], dtype=np.float32)
    """
    if len(contour) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    
    pts = contour.reshape(-1, 2)
    # Tìm hai điểm xa nhất (trục chính)
    dists = np.sqrt(np.sum((pts[None, :, :] - pts[:, None, :]) ** 2, axis=2))
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    main_axis_len = dists[i, j]

    if main_axis_len == 0:
        return np.array([1.0, 0.0], dtype=np.float32)
    
    # Tính vector trục chính và trục phụ
    vec_main = pts[j] - pts[i]
    vec_main = vec_main / np.linalg.norm(vec_main)
    vec_perp = np.array([-vec_main[1], vec_main[0]])  # Vector vuông góc

    # Chiếu các điểm lên trục phụ
    proj = pts @ vec_perp
    max_proj = np.max(proj)
    min_proj = np.min(proj)

    # Chọn các điểm biên (gần max_proj và min_proj) với ngưỡng 1% phạm vi chiếu
    proj_range = max_proj - min_proj
    threshold = 0.01 * proj_range if proj_range > 0 else 1e-6
    max_points = pts[np.abs(proj - max_proj) < threshold]
    min_points = pts[np.abs(proj - min_proj) < threshold]

    # Tìm cặp điểm mà vector nối song song với vec_perp
    max_dist = 0.0
    cos_threshold = np.cos(np.pi / 18)  # Ngưỡng góc: 10° (cos(10°) ≈ 0.984)
    for p1 in max_points:
        for p2 in min_points:
            vec_pair = p2 - p1
            norm_pair = np.linalg.norm(vec_pair)
            if norm_pair > 0:
                # Tính cos của góc giữa vec_pair và vec_perp
                cos_angle = np.abs(np.dot(vec_pair / norm_pair, vec_perp))
                if cos_angle > cos_threshold:  # Góc nhỏ hơn 10° hoặc gần 180°
                    max_dist = max(max_dist, norm_pair)

    # Dự phòng nếu không tìm thấy cặp điểm
    minor_axis_len = max_dist if max_dist > 0 else proj_range

    # Đặc trưng chuẩn hóa
    total_len = main_axis_len + abs(minor_axis_len) + 1e-6
    v1 = main_axis_len / total_len
    v2 = abs(minor_axis_len) / total_len

    return np.array([v1, v2], dtype=np.float32)

def extract_features(image):
    """
    Trích xuất tất cả đặc trưng từ một ảnh.
    Trả về dictionary các vector đặc trưng đã chuẩn hóa L1 (histogram, silhouette, elongation).
    """
    # Tìm bounding box, mask và contour cho đối tượng chim
    bbox, mask = get_bird_bbox_and_mask(image)
    # Tìm contour lớn nhất
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea) if contours else np.array([])

    # Đặc trưng hình dạng
    silhouette_feature = extract_silhouette_ratio(mask, bbox)
    elongation_feature = extract_elongation(main_contour)
    # Đặc trưng màu HSV
    hsv_grid = extract_hsv_grid(image, bbox, mask=mask)
    # Đặc trưng LBP
    lbp_feature = extract_lbp_feature(image, bbox, mask=mask)

    # Trả về dictionary các đặc trưng
    return {
        'silhouette': silhouette_feature,
        'elongation': elongation_feature,
        'hsv_grid': hsv_grid,
        'lbp': lbp_feature
    }

def combine_features(features_dict, weights):
    """
    Kết hợp các đặc trưng đã được chuẩn hóa thành một vector duy nhất.
    Các đặc trưng được nối lại (concatenate) sau khi nhân với trọng số.
    Chuẩn hóa L2 cho vector tổng hợp.
    """
    combined = []
    # Đảm bảo tổng trọng số = 1
    total_weight = sum(weights.values())
    norm_weights = {k: w / total_weight for k, w in weights.items()} if total_weight > 0 else weights

    # Kết hợp các đặc trưng có trong dictionary
    for feature_name, weight in norm_weights.items():
        if feature_name in features_dict:
            weighted_feature = features_dict[feature_name] * weight
            combined.extend(weighted_feature)

    combined = np.array(combined)
    # L2-normalize
    combined /= (norm(combined) + 1e-6)
    return combined