import cv2
import numpy as np

def extract_main_contour_bounding_box(image, resize_width=512):
    h, w = image.shape[:2]
    ratio = resize_width / float(w)
    dim = (resize_width, int(h * ratio))
    image = cv2.resize(image, dim)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)

    result_img = image.copy()
    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    roi = image[y:y + h, x:x + w]
    return roi, (x, y, w, h), result_img



def extract_hsv_feature_from_broken_edges(image, resize_width=512, show=True):
    h, w = image.shape[:2]
    ratio = resize_width / float(w)
    image_resized = cv2.resize(image, (resize_width, int(h * ratio)))

    gray    = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    inv = cv2.bitwise_not(closed)
    h2, w2 = inv.shape[:2]
    mask_ff = np.zeros((h2+2, w2+2), np.uint8)
    cv2.floodFill(inv, mask_ff, (0, 0), 255)
    inv_filled = cv2.bitwise_not(inv)
    filled = closed | inv_filled

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Không tìm thấy contour nào sau khi fill holes.")
    largest = max(contours, key=cv2.contourArea)

    return largest, ratio, image_resized
    


def extract_hsv_feature_from_broken_edges1(image, resize_width=512, show=True):
    h, w = image.shape[:2]
    ratio = resize_width / float(w)
    image_resized = cv2.resize(image, (resize_width, int(h * ratio)))

    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Fill vùng kín
    inv = cv2.bitwise_not(closed)
    h2, w2 = inv.shape[:2]
    mask_ff = np.zeros((h2+2, w2+2), np.uint8)
    cv2.floodFill(inv, mask_ff, (0, 0), 255)
    inv_filled = cv2.bitwise_not(inv)
    filled = closed | inv_filled

    # Tìm contour lớn nhất
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Không tìm thấy contour nào sau khi fill holes.")
    largest = max(contours, key=cv2.contourArea)

    if show:
        output = image_resized.copy()
        max_dist = 0
        pt1_main, pt2_main = None, None
        for i in range(len(largest)):
            for j in range(i + 1, len(largest)):
                p1 = largest[i][0]
                p2 = largest[j][0]
                dist = np.linalg.norm(p1 - p2)
                if dist > max_dist:
                    max_dist = dist
                    pt1_main, pt2_main = tuple(p1), tuple(p2)


        pt_2_3 = (
            int(pt1_main[0] * (1/3) + pt2_main[0] * (2/3)),
            int(pt1_main[1] * (1/3) + pt2_main[1] * (2/3)),
        )
        dx = pt2_main[0] - pt1_main[0]
        dy = pt2_main[1] - pt1_main[1]
        len_main = np.hypot(dx, dy)
        perp_dx = -dy / len_main
        perp_dy = dx / len_main

        threshold = 2.0  
        candidates = []

        for c in largest:
            px, py = c[0]
            vec_x = px - pt_2_3[0]
            vec_y = py - pt_2_3[1]
            proj_len = abs(vec_x * dx / len_main + vec_y * dy / len_main)
            if proj_len < threshold:
                candidates.append((px, py))

        max_perp_dist = 0
        pt_pos, pt_neg = None, None
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                d = np.linalg.norm(np.array(candidates[i]) - np.array(candidates[j]))
                if d > max_perp_dist:
                    max_perp_dist = d
                    pt_pos, pt_neg = candidates[i], candidates[j]

    return {
        'main': max_dist,
        'sub': max_perp_dist if pt_pos and pt_neg else None,
        'angle': np.rad2deg(np.arctan2(dy, dx))
    }
