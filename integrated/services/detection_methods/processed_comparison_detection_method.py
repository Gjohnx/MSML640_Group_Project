import numpy as np
import cv2
from services.detection_methods import DetectionMethod

class ProcessedComparisonDetectionMethod(DetectionMethod):

    def process(self, frame: np.ndarray):

        # Full back code if the webcam does not support 720P
        if frame is None or frame.size == 0:
            dummy = np.zeros((720, 720, 3), dtype=np.uint8)
            cube_colors = np.full((6, 3, 3), '?', dtype=str)
            return dummy, cube_colors, None

        # Crop the image according to the outer ratio
        outer_ratio = 0.5
        h, w = frame.shape[:2]
        S = min(h, w)
        outer_len = int(S * outer_ratio)
        cx, cy = w // 2, h // 2
        ox1 = cx - outer_len // 2
        oy1 = cy - outer_len // 2
        ox2 = cx + outer_len // 2
        oy2 = cy + outer_len // 2
        ox1 = max(0, ox1); oy1 = max(0, oy1)
        ox2 = min(w, ox2); oy2 = min(h, oy2)
        cropped = frame[oy1:oy2, ox1:ox2]

        # REsize the cropped image to 720x720 pixels
        cropped_720 = cv2.resize(cropped, (720, 720), interpolation=cv2.INTER_LINEAR)
        final_face = cropped_720

        # Here is just for the plceholder, no actual processing
        cube_colors = np.full((6, 3, 3), '?', dtype=str)

        

        # Corner Detection
        original = cropped_720.copy()
        H, W = original.shape[:2]

        # Preprocessing first
        alpha = 2
        beta = 30
        gaussian_kernel = (5,5)
        unsharp_weight = 2
        adj = cv2.convertScaleAbs(original, alpha=alpha, beta=beta)
        blur = cv2.GaussianBlur(adj, gaussian_kernel, 5)
        sharp = cv2.addWeighted(adj, unsharp_weight, blur, -(unsharp_weight - 1), 0)
        pre_gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)

        # When we apply ADAPTIVE THRESHOLD + CLOSING to the detection
        th = cv2.adaptiveThreshold(
            pre_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 5
        )
        th_closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        # Restrict to the corner area to the four boxes
        outer_ratio = 0.5
        inner_ratio = 0.25
        outer_size = float(W)
        inner_size = outer_size * (inner_ratio / outer_ratio)
        ring = int((outer_size - inner_size) / 2)
        TL_roi = (0, ring, 0, ring)
        TR_roi = (W-ring, W, 0, ring)
        BR_roi = (W-ring, W, H-ring, H)
        BL_roi = (0, ring, H-ring, H)

        # Detect corners using the innovative kernel on four restricted boxes
        kernel_size = 21
        k = kernel_size
        K = np.zeros((k, k), dtype=np.float32)
        center = k // 2
        for i in range(k):
            for j in range(k):
                dist = np.sqrt((i-center)**2 + (j-center)**2)
                K[i,j] = max(0, 1 - dist / (center * 1.2))
        TL_kernel = K.copy(); TL_kernel[k//2:, :] = 0; TL_kernel[:, k//2:] = 0
        TR_kernel = K.copy(); TR_kernel[k//2:, :] = 0; TR_kernel[:, :k//2] = 0
        BR_kernel = K.copy(); BR_kernel[:k//2, :] = 0; BR_kernel[:, :k//2] = 0
        BL_kernel = K.copy(); BL_kernel[:k//2, :] = 0; BL_kernel[:, k//2:] = 0
        TL_kernel /= np.sum(TL_kernel)
        TR_kernel /= np.sum(TR_kernel)
        BR_kernel /= np.sum(BR_kernel)
        BL_kernel /= np.sum(BL_kernel)

        # In the restricted boxes, we apply gaussian weighting to each pixel that increase the performance
        sigma = 1/3

        # Top-Left Corner detection, choose the largest response point
        x1, x2, y1, y2 = TL_roi
        patch = th_closed[y1:y2, x1:x2].astype(np.float32)
        h, w = patch.shape
        if h < k or w < k:
            tl = None
        else:
            resp = cv2.filter2D(patch, -1, TL_kernel)
            sx = np.linspace(-1, 1, w)
            sy = np.linspace(-1, 1, h)
            X, Y = np.meshgrid(sx, sy)
            region_weight = np.exp(-(X*X + Y*Y) / (2 * sigma * sigma))
            weighted = resp * region_weight
            ry, rx = np.unravel_index(np.argmax(weighted), weighted.shape)
            tl = np.array([rx + x1, ry + y1], dtype=np.float32)

        # Top-Right Corner detection, choose the largest response point
        x1, x2, y1, y2 = TR_roi
        patch = th_closed[y1:y2, x1:x2].astype(np.float32)
        h, w = patch.shape
        if h < k or w < k:
            tr = None
        else:
            resp = cv2.filter2D(patch, -1, TR_kernel)
            sx = np.linspace(-1, 1, w)
            sy = np.linspace(-1, 1, h)
            X, Y = np.meshgrid(sx, sy)
            region_weight = np.exp(-(X*X + Y*Y) / (2 * sigma * sigma))
            weighted = resp * region_weight
            ry, rx = np.unravel_index(np.argmax(weighted), weighted.shape)
            tr = np.array([rx + x1, ry + y1], dtype=np.float32)

        # Bottom-Right, same idea
        x1, x2, y1, y2 = BR_roi
        patch = th_closed[y1:y2, x1:x2].astype(np.float32)
        h, w = patch.shape
        if h < k or w < k:
            br = None
        else:
            resp = cv2.filter2D(patch, -1, BR_kernel)
            sx = np.linspace(-1, 1, w)
            sy = np.linspace(-1, 1, h)
            X, Y = np.meshgrid(sx, sy)
            region_weight = np.exp(-(X*X + Y*Y) / (2 * sigma * sigma))
            weighted = resp * region_weight
            ry, rx = np.unravel_index(np.argmax(weighted), weighted.shape)
            br = np.array([rx + x1, ry + y1], dtype=np.float32)

        # Bottom-Left, same idea
        x1, x2, y1, y2 = BL_roi
        patch = th_closed[y1:y2, x1:x2].astype(np.float32)
        h, w = patch.shape
        if h < k or w < k:
            bl = None
        else:
            resp = cv2.filter2D(patch, -1, BL_kernel)
            sx = np.linspace(-1, 1, w)
            sy = np.linspace(-1, 1, h)
            X, Y = np.meshgrid(sx, sy)
            region_weight = np.exp(-(X*X + Y*Y) / (2 * sigma * sigma))
            weighted = resp * region_weight
            ry, rx = np.unravel_index(np.argmax(weighted), weighted.shape)
            bl = np.array([rx + x1, ry + y1], dtype=np.float32)

        # Output the visualization image
        if any(v is None for v in [tl, tr, br, bl]):
            corners = None
            corner_vis = np.zeros_like(original)
        else:
            corners = np.float32([tl, tr, br, bl])
            corner_vis = cv2.cvtColor(th_closed, cv2.COLOR_GRAY2BGR)
            for (x, y) in corners:
                cv2.circle(corner_vis, (int(x), int(y)), 10, (0, 0, 255), -1)
        


        # Wrap perspective using the corner locations above
        warp_size = 450   # Set the wrap size to 450, which is not too large but clear enough for processing
        if corners is None:
            # If no corner detect, we put a blank zero here
            warp_img = np.zeros((warp_size, warp_size, 3), dtype=np.uint8)
        else:
            # Take the corners and perform the warping
            src_pts = corners.astype(np.float32)
            dst_pts = np.float32([
                [0, 0],
                [warp_size - 1, 0],
                [warp_size - 1, warp_size - 1],
                [0, warp_size - 1]
            ])
            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warp_img = cv2.warpPerspective(cropped_720, H, (warp_size, warp_size))



        # Color Boosting and Enhancement for the warping face
        # The original warping face below:
        img_input = warp_img.copy()

        # Gamma for brightness
        gamma = 2.5
        table = np.array([
            ((i / 255.0) ** (1.0 / gamma)) * 255
            for i in range(256)
        ]).astype("uint8")
        img_brightness = cv2.LUT(img_input, table)

        # Contrast enhancement
        contrast = 1.5
        img_f = img_brightness.astype(np.float32)
        img_bright_contrast = (img_f - 128) * contrast + 128
        img_bright_contrast = np.clip(img_bright_contrast, 0, 255).astype(np.uint8)

        # Color Boosting (using HSV space and the color convector of cv2)
        color_boost = 1.1
        hsv = cv2.cvtColor(img_bright_contrast, cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = cv2.split(hsv)
        S *= color_boost
        S = np.clip(S, 0, 255)
        hsv_boosted = cv2.merge([H, S, V]).astype(np.uint8)
        img_color_boost = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

        # Gaussian Blue (reduce the noise)
        blur_ksize = 5
        img_blur = cv2.GaussianBlur(img_color_boost, (blur_ksize, blur_ksize), 0)

        # The output image
        final_img = img_blur.copy()

        return final_img, final_img, None
    
    def reset(self):
        pass