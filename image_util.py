import torch
import numpy as np
import cv2
from comfy_api.latest import io

CAT = "Mira/SubPack"

class ImageMergeByPixelAlign(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageMergeByPixelAlign_MiraSubPack",
            display_name="Image Merge By Pixel Align",
            category=CAT,
            inputs=[
                io.Image.Input("base_img", optional=False),
                io.Image.Input("patch_img", optional=False),
                io.Int.Input("bg_color_r", default=255, min=0, max=255, step=1, tooltip="Background color Red component (0-255) to exclude from patch"),
                io.Int.Input("bg_color_g", default=0, min=0, max=255, step=1, tooltip="Background color Green component (0-255) to exclude from patch"),
                io.Int.Input("bg_color_b", default=255, min=0, max=255, step=1, tooltip="Background color Blue component (0-255) to exclude from patch"),
                io.Int.Input("bg_tolerance", default=10, min=0, max=100, step=1, tooltip="Color distance tolerance for background detection"),
                io.Float.Input("core_coverage", default=0.6, min=0.0, max=1.0, step=0.05, tooltip="Core region coverage ratio (0-1), where patch fully covers base"),
                io.Int.Input("blend_width", default=30, min=0, max=200, step=5, tooltip="Width of blend transition in pixels from core edge"),
                io.Float.Input("blend_strength", default=0.7, min=0.0, max=1.0, step=0.05, tooltip="Patch opacity in blend region (0=invisible, 1=fully opaque)"),
                io.Combo.Input("blend_mode", default="smooth", options=["smooth", "linear", "full_gradient", "none"], 
                                tooltip="Blend transition curve:\n"
                                        "- 'smooth': sigmoid within blend_width\n"
                                        "- 'linear': linear within blend_width\n"
                                        "- 'full_gradient': gradient from core edge to patch edge\n"
                                        "- 'none': no blend outside core"),
                io.Int.Input("sift_features", default=5000, min=1000, max=10000, step=500, tooltip="Number of SIFT features to detect"),
                io.Float.Input("match_ratio", default=0.7, min=0.5, max=0.9, step=0.05, tooltip="Lowe's ratio threshold for feature matching"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, base_img, patch_img, bg_color_r=0, bg_color_g=0, bg_color_b=0, 
                bg_tolerance=10, core_coverage=0.6, blend_width=30, 
                blend_strength=0.7, blend_mode="smooth", sift_features=5000, 
                match_ratio=0.7) -> io.NodeOutput:
        """
        Args:
            base_img: Base image tensor
            patch_img: Patch image tensor to merge
            bg_color_r/g/b: Background color RGB to exclude from patch (0-255)
            bg_tolerance: Color distance tolerance for background detection
            core_coverage: Core region coverage ratio (0-1), where patch fully covers base
            blend_width: Width of blend transition in pixels from core edge
            blend_strength: Patch opacity in blend region (0=invisible, 1=fully opaque)
            blend_mode: Blend transition curve 
                       ("smooth"=sigmoid within blend_width, 
                        "linear"=linear within blend_width,
                        "full_gradient"=gradient from core edge to patch edge,
                        "none"=no blend outside core)
            sift_features: Number of SIFT features to detect
            match_ratio: Lowe's ratio threshold for feature matching
        """
        
        batch_size = base_img.shape[0]
        output_tensors = []

        for i in range(batch_size):
            curr_base = base_img[i if i < base_img.shape[0] else 0]
            curr_patch = patch_img[i if i < patch_img.shape[0] else 0]

            img_base_cv = cls.tensor_to_cv2(curr_base)
            img_patch_cv = cls.tensor_to_cv2(curr_patch)

            try:
                result_cv = cls.process_single_image(
                    img_base_cv, img_patch_cv, 
                    (bg_color_b, bg_color_g, bg_color_r), bg_tolerance,
                    core_coverage, blend_width, blend_strength, blend_mode,
                    sift_features, match_ratio
                )
            except Exception as e:
                print(f"[ImageMergeByPixelAlign] Error on image {i}: {e}")
                import traceback
                traceback.print_exc()
                result_cv = img_base_cv

            result_tensor = cls.cv2_to_tensor(result_cv)
            output_tensors.append(result_tensor)

        final_output = torch.stack(output_tensors)
        return io.NodeOutput(final_output)

    @staticmethod
    def process_single_image(base_img, patch_img, bg_color, bg_tolerance,
                            core_coverage, blend_width, blend_strength, blend_mode, 
                            sift_features, match_ratio):
        """
        Merge patch image onto base using pixel alignment with center-based coverage.
        
        Strategy:
        1. Core region (center): 100% patch coverage (fully opaque)
        2. Blend region (transition): Smooth gradient from patch to base
        3. Background: 100% base image (no patch)
        
        Blend modes:
        - "smooth": Sigmoid curve for natural falloff within blend_width
        - "linear": Linear gradient within blend_width
        - "full_gradient": Core is 100% opaque, entire outer region gradients to edge
        - "none": No blending outside core (hard edge with alignment only)
        """
        gray_base = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        gray_patch = cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY)

        detector = cv2.SIFT_create(nfeatures=sift_features)
        kp1, des1 = detector.detectAndCompute(gray_patch, None)
        kp2, des2 = detector.detectAndCompute(gray_base, None)

        if des1 is None or des2 is None:
            print("[MiraSubPack:ImageMerge] No descriptors found.")
            return base_img

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < match_ratio * n.distance:
                    good_matches.append(m)

        print(f"[MiraSubPack:ImageMerge] Found {len(good_matches)} good matches")

        if len(good_matches) < 4:
            print(f"[MiraSubPack:ImageMerge] Not enough matches: {len(good_matches)}/4")
            return base_img

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask_homography = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            print("[MiraSubPack:ImageMerge] Homography calculation failed")
            return base_img

        if mask_homography is not None:
            inliers = np.sum(mask_homography)
            print(f"[MiraSubPack:ImageMerge] Inliers: {inliers}/{len(good_matches)}")

        h, w, _ = base_img.shape
        warped_patch = cv2.warpPerspective(patch_img, M, (w, h), 
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(0, 0, 0))

        bg_color_array = np.array(bg_color, dtype=np.float32)
        patch_float = patch_img.astype(np.float32)
        color_distance = np.sqrt(np.sum((patch_float - bg_color_array) ** 2, axis=2))
        
        original_mask = (color_distance > bg_tolerance).astype(np.uint8) * 255
        
        print(f"[MiraSubPack:ImageMerge] Background color: BGR{bg_color}, tolerance: {bg_tolerance}")
        print(f"[MiraSubPack:ImageMerge] Detected foreground pixels: {np.sum(original_mask > 0)}")
        
        warped_mask = cv2.warpPerspective(original_mask, M, (w, h),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=0)
        
        _, warped_mask = cv2.threshold(warped_mask, 200, 255, cv2.THRESH_BINARY)
        
        kernel_clean = np.ones((3, 3), np.uint8)
        warped_mask = cv2.erode(warped_mask, kernel_clean, iterations=1)
        
        moments = cv2.moments(warped_mask)
        if moments['m00'] == 0:
            print("[MiraSubPack:ImageMerge] Cannot find patch center")
            return base_img
            
        patch_cx = int(moments['m10'] / moments['m00'])
        patch_cy = int(moments['m01'] / moments['m00'])
        print(f"[MiraSubPack:ImageMerge] Patch center: ({patch_cx}, {patch_cy})")
        
        patch_points = np.column_stack(np.where(warped_mask > 0))
        if len(patch_points) == 0:
            return base_img
        
        distances = np.sqrt((patch_points[:, 1] - patch_cx)**2 + 
                          (patch_points[:, 0] - patch_cy)**2)
        max_radius = np.max(distances)
        print(f"[MiraSubPack:ImageMerge] Patch max radius: {max_radius:.1f} pixels")
        
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_coords - patch_cx)**2 + (y_coords - patch_cy)**2)
        
        core_radius = max_radius * core_coverage
        blend_outer_radius = core_radius + blend_width
        
        overall_mask = warped_mask.astype(np.float32) / 255.0
        
        patch_alpha = np.zeros_like(dist_from_center, dtype=np.float32)
        
        core_region = dist_from_center <= core_radius
        patch_alpha[core_region] = 1.0
        
        if blend_mode == "full_gradient":
            outer_region = (dist_from_center > core_radius) & (overall_mask > 0)
            
            outer_dist_normalized = (dist_from_center[outer_region] - core_radius) / (max_radius - core_radius)
            outer_dist_normalized = np.clip(outer_dist_normalized, 0, 1)
            
            patch_alpha[outer_region] = (1.0 - outer_dist_normalized) * blend_strength
            
        elif blend_width > 0 and blend_mode != "none":
            blend_region = (dist_from_center > core_radius) & (dist_from_center <= blend_outer_radius)
            
            normalized_dist = (dist_from_center[blend_region] - core_radius) / blend_width
            
            if blend_mode == "smooth":
                sigmoid_curve = 1.0 / (1.0 + np.exp(10 * (normalized_dist - 0.5)))
                patch_alpha[blend_region] = sigmoid_curve * blend_strength
            elif blend_mode == "linear":
                patch_alpha[blend_region] = (1.0 - normalized_dist) * blend_strength
        
        patch_alpha = patch_alpha * overall_mask
        
        patch_alpha = cv2.GaussianBlur(patch_alpha, (5, 5), 0)
        
        patch_alpha_3ch = np.stack([patch_alpha] * 3, axis=2)
        
        print(f"[MiraSubPack:ImageMerge] Core radius: {core_radius:.1f} pixels")
        if blend_mode == "full_gradient":
            print(f"[MiraSubPack:ImageMerge] Blend mode: {blend_mode} (core to edge), strength: {blend_strength}")
        else:
            print(f"[MiraSubPack:ImageMerge] Blend outer radius: {blend_outer_radius:.1f} pixels")
            print(f"[MiraSubPack:ImageMerge] Blend mode: {blend_mode}, strength: {blend_strength}")
        print(f"[MiraSubPack:ImageMerge] Core pixels (alpha=1.0): {np.sum(patch_alpha > 0.99)}")
        print(f"[MiraSubPack:ImageMerge] Blend pixels (0<alpha<1): {np.sum((patch_alpha > 0.01) & (patch_alpha < 0.99))}")
        
        base_float = base_img.astype(np.float32)
        patch_float = warped_patch.astype(np.float32)
        
        result = patch_float * patch_alpha_3ch + base_float * (1.0 - patch_alpha_3ch)
        
        output = np.clip(result, 0, 255).astype(np.uint8)
        
        print("[MiraSubPack:ImageMerge] Blend complete")

        return output

    @staticmethod
    def tensor_to_cv2(tensor_img):
        """Convert Torch Tensor (H,W,C) RGB 0..1 to Numpy (H,W,C) BGR 0..255"""
        np_img = tensor_img.numpy()
        np_img = (np_img * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_tensor(cv2_img):
        """Convert Numpy (H,W,C) BGR 0..255 to Torch Tensor (H,W,C) RGB 0..1"""
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        rgb_img = rgb_img.astype(np.float32) / 255.0
        return torch.from_numpy(rgb_img)