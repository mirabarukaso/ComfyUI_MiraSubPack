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
            category=CAT,
            inputs=[
                io.Image.Input("base_img", optional=False),
                io.Image.Input("patch_img", optional=False),
                io.Float.Input("core_coverage", default=0.6, min=0.0, max=1.0, step=0.05),
                io.Int.Input("blend_width", default=30, min=0, max=100, step=1),
                io.Int.Input("feather_size", default=15, min=1, max=51, step=2),
                io.Float.Input("edge_blend_ratio", default=0.7, min=0.0, max=1.0, step=0.05),
                io.Int.Input("sift_features", default=5000, min=1000, max=10000, step=500),
                io.Float.Input("match_ratio", default=0.7, min=0.5, max=0.9, step=0.05),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, base_img, patch_img, core_coverage=0.6, blend_width=30, 
                feather_size=15, edge_blend_ratio=0.7, sift_features=5000, 
                match_ratio=0.7) -> io.NodeOutput:
        """
        Args:
            base_img: Base image tensor
            patch_img: Patch image tensor to merge
            core_coverage: Core region coverage ratio (0-1), where patch fully covers base
            blend_width: Width of blend edge region in pixels
            feather_size: Feather radius for edge smoothing
            edge_blend_ratio: Patch weight in blend region
            sift_features: Number of SIFT features to detect
            match_ratio: Lowe's ratio threshold for feature matching
        """
        
        if feather_size % 2 == 0:
            feather_size += 1
        
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
                    core_coverage, blend_width, feather_size, 
                    edge_blend_ratio, sift_features, match_ratio
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
    def process_single_image(base_img, patch_img, core_coverage, blend_width, 
                            feather_size, edge_blend_ratio, sift_features, match_ratio):
        """
        Merge patch image onto base using pixel alignment with center-based coverage.
        
        Strategy:
        1. Core region (center of patch): 100% patch coverage
        2. Blend region (around core): Gradient blending
        3. Outer region (edge of patch): Weaker blending
        4. Background: 100% base image
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

        patch_gray = cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY)
        _, original_mask = cv2.threshold(patch_gray, 10, 255, cv2.THRESH_BINARY)
        
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
        core_mask = (dist_from_center <= core_radius).astype(np.float32)
        core_mask = core_mask * (warped_mask.astype(np.float32) / 255.0)
        
        blend_outer_radius = core_radius + blend_width
        #blend_mask_raw = ((dist_from_center > core_radius) & (dist_from_center <= blend_outer_radius)).astype(np.float32)
        
        blend_gradient = np.zeros_like(dist_from_center, dtype=np.float32)
        blend_region = (dist_from_center > core_radius) & (dist_from_center <= blend_outer_radius)
        if blend_width > 0:
            blend_gradient[blend_region] = 1.0 - ((dist_from_center[blend_region] - core_radius) / blend_width)
        else:
            blend_gradient[blend_region] = 1.0
        
        if feather_size > 1:
            blend_gradient = cv2.GaussianBlur(blend_gradient, (feather_size, feather_size), 0)
        
        blend_mask = blend_gradient * (warped_mask.astype(np.float32) / 255.0)
        blend_mask = blend_mask * (1.0 - core_mask)
        
        outer_mask = (warped_mask.astype(np.float32) / 255.0) - core_mask - blend_mask
        outer_mask = np.clip(outer_mask, 0, 1)
        
        core_mask_3ch = np.stack([core_mask] * 3, axis=2)
        blend_mask_3ch = np.stack([blend_mask] * 3, axis=2)
        outer_mask_3ch = np.stack([outer_mask] * 3, axis=2)
        
        print(f"[MiraSubPack:ImageMerge] Core radius: {core_radius:.1f} pixels")
        print(f"[MiraSubPack:ImageMerge] Blend radius: {blend_outer_radius:.1f} pixels")
        print(f"[MiraSubPack:ImageMerge] Core pixels: {np.sum(core_mask > 0.5)}")
        print(f"[MiraSubPack:ImageMerge] Blend pixels: {np.sum(blend_mask > 0.1)}")
        print(f"[MiraSubPack:ImageMerge] Outer pixels: {np.sum(outer_mask > 0.1)}")
        
        base_float = base_img.astype(np.float32)
        patch_float = warped_patch.astype(np.float32)
        
        total_patch_mask_raw = np.clip(core_mask_3ch + blend_mask_3ch + outer_mask_3ch, 0, 1)
        background_mask = 1.0 - total_patch_mask_raw
        
        patch_region = (total_patch_mask_raw > 0.01)
        
        core_mask_3ch_norm = core_mask_3ch.copy()
        blend_mask_3ch_norm = blend_mask_3ch.copy()
        outer_mask_3ch_norm = outer_mask_3ch.copy()
        
        layer_sum = core_mask_3ch + blend_mask_3ch + outer_mask_3ch
        layer_sum = np.maximum(layer_sum, 1e-6)
        
        core_mask_3ch_norm = np.divide(core_mask_3ch, layer_sum, 
                                       where=patch_region, out=core_mask_3ch_norm)
        blend_mask_3ch_norm = np.divide(blend_mask_3ch, layer_sum,
                                        where=patch_region, out=blend_mask_3ch_norm)
        outer_mask_3ch_norm = np.divide(outer_mask_3ch, layer_sum,
                                        where=patch_region, out=outer_mask_3ch_norm)
                
        patch_result = (patch_float * core_mask_3ch_norm +
                       (patch_float * edge_blend_ratio + 
                        base_float * (1.0 - edge_blend_ratio)) * blend_mask_3ch_norm +
                       (patch_float * (edge_blend_ratio * 0.8) + 
                        base_float * (1.0 - edge_blend_ratio * 0.8)) * outer_mask_3ch_norm)
        
        result = patch_result * total_patch_mask_raw + base_float * background_mask
        
        output = np.clip(result, 0, 255).astype(np.uint8)
        
        print(f"[MiraSubPack:ImageMerge] Blend complete (core_coverage={core_coverage}, blend_width={blend_width})")

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