import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, predict, load_image
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
import torchvision.transforms as T
from torchvision.ops import box_convert
import gc
from pathlib import Path
from tqdm import tqdm
from torchvision.models import resnet34
from torchvision import transforms

# Set memory management configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class VideoObjectFeatureExtractor:
    def __init__(self,
                 sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
                 sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
                 grounding_dino_config="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 grounding_dino_checkpoint="gdino_checkpoints/groundingdino_swint_ogc.pth",
                 device="cuda"):
        """Initialize models and configurations"""
        self.device = device
        
        # Enable memory optimization
        torch.cuda.empty_cache()
        gc.collect()
        
        # 修改：使用 float16 而不是 bfloat16
        self.dtype = torch.float16
        
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Initialize ResNet34
        self.feature_extractor = resnet34(pretrained=True).to(device)
        self.feature_extractor.eval()
        # Remove the final fully connected layer
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        # Initialize SAM 2 models
        with torch.cuda.amp.autocast(dtype=self.dtype):
            self.video_predictor = build_sam2_video_predictor(sam2_model_config, sam2_checkpoint)
            sam2_model = build_sam2(sam2_model_config, sam2_checkpoint)
            self.image_predictor = SAM2ImagePredictor(sam2_model)
            
            # Initialize Grounding DINO from local checkpoint
            self.grounding_model = load_model(
                model_config_path=grounding_dino_config,
                model_checkpoint_path=grounding_dino_checkpoint,
                device=device
            )

    def process_video(self, video_path, text_prompt, output_dir="./features",
                     box_threshold=0.25,    
                     text_threshold=0.25,   
                     max_size=480):
        """Process video and extract features for specified object"""
        
        # Create output directories
        output_dir = Path(output_dir)
        frames_dir = output_dir / "frames"
        features_dir = output_dir / "features"
        annotated_frames_dir = output_dir / "annotated_frames"
        
        # 创建所有必要的目录
        for d in [frames_dir, features_dir, annotated_frames_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        # 使用 OpenCV 读取视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video at {video_path}")
        
        frame_paths = []
        frame_idx = 0
        
        with sv.ImageSink(target_dir_path=frames_dir, overwrite=True) as sink:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame
                h, w = frame.shape[:2]
                if h > w:
                    new_h = min(h, max_size)
                    new_w = int(w * new_h / h)
                else:
                    new_w = min(w, max_size)
                    new_h = int(h * new_w / w)
                frame = cv2.resize(frame, (new_w, new_h))
                
                # Save frame
                frame_path = f"{frame_idx:05d}.jpg"
                sink.save_image(frame, frame_path)
                frame_paths.append(str(frames_dir / frame_path))
                
                frame_idx += 1
                
                if frame_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        cap.release()
        
        # Initialize video predictor state
        inference_state = self.video_predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,  # 添加内存优化选项
            async_loading_frames=True    # 添加异步加载选项
        )
        
        # Process first frame
        try:
            with torch.cuda.amp.autocast(enabled=False):
                image_source, image = load_image(frame_paths[0])
                self.image_predictor.set_image(image_source)
                
                # 使用更精确的文本提示
                text_prompt = text_prompt.lower().strip()
                if not text_prompt.endswith('.'):
                    text_prompt += '.'
                
                print(f"Using text prompt: {text_prompt}")
                
                boxes, confidences, labels = predict(
                    model=self.grounding_model,
                    image=image,
                    caption=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
                
                if len(boxes) == 0:
                    print("No objects detected in first frame")
                    return None, None, None
                
                h, w, _ = image_source.shape
                input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
                input_boxes = input_boxes * torch.tensor([w, h, w, h], device=input_boxes.device)
                input_boxes = input_boxes.cpu().numpy()
                
                masks, scores, logits = self.image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                
                # 修改掩码选择部分的代码
                if len(masks) > 0:
                    valid_indices = []
                    scores = []
                    positions = []  # 记录位置信息
                    
                    for idx, mask in enumerate(masks):
                        # 确保掩码是布尔类型
                        mask = mask.astype(bool)
                        
                        # 获取掩码区域的原始图像
                        masked_region = image_source.copy()
                        masked_region[~mask] = 0
                        
                        # 转换到 HSV 颜色空间
                        hsv = cv2.cvtColor(masked_region, cv2.COLOR_RGB2HSV)
                        
                        # 定义绿色范围
                        lower_green = np.array([35, 40, 40])
                        upper_green = np.array([95, 255, 255])
                        
                        # 创建绿色掩码
                        green_mask = cv2.inRange(hsv, lower_green, upper_green)
                        
                        # 计算绿色像素的比例
                        green_ratio = np.sum(green_mask) / (np.sum(mask) + 1e-6)
                        
                        # 计算掩码的位置和大小
                        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                        center_x = x + w/2
                        relative_x = center_x / mask.shape[1]  # 相对位置
                        
                        # 计算大小得分
                        area = np.sum(mask)
                        size_score = min(area / (mask.shape[0] * mask.shape[1] * 0.05), 1.0)
                        
                        # 计算总分数
                        total_score = (
                            green_ratio * 0.7 +     # 颜色权重
                            size_score * 0.3        # 大小权重
                        )
                        
                        print(f"Object {idx} - Green: {green_ratio:.2f}, Size: {size_score:.2f}, "
                              f"Total: {total_score:.2f}, Position: {relative_x:.2f}")
                        
                        # 如果得分超过阈值，加入候选列表
                        if total_score > 0.3 and green_ratio > 0.2:  # 提高颜色要求
                            valid_indices.append(idx)
                            scores.append(total_score)
                            positions.append(relative_x)
                    
                    if len(valid_indices) >= 2:
                        # 根据位置选择最左和最右的两个物体
                        sorted_indices = [x for _, x in sorted(zip(positions, valid_indices))]
                        selected_indices = [sorted_indices[0], sorted_indices[-1]]  # 选择最左和最右的
                        
                        # 只保留选中的两个掩码
                        masks = [masks[idx].astype(bool) for idx in selected_indices]
                        labels = [labels[idx] for idx in selected_indices]
                        input_boxes = input_boxes[selected_indices]
                        print(f"Selected left and right objects with positions: "
                              f"{[positions[valid_indices.index(idx)] for idx in selected_indices]}")
                    else:
                        print("Not enough suitable objects found")
                        return None, None, None
                
                print(f"Generated masks shape: {masks[0].shape if len(masks) > 0 else 'No masks'}")
                
                # 修改注册物体的部分
                if len(masks) > 0:
                    print(f"Registering {len(masks)} objects")
                    # 注册所有检测到的物体
                    for obj_id, mask in enumerate(masks, start=1):
                        if isinstance(mask, np.ndarray):
                            mask = torch.from_numpy(mask).to(self.device)
                        
                        print(f"Registering object {obj_id}")
                        _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=0,
                            obj_id=obj_id,  # 为每个物体分配唯一的ID
                            mask=mask
                        )
            
        except Exception as e:
            print(f"Error processing first frame: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
            
        # Register objects and track
        left_features = []   # 左侧抓手特征
        right_features = []  # 右侧抓手特征
        all_masks = []
        video_segments = {}
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                frame = cv2.imread(frame_paths[out_frame_idx])
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 处理掩码
                frame_masks = []
                for mask_logit in out_mask_logits:
                    mask = (mask_logit > 0.5).cpu().numpy()
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                    frame_masks.append(mask)
                
                if len(frame_masks) > 0:
                    print(f"Frame {out_frame_idx}: Tracking {len(frame_masks)} objects")
                    
                    # 处理特征提取和保存
                    for idx, mask in enumerate(frame_masks[:2]):
                        try:
                            # 创建掩码图像
                            masked_img = frame_rgb.copy()
                            mask_3d = np.stack([mask] * 3, axis=-1)
                            masked_img[~mask_3d] = 0
                            
                            # 提取特征
                            masked_pil = Image.fromarray(masked_img)
                            img_tensor = transform(masked_pil).unsqueeze(0).to(self.device)
                            
                            with torch.no_grad():
                                features = self.feature_extractor(img_tensor)
                                features = features.squeeze().cpu().numpy()
                            
                            if idx == 0:
                                left_features.append(features)
                            else:
                                right_features.append(features)
                        except Exception as e:
                            print(f"Error processing mask {idx}: {e}")
                            continue
                    
                    # 保存可视化结果到 annotated_frames 目录
                    self._save_visualization(frame_rgb, frame_masks, out_frame_idx, annotated_frames_dir)
                    all_masks.extend(frame_masks[:2])
                
                video_segments[out_frame_idx] = {
                    out_obj_id: mask for out_obj_id, mask in zip(out_obj_ids, frame_masks)
                }
                
                if out_frame_idx % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            traceback.print_exc()
            if len(left_features) == 0 and len(right_features) == 0:
                return None, None, None
        
        # Save results
        if left_features and right_features:
            left_features = np.stack(left_features, axis=0)
            right_features = np.stack(right_features, axis=0)
            
            features_dir = Path(output_dir) / "features"
            features_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(features_dir / "feature1.npy", left_features)
            np.save(features_dir / "feature2.npy", right_features)
            
            print(f"Saved left gripper features: {left_features.shape}")
            print(f"Saved right gripper features: {right_features.shape}")

            return np.stack([left_features, right_features], axis=1), np.array(all_masks), video_segments
        
        return None, None, None

    def _save_visualization(self, image, masks, frame_idx, output_dir):
        """Save visualization of segmentation"""
        if len(masks) == 0:
            cv2.imwrite(str(output_dir / f"frame_{frame_idx:05d}.jpg"), 
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return
        
        try:
            # 确保掩码是numpy数组
            if isinstance(masks, list):
                masks = np.stack(masks)
            elif isinstance(masks, np.ndarray) and masks.ndim == 2:
                masks = np.expand_dims(masks, 0)
            
            vis_img = image.copy()
            
            # 为左右抓手使用不同的颜色
            colors = [(0, 255, 0), (255, 0, 0)]  # 左侧绿色，右侧红色
            labels = ["Left Gripper", "Right Gripper"]
            
            # 逐个处理掩码
            for idx, mask in enumerate(masks):
                if idx >= 2:  # 只处理前两个掩码
                    break
                    
                color = colors[idx]
                label = labels[idx]
                
                # 创建掩码叠加
                mask_overlay = np.zeros_like(image)
                mask_overlay[mask] = color
                vis_img = cv2.addWeighted(vis_img, 1.0, mask_overlay, 0.5, 0)
                
                # 添加轮廓
                contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, color, 2)
                
                # 添加标签
                if contours:
                    # 使用轮廓中心作为标签位置
                    M = cv2.moments(contours[0])
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(vis_img, label, (cX-40, cY), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 保存结果
            cv2.imwrite(str(output_dir / f"frame_{frame_idx:05d}.jpg"), 
                        cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                    
        except Exception as e:
            print(f"Error in visualization: {e}")
            # 保存原始图像作为后备
            cv2.imwrite(str(output_dir / f"frame_{frame_idx:05d}.jpg"), 
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def main():
    extractor = VideoObjectFeatureExtractor()
    
    # 定义所有数据集路径
    datasets = [
        {
            'path': "/home/zfb/Grounded-SAM-2/Take 2025-02-21 03.15.24 PM",
            'video': "cs3tc_151532.274-151605.559.mp4"  
        },
        {
            'path': "/home/zfb/Grounded-SAM-2/Take 2025-02-21 03.30.57 PM",
            'video': "ghgu8_153105.001-153137.950.mp4"  
        },
        {
            'path': "/home/zfb/Grounded-SAM-2/Take 2025-02-21 03.39.06 PM",
            'video': "nmup3_153913.841-153947.498.mp4"  
        }
    ]
    
    text_prompt = "two green soft robotic grippers on the left and right sides."
    
    # 处理每个数据集
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset['path']}")
        video_path = str(Path(dataset['path']) / dataset['video'])
        output_dir = Path(dataset['path'])
        
        # 首先检查文件是否存在
        if not Path(video_path).exists():
            print(f"Video file not found: {video_path}")
            continue
            
        features, masks, segments = extractor.process_video(
            video_path=video_path,
            text_prompt=text_prompt,
            output_dir=output_dir,  # 使用数据集目录作为输出目录
            box_threshold=0.15,
            text_threshold=0.15
        )
        
        if features is not None:
            print(f"Extracted features shape: {features.shape}")
            if masks is not None:
                print(f"Extracted masks shape: {masks.shape}")
            print(f"Processed {len(segments)} frames")
            
            # 创建原始帧视频
            create_video_from_images(
                str(output_dir / "frames"),
                str(output_dir / "output.mp4")
            )
            
            # 创建标注帧视频
            create_video_from_images(
                str(output_dir / "annotated_frames"),
                str(output_dir / "segmentation_result.mp4")
            )
            print(f"Created videos at {output_dir}")
        else:
            print(f"Failed to process {video_path}")

if __name__ == "__main__":
    main() 