# examples/wanvideo/model_training/evaluate_metrics.py
import os
import torch
import argparse
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms.functional import to_pil_image

# 尝试导入 LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: `lpips` library not found. LPIPS metric will be skipped. Install with `pip install lpips`.")
    LPIPS_AVAILABLE = False

def sobel_edge_map(img_tensor: torch.Tensor) -> torch.Tensor:
    """从图像张量计算Sobel边缘图，与训练脚本中的版本保持一致。"""
    if img_tensor.size(1) == 3:
        gray = 0.299 * img_tensor[:, 0] + 0.587 * img_tensor[:, 1] + 0.114 * img_tensor[:, 2]
    else:
        gray = img_tensor[:, 0]
    gray = gray.unsqueeze(1)
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img_tensor.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.permute(0, 1, 3, 2)
    gx = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
    
    # 归一化并锐化
    magnitude = magnitude / (magnitude.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return torch.sigmoid((magnitude - 0.1) * 10.0)

def calculate_metrics(output_dir: str, device: str = "cuda"):
    """
    为一次完整的训练运行计算评估指标。
    它期望在 output_dir 中找到 'ground_truth.pt' 和 'predicted_frames.pt' 文件。
    """
    print(f"Evaluating results in: {output_dir}")
    
    gt_path = os.path.join(output_dir, "ground_truth.pt")
    pred_path = os.path.join(output_dir, "predicted_frames.pt")

    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        print(f"Warning: Could not find ground_truth.pt or predicted_frames.pt in {output_dir}. Skipping evaluation.")
        return

    # 加载数据
    frames_gt = torch.load(gt_path, map_location=device)
    frames_pred = torch.load(pred_path, map_location=device)
    
    B, T, C, H, W = frames_gt.shape
    metrics = {}

    # 1. LPIPS (感知相似度)
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        # LPIPS期望图像范围在 [-1, 1]
        frames_gt_lpips = frames_gt.view(-1, C, H, W) * 2 - 1
        frames_pred_lpips = frames_pred.view(-1, C, H, W) * 2 - 1
        with torch.no_grad():
            lpips_score = lpips_fn(frames_gt_lpips, frames_pred_lpips).mean().item()
        metrics['LPIPS'] = lpips_score
    else:
        metrics['LPIPS'] = float('nan')

    # 2. Edge IoU (结构相似度)
    with torch.no_grad():
        edges_gt = sobel_edge_map(frames_gt.view(-1, C, H, W)) > 0.5
        edges_pred = sobel_edge_map(frames_pred.view(-1, C, H, W)) > 0.5
        intersection = (edges_gt & edges_pred).sum().float()
        union = (edges_gt | edges_pred).sum().float()
        edge_iou = (intersection / (union + 1e-6)).item()
        metrics['Edge_IoU'] = edge_iou

    # 3. CLIP Score (多模态时序一致性)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        # 我们计算生成视频的相邻帧之间的CLIP相似度
        pil_images = [to_pil_image(frame.clamp(0, 1)) for frame_seq in frames_pred for frame in frame_seq]
        clip_inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True).to(device)
        image_features = clip_model.get_image_features(**clip_inputs).view(B, T, -1)
    
        clip_consistency = torch.nn.functional.cosine_similarity(image_features[:, 1:], image_features[:, :-1], dim=-1).mean().item()
        metrics['CLIP_Consistency'] = clip_consistency

    # 将指标保存到CSV文件
    metrics_df = pd.DataFrame([metrics])
    csv_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Wrote metrics to {csv_path}")
    print("Metrics:")
    print(metrics_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate metrics for a temporal model training run.")
    parser.add_argument("output_dir", type=str, help="Directory containing the saved model outputs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on.")
    args = parser.parse_args()
    
    # 确保文件夹存在
    if not os.path.isdir(args.output_dir):
        print(f"Error: Output directory not found at {args.output_dir}")
        exit(1)
        
    calculate_metrics(args.output_dir, args.device)