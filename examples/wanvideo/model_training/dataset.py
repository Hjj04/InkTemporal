# examples/wanvideo/model_training/dataset.py (FINAL CORRECTED VERSION)

import os
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from decord import VideoReader, cpu # 使用 decord

class RealVideoDataset(Dataset):
    def __init__(self, metadata_csv_path, videos_root_dir, num_frames=16, height=256, width=256):
        """
        Args:
            metadata_csv_path (str): Path to the metadata CSV file.
            videos_root_dir (str): Root directory where videos are stored.
            num_frames (int): Number of frames to sample from each video.
            height (int): Target height for frames.
            width (int): Target width for frames.
        """
        self.videos_root_dir = videos_root_dir
        self.num_frames = num_frames
        self.height = height
        self.width = width

        try:
            self.metadata = pd.read_csv(metadata_csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata CSV file not found at: {metadata_csv_path}")

        print(f"Dataset initialized with {len(self.metadata)} videos.")
        print(f"CSV columns found: {self.metadata.columns.tolist()}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        try:
            video_info = self.metadata.iloc[idx]
            
            # 使用 'video' 和 'prompt' 列名
            relative_video_path = video_info['video']
            text = video_info['prompt']
            
            full_video_path = os.path.join(self.videos_root_dir, relative_video_path)

            if not os.path.exists(full_video_path):
                raise FileNotFoundError(f"Video file not found: {full_video_path}")

            # --- 使用 decord 读取视频 ---
            vr = VideoReader(full_video_path, ctx=cpu(0), width=self.width, height=self.height)
            total_frames = len(vr)

            if total_frames == 0:
                raise ValueError(f"Video file is empty or unreadable: {full_video_path}")

            # 如果视频帧数不足，进行循环填充
            if total_frames < self.num_frames:
                # 创建一个索引列表，循环选择帧
                indices = list(range(total_frames))
                indices = (indices * (self.num_frames // total_frames + 1))[:self.num_frames]
                start_index = 0
            else:
                # 随机选择起始点进行采样
                start_index = random.randint(0, total_frames - self.num_frames)
                indices = list(range(start_index, start_index + self.num_frames))

            # 批量获取帧，这比一帧一帧读快得多
            video_frames = vr.get_batch(indices).asnumpy() # (T, H, W, C) numpy array
            
            # --- 帧预处理 ---
            # (T, H, W, C) -> (T, C, H, W) torch tensor
            frames_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2)
            
            # 归一化到 [0, 1]
            frames_tensor = frames_tensor / 255.0
            
            # 确保尺寸正确 (decord 在初始化时已经 resize，这里作为双重保险)
            if frames_tensor.shape[2] != self.height or frames_tensor.shape[3] != self.width:
                frames_tensor = F.interpolate(frames_tensor, size=(self.height, self.width), mode='bilinear', align_corners=False)

            return text, frames_tensor

        except Exception as e:
            # **重要**: 捕获异常后，打印详细信息并重新抛出，而不是无限递归
            print(f"ERROR: Failed to process video at index {idx} ({self.metadata.iloc[idx].get('video', 'N/A')}).")
            print(f"Original exception: {e}")
            # 重新抛出异常，让 PyTorch DataLoader 知道这个 worker 失败了
            # DataLoader 默认会尝试重新加载几次，如果持续失败则会停止训练，这是正确的行为
            raise e