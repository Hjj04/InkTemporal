# TemporalModule Design Document

## 1. 目的与动机 (Purpose and Motivation)

`TemporalModule` 的核心目标是在视频生成的 latent 空间中，实现前一帧信息向当前帧的有效传递与融合。通过对前一帧的 latent 表征 (`z_prev`) 进行光流扭曲（warping），并将其与当前帧的 latent (`z_cur`) 进行自适应融合，该模块旨在显著提升生成视频的帧间连续性和稳定性，减少闪烁和内容突变。

## 2. API 接口说明 (API Specification)

模块的核心 `forward` 方法定义了其输入输出接口。

```python
def forward(
    self,
    z_prev: torch.Tensor,
    z_cur: torch.Tensor,
    s_prev: torch.Tensor = None,
    s_cur: torch.Tensor = None,
    flow: torch.Tensor = None,
):
    """
    对 latent 和可选的 style embedding 进行时序融合。

    Args:
        z_prev: torch.Tensor, shape [B, C, H, W]
            前一帧的 latent 表征。
        z_cur:  torch.Tensor, shape [B, C, H, W]
            当前帧的 latent 表征。
        s_prev: torch.Tensor or None, shape [B, Sdim]
            可选的，前一帧的 style embedding。
        s_cur:  torch.Tensor or None, shape [B, Sdim]
            可选的，当前帧的 style embedding。
        flow:   torch.Tensor or None, shape [B, 2, H, W]
            从前一帧到当前帧的光流场。格式为像素偏移量 (dx, dy)。
            如果为 None，则模块将执行身份扭曲（即 z_warp = z_prev）。

    Returns:
        z_fused: torch.Tensor, shape [B, C, H, W]
            融合后的 latent 表征。
        s_fused: torch.Tensor or None, shape [B, Sdim]
            融合后的 style embedding，如果输入了 s_prev 和 s_cur。
        aux: dict
            一个包含调试信息的字典，例如：
            {'alpha': alpha_value, 'z_warp_refined_mean': mean_of_warped_latent}
    """
```

## 3. 核心设计与约定 (Core Design and Conventions)

### 3.1. 光流格式约定 (Flow Format Convention)

-   **输入格式**: `forward` 方法接收的 `flow` 张量应为 **像素偏移量 (pixel offsets)**。其 shape 为 `[B, 2, H, W]`，其中 `flow[:, 0, :, :]` 代表水平方向的偏移 `dx`，`flow[:, 1, :, :]` 代表垂直方向的偏移 `dy`。单位为像素。
-   **内部转换**: 在 `warp_latent` 方法内部，该像素偏移量会被转换为 `torch.nn.functional.grid_sample` 所需的归一化坐标。转换公式如下：
    -   `dx_norm = dx / (W / 2.0)`
    -   `dy_norm = dy / (H / 2.0)`
    最终生成的采样网格 `sample_grid` 的范围在 `[-1, 1]` 之间。
-   **`align_corners`**: `grid_sample` 调用中设置 `align_corners=True`。这意味着输入的 `-1` 和 `1` 坐标被映射到输入张量的角点像素中心。这是一种常见的约定，但需要注意，如果使用的光流估计算法（如 RAFT）有不同的坐标系假设，可能需要进行相应的调整。
-   **`flow` 为 `None`**: 如果不提供 `flow`，模块默认将 `z_prev` 直接作为 `z_warp`，即执行身份扭曲。未来的实现可以扩展为在这种情况下内部预测一个小的运动流。

### 3.2. Alpha 门控机制 (Alpha Gating Mechanism)

融合过程由一个门控参数 `alpha` 控制，公式为 `z_fused = alpha * z_warp_refined + (1.0 - alpha) * z_cur`。

-   **实现**: 在当前的骨架实现中，`alpha` 是一个可学习的标量（scalar）。它在内部被存储为 `alpha_param`（一个 logit），通过 `torch.sigmoid(self.alpha_param)` 得到 `(0, 1)` 范围内的 `alpha` 值。这种方式可以确保 `alpha` 在优化过程中保持在有效范围内，且无梯度约束。
-   **可学习性**: 通过 `__init__` 方法的 `learnable_alpha: bool` 参数，可以控制 `alpha` 是否作为 `nn.Parameter`（可训练）或 `nn.Buffer`（固定）。
-   **初始化**: `alpha_init` 参数用于设置 `alpha` 的初始值。建议不要将其初始值设得过高（如 > 0.8），以避免在训练初期过度依赖不准确的 `z_warp`，从而破坏当前帧的细节。一个 `0.2` 到 `0.5` 之间的初始值是比较稳妥的选择。
-   **Warmup 策略 (建议)**: 为了在训练早期稳定学习过程，建议对 `alpha` 值进行 warmup。一个可行的策略是，在训练的前 N 个步骤（例如 5000 steps）内，将 `alpha` 从一个较小的值（如 0.0 或 0.1）线性地增加到一个目标值（如 0.8）。这可以在训练循环中通过外部调度器或回调函数实现。

## 4. 后续扩展计划 (Future Extension Plans)

当前的骨架实现为后续的功能扩展提供了基础。

-   **内置光流预测器 (Latent Flow Predictor)**: 当外部不提供 `flow` 时，模块内部可以集成一个轻量级的卷积网络，用于从 `z_prev` 和 `z_cur` 直接预测一个低分辨率的光流场。
-   **多尺度融合 (Multiscale Hook)**: `TemporalModule` 可以被设计为在 U-Net 的多个分辨率层级上应用。通过在解码器的不同阶段注入时序信息，可以更有效地保持从粗糙结构到精细纹理的一致性。
-   **边缘保持损失 (Edge Loss Integration)**: 为了防止 `z_warp` 过程中的模糊，可以引入一个额外的损失项。例如，可以计算 `z_warp` 和 `z_cur` 在边缘区域（可通过 Sobel 算子等检测）的差异，并将其加入总损失中，以鼓励模块保持图像的清晰度。
-   **空间可变 Alpha (Spatial Alpha Map)**: 当前的 `alpha` 是一个全局标量。可以将其扩展为一个空间映射（shape `[B, 1, H, W]`），使得融合权重可以根据图像内容（例如，运动区域和静止区域）进行自适应调整。

## 5. 调试与可视化建议 (Debugging and Visualization Suggestions)

-   **可视化 `z_warp`**: 验证 `warp_latent` 功能是否正常的最佳方法是将其输出 `z_warp`（或 `z_warp_refined`）送入 VAE 的解码器，并将结果图像保存。通过对比解码后的 `z_prev`、`z_cur` 和 `z_warp`，可以直观地判断扭曲效果是否符合预期。
-   **监控 `alpha` 值**: 在训练过程中，强烈建议将 `aux['alpha']` 的值记录到 TensorBoard 或其他日志系统中。观察 `alpha` 随时间的变化趋势，可以帮助判断模块是否在有效学习融合权重。一个健康的的 `alpha` 曲线通常会从初始值逐渐上升并稳定在一个合理的范围内。
-   **检查梯度**: 确保所有可学习参数（`alpha_param`，`refiner` 的权重等）在 `loss.backward()` 后都有非空的梯度，以确认它们参与了优化过程。