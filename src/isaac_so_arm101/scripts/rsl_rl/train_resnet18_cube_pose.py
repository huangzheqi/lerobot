from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models


class CubePoseDataset(Dataset):
    def __init__(self, data_dir: str | Path):
        files = sorted(Path(data_dir).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No npz files found in {data_dir}")
        self.fixed_rgb = []
        self.object_xy = []
        for file in files:
            sample = np.load(file)
            self.fixed_rgb.append(sample["fixed_rgb"])
            self.object_xy.append(sample["object_position"][:2])
        self.fixed_rgb = np.stack(self.fixed_rgb)
        self.object_xy = np.stack(self.object_xy).astype(np.float32)

    def __len__(self) -> int:
        return self.fixed_rgb.shape[0]

    def __getitem__(self, idx: int):
        rgb = torch.from_numpy(self.fixed_rgb[idx]).float()
        if rgb.ndim != 3:
            raise ValueError(f"Invalid fixed_rgb shape: {rgb.shape}")
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        rgb = rgb.permute(2, 0, 1)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        rgb = torch.nn.functional.interpolate(rgb.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb = (rgb - mean) / std
        xy = torch.from_numpy(self.object_xy[idx])
        return rgb, xy


def build_model(pretrained: bool = True) -> nn.Module:
    # ImageNet-pretrained backbone gives a far stronger init for localising a small cube than random
    # weights (the from-scratch model under-fit on the harder current-env images: corr ~0.75). The
    # inference builder (vision_pose_resnet.py) uses weights=None and just loads the trained
    # state_dict, so the chosen backbone init here does not affect deployment. Needs torchvision to
    # fetch/cached the ImageNet weights; pass --no-pretrained to fall back to random init if offline.
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def evaluate_mae(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_err = 0.0
    total_n = 0
    with torch.inference_mode():
        for rgb, target_xy in loader:
            rgb = rgb.to(device)
            target_xy = target_xy.to(device)
            pred_xy = model(rgb)
            total_err += (pred_xy - target_xy).abs().sum(dim=1).sum().item()
            total_n += target_xy.shape[0]
    return total_err / max(total_n, 1)


def evaluate_fit(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Report per-axis correlation, R^2 and mean XY error (metres).

    MAE alone hides the failure mode where the model ignores the image and predicts the dataset
    mean: that still yields a modest MAE but corr~=0 / R^2<=0. A genuinely localising model has
    corr>~0.9. Always check corr/R^2, not just MAE.
    """
    model.eval()
    preds = []
    targets = []
    with torch.inference_mode():
        for rgb, target_xy in loader:
            preds.append(model(rgb.to(device)).cpu())
            targets.append(target_xy.cpu())
    pred = torch.cat(preds)
    tgt = torch.cat(targets)
    out = {}
    for i, axis in enumerate(("x", "y")):
        p, t = pred[:, i], tgt[:, i]
        pc, tc = p - p.mean(), t - t.mean()
        denom = (pc.std(unbiased=False) * tc.std(unbiased=False)).item()
        out[f"corr_{axis}"] = (pc * tc).mean().item() / denom if denom > 1e-9 else float("nan")
        ss_res = ((t - p) ** 2).sum().item()
        ss_tot = (tc ** 2).sum().item()
        out[f"r2_{axis}"] = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")
    out["xy_err_m"] = torch.norm(pred - tgt, dim=1).mean().item()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cube_pose_dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="selected_models/resnet18_cube_pose.pt")
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True,
        help="Use an ImageNet-pretrained ResNet18 backbone (default). Use --no-pretrained if offline.",
    )
    args = parser.parse_args()

    dataset = CubePoseDataset(args.data_dir)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        for rgb, target_xy in train_loader:
            rgb = rgb.to(device)
            target_xy = target_xy.to(device)
            pred_xy = model(rgb)
            loss = criterion(pred_xy, target_xy)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch + 1}/{args.epochs} loss={loss.item():.6f}")

    train_mae = evaluate_mae(model, train_loader, device)
    val_mae = evaluate_mae(model, val_loader, device)
    print(f"MAE_xy(train)={train_mae:.6f}")
    print(f"MAE_xy(val)={val_mae:.6f}")
    fit = evaluate_fit(model, val_loader, device)
    print(
        f"FIT(val) corr_x={fit['corr_x']:.3f} corr_y={fit['corr_y']:.3f} "
        f"R2_x={fit['r2_x']:.3f} R2_y={fit['r2_y']:.3f} xy_err={100 * fit['xy_err_m']:.2f}cm"
    )
    if min(fit["corr_x"], fit["corr_y"]) < 0.8:
        print(
            "[WARNING] val corr < 0.8 -> the model is NOT localising the cube (likely predicting the "
            "dataset mean). Do not deploy; check data quality / train longer / use a pretrained backbone."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"Saved model: {output}")


if __name__ == "__main__":
    main()
