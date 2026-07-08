# Modified for PS-MOT (ECCV 2026), CVPU Kai Luo. All Rights Reserved.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
except ImportError:
    Image = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_SPLIT = "train"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "sam3.pt"
DEFAULT_PSEUDO_LABEL_FILE = "sam3_pgt.txt"
DEFAULT_NOISE_SIGMA = 48.0
DEFAULT_NEGATIVE_RADIUS = 40.0
DEFAULT_RETRY_OFFSETS = ((-4.0, -4.0), (4.0, 4.0), (0.0, -6.0), (0.0, 6.0))
DEFAULT_RETRY_SCORE_THRESHOLD = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MOT-style pseudo boxes from point prompts with SAM 3."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root. It can point to the dataset directory or directly to a split directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset folder name under data-root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help="Dataset split name used when data-root points to the dataset root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the SAM 3 checkpoint.",
    )
    parser.add_argument(
        "--pseudo-label-file",
        type=str,
        default=DEFAULT_PSEUDO_LABEL_FILE,
        help="Name of the pseudo-label file written under each sequence gt folder.",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=DEFAULT_NOISE_SIGMA,
        help="Standard deviation of Gaussian noise applied to the prompt center.",
    )
    parser.add_argument(
        "--negative-radius",
        type=float,
        default=DEFAULT_NEGATIVE_RADIUS,
        help="Maximum distance for selecting negative prompts.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence names to process.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process per sequence.",
    )
    return parser.parse_args()


def is_sequence_dir(path: Path) -> bool:
    return (path / "gt" / "gt.txt").is_file() and (path / "img1").is_dir()


def resolve_split_roots(data_root: Path, dataset: str | None, split: str) -> list[Path]:
    """Resolve one or more directories that contain sequence folders."""
    if dataset:
        return [data_root / dataset / split]

    if (data_root / split).is_dir():
        return [data_root / split]

    if data_root.is_dir() and any(is_sequence_dir(path) for path in data_root.iterdir() if path.is_dir()):
        return [data_root]

    if data_root.is_dir():
        split_roots = [
            path / split
            for path in sorted(data_root.iterdir())
            if path.is_dir() and (path / split).is_dir()
        ]
        if split_roots:
            return split_roots

    return [data_root]


def load_mot_gt_with_ids(gt_path: Path) -> dict[int, list[dict[str, object]]]:
    """Load MOT annotations from a GT file and group them by frame id."""
    gt_dict: dict[int, list[dict[str, object]]] = {}
    if not gt_path.exists():
        return gt_dict

    with gt_path.open("r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue

            frame_id = int(parts[0])
            track_id = int(parts[1])
            box = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            gt_dict.setdefault(frame_id, []).append({"id": track_id, "box": box})

    return gt_dict


def is_torch_tensor(value: object) -> bool:
    return torch is not None and torch.is_tensor(value)


def masks_to_bboxes(masks: np.ndarray | torch.Tensor) -> list[list[float] | None]:
    """Convert binary masks to [x1, y1, x2, y2] boxes."""
    if is_torch_tensor(masks):
        masks = masks.cpu().numpy()

    bboxes: list[list[float] | None] = []
    for mask in masks:
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            bboxes.append(None)
            continue

        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        bboxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])

    return bboxes


def normalize_sam_output(
    masks: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize SAM output so masks are always (N, H, W) and scores are always (N,)."""
    if is_torch_tensor(masks):
        masks = masks.cpu().numpy()
    if is_torch_tensor(scores):
        scores = scores.cpu().numpy()

    score_array = np.atleast_1d(np.squeeze(scores))
    mask_array = np.squeeze(masks)

    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    elif mask_array.ndim == 3 and len(score_array) == 1 and mask_array.shape[0] != 1:
        mask_array = mask_array[None, ...]

    return mask_array, score_array


class ShapeStabilityManager:
    """Track per-target area and aspect-ratio history for lightweight sanity checks."""

    def __init__(self, window_size: int = 5) -> None:
        self.history: dict[int, dict[str, list[float]]] = {}
        self.window_size = window_size

    def update(self, track_id: int, box: list[float] | None) -> None:
        if box is None:
            return

        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height
        ratio = height / (width + 1e-6)

        history = self.history.setdefault(track_id, {"areas": [], "ratios": []})
        history["areas"].append(area)
        history["ratios"].append(ratio)

        if len(history["areas"]) > self.window_size:
            history["areas"].pop(0)
            history["ratios"].pop(0)

    def get_shape_score(self, track_id: int, box: list[float]) -> float:
        """Return a score in [0, 1] based on the aspect-ratio consistency."""
        history = self.history.get(track_id)
        if not history or not history["ratios"]:
            return 1.0

        width = box[2] - box[0]
        height = box[3] - box[1]
        current_ratio = height / (width + 1e-6)
        average_ratio = float(np.mean(history["ratios"]))
        return float(np.exp(-abs(current_ratio - average_ratio) / 0.8))

    def is_anomaly(self, track_id: int, box: list[float]) -> bool:
        """Detect abrupt area changes for the current track."""
        history = self.history.get(track_id)
        if not history or len(history["areas"]) < 3:
            return False

        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height
        average_area = float(np.mean(history["areas"]))
        return area > average_area * 2.0 or area < average_area * 0.4


def list_sequence_dirs(split_root: Path, sequence_names: list[str] | None = None) -> list[Path]:
    sequence_filter = set(sequence_names) if sequence_names else None
    sequence_dirs = sorted([
        path
        for path in split_root.iterdir()
        if path.is_dir() and (sequence_filter is None or path.name in sequence_filter)
    ])
    if sequence_filter is not None:
        found = {path.name for path in sequence_dirs}
        missing = sorted(sequence_filter - found)
        if missing:
            raise FileNotFoundError(f"Sequences not found under {split_root}: {', '.join(missing)}")
    return sequence_dirs


def get_frame_paths(img_dir: Path) -> list[Path]:
    frame_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    return sorted(frame_paths)


def build_prompt_arrays(
    center: np.ndarray,
    centers: dict[int, list[float]],
    track_id: int,
    noise_sigma: float,
    negative_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    dx = np.random.normal(0.0, noise_sigma)
    dy = np.random.normal(0.0, noise_sigma)
    prompt_center = [float(center[0] + dx), float(center[1] + dy)]

    negative_points = [
        other_center
        for other_id, other_center in centers.items()
        if other_id != track_id and np.linalg.norm(center - np.array(other_center)) < negative_radius
    ]

    point_coords = np.array([[prompt_center] + negative_points], dtype=np.float32)
    point_labels = np.array([[1] + [0] * len(negative_points)], dtype=np.int32)
    return point_coords, point_labels


def select_box_from_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    track_id: int,
    shape_manager: ShapeStabilityManager,
) -> tuple[list[float] | None, float, float]:
    best_box: list[float] | None = None
    best_combined_score = -1.0
    best_raw_score = 0.0

    boxes = masks_to_bboxes(masks)
    for idx, score in enumerate(scores):
        if idx >= len(boxes) or boxes[idx] is None:
            continue

        shape_score = shape_manager.get_shape_score(track_id, boxes[idx])
        combined_score = float(score) * 0.6 + shape_score * 0.4
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_box = boxes[idx]
            best_raw_score = float(score)

    return best_box, best_raw_score, best_combined_score


def generate_sequence_pseudo_labels(
    seq_dir: Path,
    model: Any,
    processor: Any,
    shape_manager: ShapeStabilityManager,
    pseudo_label_file: str,
    noise_sigma: float,
    negative_radius: float,
    max_frames: int | None = None,
) -> None:
    gt_path = seq_dir / "gt" / "gt.txt"
    img_dir = seq_dir / "img1"
    output_path = seq_dir / "gt" / pseudo_label_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not gt_path.exists() or not img_dir.exists():
        return

    gt_data = load_mot_gt_with_ids(gt_path)
    frame_paths = get_frame_paths(img_dir)
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    with output_path.open("w") as f_out:
        for frame_path in tqdm(frame_paths, desc=f"Processing {seq_dir.name}"):
            frame_id = int(frame_path.stem)
            objects = gt_data.get(frame_id, [])
            if not objects:
                continue

            image = Image.open(frame_path).convert("RGB")
            inference_state = processor.set_image(image)

            centers = {
                int(obj["id"]): [obj["box"][0] + obj["box"][2] / 2.0, obj["box"][1] + obj["box"][3] / 2.0]
                for obj in objects
            }

            for obj in objects:
                track_id = int(obj["id"])
                bx, by, bw, bh = map(float, obj["box"])
                center = np.array([bx + bw / 2.0, by + bh / 2.0])

                point_coords, point_labels = build_prompt_arrays(
                    center=center,
                    centers=centers,
                    track_id=track_id,
                    noise_sigma=noise_sigma,
                    negative_radius=negative_radius,
                )

                try:
                    masks_raw, scores_raw, _ = model.predict_inst(
                        inference_state,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                    )
                    masks, scores = normalize_sam_output(masks_raw, scores_raw)
                except Exception:
                    continue

                best_box, best_raw_score, best_combined_score = select_box_from_masks(
                    masks=masks,
                    scores=scores,
                    track_id=track_id,
                    shape_manager=shape_manager,
                )

                if best_box is None or shape_manager.is_anomaly(track_id, best_box):
                    for offset_x, offset_y in DEFAULT_RETRY_OFFSETS:
                        retry_point_coords = np.array(
                            [[[center[0] + offset_x, center[1] + offset_y]]], dtype=np.float32
                        )
                        try:
                            retry_masks_raw, retry_scores_raw, _ = model.predict_inst(
                                inference_state,
                                point_coords=retry_point_coords,
                                point_labels=np.array([[1]]),
                                multimask_output=True,
                            )
                            retry_masks, retry_scores = normalize_sam_output(retry_masks_raw, retry_scores_raw)
                            retry_boxes = masks_to_bboxes(retry_masks)
                            for retry_idx, retry_score in enumerate(retry_scores):
                                if retry_idx >= len(retry_boxes) or retry_boxes[retry_idx] is None:
                                    continue
                                retry_box = retry_boxes[retry_idx]
                                retry_combined_score = float(retry_score) * shape_manager.get_shape_score(
                                    track_id, retry_box
                                )
                                if retry_combined_score > best_combined_score:
                                    best_combined_score = retry_combined_score
                                    best_box = retry_box
                                    best_raw_score = float(retry_score)
                            if best_combined_score > DEFAULT_RETRY_SCORE_THRESHOLD:
                                break
                        except Exception:
                            continue

                if best_box is None:
                    f_out.write(f"{frame_id},{track_id},{bx:.2f},{by:.2f},{bw:.2f},{bh:.2f},1,1,0.0100\n")
                    continue

                shape_manager.update(track_id, best_box)
                x1, y1, x2, y2 = best_box
                final_quality = best_raw_score * shape_manager.get_shape_score(track_id, best_box)
                f_out.write(
                    f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,1,{final_quality:.4f}\n"
                )


def build_sam3_processor(checkpoint: Path, device: torch.device) -> tuple[Any, Any]:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        enable_inst_interactivity=True,
    ).to(device)
    return model, Sam3Processor(model)


def main() -> None:
    args = parse_args()
    split_roots = resolve_split_roots(args.data_root, args.dataset, args.split)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be a positive integer.")
    if torch is None:
        raise ImportError("PyTorch is required to run SAM 3. Please use an environment with torch installed.")
    if Image is None:
        raise ImportError("Pillow is required to load images. Please install pillow in the current environment.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading SAM 3 model...")
    model, processor = build_sam3_processor(args.checkpoint, device)

    missing_roots = [split_root for split_root in split_roots if not split_root.exists()]
    if missing_roots:
        missing = ", ".join(str(path) for path in missing_roots)
        raise FileNotFoundError(f"Split root does not exist: {missing}")

    shape_manager = ShapeStabilityManager()
    for split_root in split_roots:
        for seq_dir in list_sequence_dirs(split_root, sequence_names=args.sequences):
            generate_sequence_pseudo_labels(
                seq_dir=seq_dir,
                model=model,
                processor=processor,
                shape_manager=shape_manager,
                pseudo_label_file=args.pseudo_label_file,
                noise_sigma=args.noise_sigma,
                negative_radius=args.negative_radius,
                max_frames=args.max_frames,
            )

    print(f"\n[Done] Pseudo labels saved as {args.pseudo_label_file} in each sequence gt folder.")


if __name__ == "__main__":
    main()
