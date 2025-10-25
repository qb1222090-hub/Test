"""Generate a basic veneer visualization from a patient's teeth image.

This script performs a lightweight color-based segmentation of the teeth
region and applies a whitening veneer effect.  The goal is to provide a quick
approximation that can be reviewed during consultations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VeneerParameters:
    """Parameters that control the veneer simulation."""

    whitening: float = 0.45
    brightness_boost: float = 0.15
    smooth_sigma: float = 9.0
    mask_softness: int = 15

    def clamp(self) -> "VeneerParameters":
        """Ensure parameter values remain within reasonable bounds."""

        self.whitening = float(np.clip(self.whitening, 0.0, 1.0))
        self.brightness_boost = float(np.clip(self.brightness_boost, 0.0, 1.0))
        self.smooth_sigma = float(np.clip(self.smooth_sigma, 0.0, 25.0))
        self.mask_softness = int(max(0, self.mask_softness))
        return self


def read_image(path: Path) -> np.ndarray:
    """Read an image from disk and return it in BGR order."""

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image from '{path}'.")
    return image


def detect_teeth_mask(image: np.ndarray) -> np.ndarray:
    """Return a soft mask that roughly highlights teeth.

    The routine converts the image to HSV color space and selects bright pixels
    with low saturation – a heuristic that matches tooth enamel reasonably well
    for consultation imagery.  The mask is refined with morphological closing
    and optional Gaussian smoothing.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 160])
    upper = np.array([180, 70, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (0, 0), 3.0)

    mask = mask.astype(np.float32) / 255.0
    return mask


def apply_veneer_effect(
    image: np.ndarray, mask: np.ndarray, params: VeneerParameters
) -> np.ndarray:
    """Apply a whitening veneer effect to the masked region of the image."""

    params = params.clamp()
    image_float = image.astype(np.float32) / 255.0

    # Base veneer tone: slightly warm while remaining bright.
    veneer_tone = np.array([1.0, 0.98, 0.92], dtype=np.float32)

    mask_soft = mask
    if params.mask_softness > 0:
        mask_soft = cv2.GaussianBlur(
            mask, (0, 0), params.mask_softness, borderType=cv2.BORDER_REPLICATE
        )
        mask_soft = np.clip(mask_soft, 0.0, 1.0)

    # Whitening effect by mixing with veneer tone.
    veneer_overlay = image_float * (1.0 - params.whitening) + veneer_tone * params.whitening

    # Brightness boost applied only to masked region.
    boosted = veneer_overlay + params.brightness_boost
    boosted = np.clip(boosted, 0.0, 1.0)

    # Combine original and veneered areas with a smooth mask.
    mask_soft = mask_soft[..., None]
    veneered = image_float * (1.0 - mask_soft) + boosted * mask_soft

    if params.smooth_sigma > 0:
        smooth = cv2.GaussianBlur(
            veneered, (0, 0), params.smooth_sigma, borderType=cv2.BORDER_REPLICATE
        )
        detail_mask = 1.0 - mask_soft
        veneered = smooth * mask_soft + veneered * detail_mask

    veneered = np.clip(veneered * 255.0, 0, 255).astype(np.uint8)
    return veneered


def create_side_by_side(original: np.ndarray, veneered: np.ndarray) -> np.ndarray:
    """Return a comparison image with original and veneered results."""

    height = max(original.shape[0], veneered.shape[0])
    width = original.shape[1] + veneered.shape[1]
    result = np.zeros((height, width, 3), dtype=np.uint8)

    result[: original.shape[0], : original.shape[1]] = original
    result[: veneered.shape[0], original.shape[1] : width] = veneered

    cv2.putText(
        result,
        "Original",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        "Veneer Preview",
        (original.shape[1] + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = VeneerParameters()
    parser.add_argument("image", type=Path, help="Input image showing the patient's teeth")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("veneer_preview.png"),
        help="Path where the veneer preview image should be saved.",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Save a side-by-side comparison instead of only the veneered image.",
    )
    parser.add_argument(
        "--whitening",
        type=float,
        default=defaults.whitening,
        help="Intensity of the whitening effect (0-1).",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=defaults.brightness_boost,
        help="Additional brightness applied to the veneer (0-1).",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=defaults.smooth_sigma,
        help="Gaussian blur sigma to smooth the veneer (0 disables smoothing).",
    )
    parser.add_argument(
        "--softness",
        type=int,
        default=defaults.mask_softness,
        help="Gaussian blur sigma for softening the veneer mask edge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    params = VeneerParameters(
        whitening=args.whitening,
        brightness_boost=args.brightness,
        smooth_sigma=args.smooth,
        mask_softness=args.softness,
    )

    original = read_image(args.image)
    mask = detect_teeth_mask(original)
    veneered = apply_veneer_effect(original, mask, params)

    if args.side_by_side:
        preview = create_side_by_side(original, veneered)
    else:
        preview = veneered

    cv2.imwrite(str(args.output), preview)
    print(f"Saved veneer preview to {args.output}")


if __name__ == "__main__":
    main()
