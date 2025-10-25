# Veneer Visualization Prototype

This repository contains a lightweight Python script that generates a
consultation-ready veneer preview from a photo of a patient's current teeth.
The script relies on simple color heuristics to isolate the enamel and applies
an adjustable whitening veneer overlay.

## Requirements

- Python 3.9+
- [OpenCV](https://pypi.org/project/opencv-python/)
- [NumPy](https://numpy.org/)

You can install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python veneer_visualizer.py path/to/teeth_photo.jpg --side-by-side --output preview.png
```

Key options:

- `--whitening`: Intensity of the veneer whitening effect (0-1, default 0.45).
- `--brightness`: Additional brightness boost inside the veneer (0-1, default 0.15).
- `--smooth`: Gaussian blur sigma used to smooth the veneer shading (default 9.0).
- `--softness`: Softens the veneer mask edges for a natural transition (default 15).
- `--side-by-side`: Save a comparison of the original and veneer preview.

The command above saves a `preview.png` file containing a comparison between the
original image and the simulated veneer result.

## Notes

This is a heuristic prototype that works best on evenly lit photographs where
the teeth are visible. For clinical workflows, integrate with a higher-quality
segmentation model or manual masking step as needed.
