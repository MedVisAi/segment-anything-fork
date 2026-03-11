# Segment Anything

A foundation model for image segmentation, trained on a large dataset and designed to work with diverse prompts.

## Overview

**Segment Anything** (SAM) is a foundation model for image segmentation that can segment any object in an image given appropriate prompts. The model is trained on a diverse dataset of images and can generalize well to new domains without additional training. 

SAM addresses the segmentation task by predicting segmentation masks given:
- **Point prompts**: Click on objects to segment
- **Box prompts**: Provide bounding boxes
- **Automatic prompts**: Generate masks for all objects in an image

The model consists of three main components:
- **Image Encoder**: A Vision Transformer-based encoder that processes the input image
- **Prompt Encoder**: Encodes various types of prompts (points, boxes, etc.)
- **Mask Decoder**: Efficiently decodes masks from image and prompt embeddings

## Model Architecture

The SAM architecture is based on an image encoder-decoder design with prompt encoding:

```
┌─────────────────┐
│  Input Image    │
└────────┬────────┘
         │
    ┌────▼─────────────────┐
    │  Image Encoder (ViT) │
    │  1024×1024 input     │
    │  256D embeddings     │
    └────┬─────────────────┘
         │
    ┌────┴──────────────────────┐
    │                           │
    │  ┌──────────────────────┐ │
    │  │  Prompt Encoder      │ │
    │  │  (Points/Boxes/Text) │ │
    │  └──────────┬───────────┘ │
    │             │             │
    │        ┌────▼─────────────┤
    │        │  Mask Decoder    │
    │        │  (Transformer)   │
    │        └────┬─────────────┘
    │             │
    └─────────────┼─────────────┘
         │        │
         │   ┌────▼──────────────┐
         │   │  Output Masks     │
         │   │  + Confidence     │
         │   └───────────────────┘
```

### Model Variants

Three pre-trained model sizes are available:

| Model | Encoder Depth | Encoder Width | Memory | Speed |
|-------|---------------|---------------|--------|-------|
| ViT-H (default) | 32 | 1280 | ~2.6GB | Slower |
| ViT-L | 24 | 1024 | ~2.0GB | Medium |
| ViT-B | 12 | 768 | ~1.1GB | Faster |

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.7+
- torchvision

### Basic Installation

```bash
pip install -e .
```

### With Optional Dependencies

For full functionality including ONNX export and the command-line tools:

```bash
pip install -e ".[all]"
```

### Optional Dependencies

- `opencv-python`: For image processing  
- `pycocotools`: For COCO format output in batch processing
- `onnx`, `onnxruntime`: For ONNX model export and inference
- `matplotlib`: For visualization

**Development dependencies** (for contributing):

```bash
pip install -e ".[dev]"
```

Includes: `black`, `isort`, `flake8`, `mypy`

## Quick Start

### 1. Prompt-based Segmentation (SamPredictor)

Run inference with prompts on a single image:

```python
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

# Load model
sam_checkpoint = "path/to/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Load image
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set image embedding
predictor.set_image(image)

# Predict mask from point prompt
points = [[510, 373], [637, 375]]  # x, y coordinates
labels = [1, 1]  # 1 = include, 0 = exclude
masks, scores, logits = predictor.predict(
    point_coords=points,
    point_labels=labels,
)

# Or predict from box prompt
box = [384, 145, 1000, 625]  # [x0, y0, x1, y1]
masks, scores, logits = predictor.predict(box=box)
```

### 2. Automatic Mask Generation (SamAutomaticMaskGenerator)

Generate masks for all objects in an image automatically:

```python
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = mask_generator.generate(image)

# Each mask dict contains:
# - 'segmentation': RLE encoded mask
# - 'area': mask area in pixels
# - 'bbox': bounding box [x, y, w, h]
# - 'predicted_iou': predicted IoU
# - 'stability_score': stability score
# - 'crop_box': crop box used to generate mask
```

## Usage

### Command-line Tools

#### Batch Mask Generation

Run automatic segmentation on images:

```bash
python scripts/amg.py \
  --checkpoint sam_vit_h_4b8939.pth \
  --input image.jpg \
  --output output_dir \
  --model-type vit_h
```

Options:
- `--input`: Image file or directory with images
- `--output`: Output directory  
- `--model-type`: Model size (`vit_h`, `vit_l`, or `vit_b`)
- `--checkpoint`: Path to model checkpoint
- `--output-mode`: Save format (`coco_rle` or `binary_mask`)
- `--convert-to-rle`: Convert masks to COCO format

#### ONNX Model Export

Export models for deployment in web applications or with ONNX runtime:

```bash
python scripts/export_onnx_model.py \
  --checkpoint sam_vit_h_4b8939.pth \
  --output sam_onnx \
  --model-type vit_h
```

This exports the prompt encoder and mask decoder as ONNX models that can run in the browser with WebAssembly.

## Examples

See the `notebooks/` directory for detailed examples:

- **[predictor_example.ipynb](notebooks/predictor_example.ipynb)**: Interactive segmentation with point and box prompts
- **[automatic_mask_generator_example.ipynb](notebooks/automatic_mask_generator_example.ipynb)**: Automatic mask generation for full images
- **[onnx_model_example.ipynb](notebooks/onnx_model_example.ipynb)**: Using ONNX-exported models

## Web Demo

A front-end only web demo is available in the `demo/` directory. It loads images and runs segmentation in the browser using ONNX with WebAssembly and multi-threading support.

To run locally:

```bash
cd demo
yarn install
yarn start
```

The demo loads pre-computed image embeddings (`.npy` files) to enable fast inference in the browser.

## Model Checkpoints

Pre-trained model weights are required to use SAM. Download them from the official SAM repository or specify the checkpoint path when loading:

```python
from segment_anything import sam_model_registry

# Automatically downloads checkpoint if needed
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
```

Available models:
- `vit_h`: Largest model, best quality
- `vit_l`: Large model, good balance  
- `vit_b`: Base model, fastest inference

## API Reference

### SamPredictor

```python
class SamPredictor:
    def __init__(self, sam_model: Sam)
    def set_image(self, image: np.ndarray, image_format: str = "RGB")
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    def reset_image()
```

**Returns:**
- `masks`: Array of predicted masks (num_masks, height, width)
- `iou_predictions`: Predicted intersection-over-union of masks
- `low_res_logits`: Low resolution logits for mask refinement

### SamAutomaticMaskGenerator

```python
class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        ...
    )
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]
```

**Returns:** List of mask dictionaries with keys:
- `segmentation`: Mask in RLE format
- `area`: Mask area in pixels
- `bbox`: Bounding box `[x, y, width, height]`
- `predicted_iou`: Predicted IoU quality score
- `stability_score`: Stability score
- `crop_box`: Source crop region

## Performance

SAM achieves strong zero-shot performance on diverse segmentation tasks without task-specific training. Performance varies by model size and task:

| Model | Speed (FPS) | Memory | Output Quality |
|-------|-------------|--------|-----------------|
| ViT-H | ~5-10 | ~2.6GB | Highest |
| ViT-L | ~10-15 | ~2.0GB | High |
| ViT-B | ~20-30 | ~1.1GB | Good |

Inference speed depends on image size, hardware, and prompt type.

## License

The model code is licensed under the Apache 2.0 license. See the [LICENSE](LICENSE) file for details.

## Citation

If you use SAM in your research, please cite it as:

```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Darrell, Trevor and Dollár, Piotr},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Submitting pull requests
- Code style and linting
- Testing requirements
- License agreement

Code style is enforced using `black`, `isort`, and `flake8`. Run the linter with:

```bash
bash linter.sh
```

## Acknowledgements

**Segment Anything** is built upon:
- Vision Transformer (ViT) architecture from the PyTorch and Meta AI communities
- Inspired by foundation models in computer vision
- Trained on a diverse, large-scale dataset of images

## Related Resources

- [Official SAM Project Page](https://segment-anything.com/)
- [SAM Paper](https://arxiv.org/abs/2304.02643)
- [Meta AI Blog Post](https://ai.facebook.com/research/segment-anything/)

---

**Meta Platforms, Inc.** © 2023. All rights reserved.
