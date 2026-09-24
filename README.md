# ConvX

ConvX is a fully convolutional explainable image-classification method. The network produces class scores and class-specific spatial heatmaps in one forward pass. ConvX+ adds optional pixel-level supervision from segmentation masks or sparse scribble annotations. Benchmarking results on the PASCAL VOC 2012 dataset achieved a mean Intersection over Union (mIoU) of 0.35 using only image-level labels in a flat, deterministic runtime. Furthermore, using just five ground-truth masks per class, our hybrid variant achieved an mIoU of 0.48, substantially outperforming Grad-CAM and variants while maintaining the same low latency as ConvX. We also demonstrate that with five sparse scribble annotations per class, ConvX+ achieved an mIoU of 0.45, offering an attractive balance between annotation effort and accuracy.

## Paper

If you make use of this code or the findings in your work, please cite the original conference paper:

#### Standard Text
> George, A., Mihaylova, L., Anderson, S. (2026). ConvX: Efficient XAI for Real-Time Computer Vision with Single Forward-Pass Explanations. In: Proceedings of the 3rd International Conference on Explainable AI for Neural and Symbolic Methods. Springer.

## Repository Layout

```text
ConvX/
├── src/convx/                            # Model, losses, datasets, metrics, and training engines
├── scripts/                              # Training, evaluation, data, and visualization entry points
├── data_sbd/                             # Local SBD dataset
├── data_voc/                             # Local PASCAL VOC dataset 
├── data_scribbles/                       # Local scribble annotations 
├── models/                               # Generated model checkpoints
├── plots/                                # Generated visualizations 
├── supervised_indices_5_per_class.json   # Pixel-level supervision indices
├── requirements.txt
└── pyproject.toml
```

## Setup

Open this folder (`ConvX`), then run these commands in the terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Dataset Preparation

Before training or generating scribbles, run the dataset preparation script from the repository root:

```powershell
python -m scripts.prepare_data
```

It downloads and validates the SBD `train_noval` segmentation split in `data_sbd` and the PASCAL VOC 2012 validation split in `data_voc`.

### ConvX+ Pixel-Level Supervision Subset

Both ConvX+ methods (Mask and Scribble) use pixel-level supervision from five training images per class, following the paper's default setting. The repository includes `supervised_indices_5_per_class.json`, which records the SBD training-set indices used for this subset so the same samples are selected consistently.

- **ConvX+ (Mask)** uses the pixel-level ground-truth masks supplied by SBD directly during training.
- **ConvX+ (Scribble)** uses sparse scribble annotations for the same selected training images. The supplied annotations are stored in `data_scribbles` as `scribble_<dataset-index>.npy` files. If the annotations are not present, generate them using the scribble annotator (refer Appendix).

## Training and Evaluation

Train the image-label-only ConvX model and evaluate it on the validation set:

```powershell
python -m scripts.train_convx
```

Train ConvX+ using dense SBD ground-truth masks for pixel-level supervision:

```powershell
python -m scripts.train_convx_plus_mask
```

Train ConvX+ using sparse scribble annotations for pixel-level supervision:

```powershell
python -m scripts.train_convx_plus_scribble
```

Train the standard classifier baseline for comparison:

```powershell
python -m scripts.train_classifier
```

Generate visual comparisons of model predictions and explanations. Uses Grad-CAM, Grad-CAM++, Eigen-CAM and Score-CAM.

```powershell
python -m scripts.visual_comparison
```

Compute explainability metrics for the trained model. Note that this takes time to complete.

```powershell
python -m scripts.run_explainability_metrics
```

Training and evaluation scripts write checkpoints beneath `models/` and generated artifacts beneath `plots/`.


## Appendix: Scribble Annotation Tutorial

1. Download and prepare the dataset as shown above.
2. Launch the interactive annotator:

   ```powershell
   python -m scripts.generate_scribbles
   ```

3. The annotator opens each image from the subset in `supervised_indices_5_per_class.json`. The left panel shows the original ground-truth segmentation and the right shows the original image; draw scribbles on either of the panels. Click and drag over object regions to mark sparse strokes. The strokes inherit their pixel labels from the ground-truth mask; pixels outside drawn strokes remain unannotated (label `255`).
4. Press **Space** or **Enter** to save the annotation and move to the next selected image. The annotator writes `data_scribbles/scribble_<dataset-index>.npy` and a visual preview in `scribble_previews/`. Existing annotations are loaded when revisiting an image, and saving replaces that image's annotation.
5. Press **S** to skip an image without changing its annotation, **C** or **R** to clear the current strokes, **+** or **-** to change the brush size, or **Q** / **Esc** to quit. You can rerun the annotator to continue; previously saved annotations are loaded automatically.
6. Once annotations are complete, train and evaluate the scribble-supervised ConvX+ model.

#### Annotating more than five images per class

1. Open `src/convx/config.py` and set `PIXEL_SUPERVISION_COUNT` to the desired count (for example, change `5` to `10`). Save the file before launching any scripts.
2. Ensure SBD is prepared, then run the annotator from the repository root. It will select the configured number of images per class and use or create the matching `supervised_indices_<count>_per_class.json` file.
3. Draw and save scribbles for the selected images using the controls above. Existing files in `data_scribbles` are loaded when present; save annotations for every selected index that does not already have one. The additional annotations are saved as `scribble_<dataset-index>.npy.
4. Train and evaluate the scribble-supervised ConvX+ model using the selected count.