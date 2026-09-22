# SAM 3.1 Text Prompt Tracker for ArcGIS Pro

Detect objects from text prompts and track them through local video using SAM 3.1
Object Multiplex. Supports multiple categories, stable external track IDs,
streaming video processing, CSV export, annotated MP4 and optional pixel-space
feature classes.

## Quick start

1. Use Windows with ArcGIS Pro 3.7, its matching Esri deep-learning libraries and
   a compatible NVIDIA CUDA GPU. See [Environment](ENVIRONMENT.md).
2. Download this repository using **Code > Download ZIP**, then extract the whole
   folder. Alternatively, clone the repository.
3. In ArcGIS Pro's Catalog pane, right-click **Toolboxes > Add Toolbox** and select
   **SAM31_TextPromptTracker.pyt** directly in the extracted folder.
4. Select an input video, enter a prompt such as `truck` or `car; swimming pool`,
   and choose an output folder.
5. Select the separately distributed compatible **SAM 3.1 DLPK** in the model
   parameter. Model weights and the DLPK are not included in this repository;
   obtain the companion package from the publisher before running the tool.
6. Start with **Short trial**, 90 frames, and a new run name. Inspect the CSV and
   annotated video before processing a longer video.

Keep `src/`, `SAM31_ObjectTracker/` and both XML help files beside the `.pyt`.
Do not move only the toolbox file. Restart ArcGIS Pro when switching releases.

## Model package

This toolbox requires the pinned SAM 3.1 Object Multiplex model described in
[VERSION.json](VERSION.json). Arbitrary SAM packages and fine-tuned weights are
not accepted. The DLPK must contain one EMD at its archive root, with
`model/sam3.1_multiplex.pt` and `model/bpe_simple_vocab_16e6.txt.gz` beneath it.
Model assets are validated and cached locally; the file selector does not upload
anything. Repeated runs in the same Python process reuse loaded models.

## Usage and limitations

See the [User guide](USER_GUIDE.md) for parameters, outputs and troubleshooting.

- Commas or semicolons separate object categories. More categories and simultaneous
  tracks can increase processing time and GPU memory use.
- Optional **Group Object Parts** uses geometric and temporal evidence to combine
  related tracks in outputs. It is a heuristic and may group incorrectly.
- Feature-class coordinates are video pixels, not geographic map coordinates.
- The source video is not modified. Annotated MP4 output does not retain KLV/MISB
  metadata and is not an FMV replacement.
- Identity quality, detector accuracy and long-video performance depend on the
  footage and require validation for the intended use.

This repository contains the custom text-prompt toolbox runtime. The companion
DLPK's native ArcGIS object-tracker adapter is distributed separately.

## Licensing

Original toolbox, integration and documentation contributions are licensed under
[CC BY-NC-SA 4.0](LICENSE): attribution is required, use must be noncommercial,
and shared adaptations must follow the license's ShareAlike terms.

Meta SAM components and the separately distributed model retain the
[SAM License](licenses/SAM_LICENSE). Adapted CLIP components retain their
[MIT notice](licenses/CLIP_LICENSE). These components are not relicensed under CC.
See [LICENSE_SCOPE.md](LICENSE_SCOPE.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the boundaries and attribution.
