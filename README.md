# SAM 3.1 Text Prompt Tracker for ArcGIS Pro

Detect objects from text prompts and track them through local video using SAM 3.1
Object Multiplex. Export tracking observations to CSV, annotated MP4 and optional
pixel-space features.

## Quick start

1. Use ArcGIS Pro 3.7 on Windows, matching Esri deep-learning libraries and a
   compatible NVIDIA CUDA GPU. See [prerequisites](USER_GUIDE.md#prerequisites).
2. Select **Code > Download ZIP** and extract the complete repository, or clone it.
3. In ArcGIS Pro, choose **Catalog > Toolboxes > Add Toolbox** and select
   **SAM31_TextPromptTracker.pyt** from the extracted folder.
4. Select an input video, enter a prompt such as `truck` or `car; swimming pool`,
   and choose an output folder. Commas and semicolons separate categories.
5. Select the separately distributed compatible **SAM 3.1 DLPK**. Obtain the
   companion package from the publisher; model weights are not included here.
6. Start with **Short trial**, 90 frames, and a new run name. Review the outputs
   before processing a longer video.

Keep `src/`, `SAM31_ObjectTracker/` and both XML help files beside the `.pyt`.
Restart ArcGIS Pro when switching releases. The [user guide](USER_GUIDE.md) covers
model requirements, parameters, outputs and troubleshooting.

## Important limitations

- More objects or text categories can increase processing time and GPU memory use.
- Optional **Group Object Parts** is heuristic and can group incorrectly.
- Toolbox feature coordinates are video pixels, not geographic map locations.
- Source video is unchanged; annotated MP4 does not retain KLV/MISB telemetry.
- Validate identity quality and detection accuracy on representative footage.

The native ArcGIS object-tracker adapter is distributed separately in the DLPK.

## License

Independently authored toolbox contributions use [Apache 2.0](licenses/APACHE_LICENSE).
Meta SAM and CLIP components retain their own terms. See
[license scope and third-party notices](licenses/THIRD_PARTY_NOTICES.md) and
[NOTICE](licenses/NOTICE).

Developed by [Mohamed Ahmed](https://mohamedahmed.ca).
