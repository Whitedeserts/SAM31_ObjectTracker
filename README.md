# SAM 3.1 Text Prompt Tracker for ArcGIS Pro

Detect objects from text prompts and track them through local video using SAM 3.1
Object Multiplex. Export tracking observations to CSV, annotated MP4 and optional
geospatial detection points from compatible FMV metadata.

## Requirements

Before using the toolbox, make sure the following are installed and available:

- **ArcGIS Pro on Windows** — [see software access options](https://learn.arcgis.com/en/become-a-member/).
- **ArcGIS Image Analyst** — required for the workflow, with a license available in ArcGIS Pro.
- **Deep Learning Libraries for ArcGIS Pro** — install the libraries that match your
  ArcGIS Pro version using [Esri's installation instructions](https://pro.arcgis.com/en/pro-app/latest/help/analysis/deep-learning/install-deep-learning-frameworks.htm).
- **A compatible NVIDIA CUDA GPU and driver** — recommended a minimum of 8 GB of dedicated memory.
- **A compatible SAM 3.1 DLPK** — obtained from ArcGIS Living Atlas.

> **Important:** Install the Deep Learning Libraries in the ArcGIS Pro Python
> environment you will use **before running the toolbox**. This project does not
> install ArcGIS Pro, extensions or Python dependencies automatically. Without the
> Deep Learning Libraries, model loading/inference will fail.

The documented tested setup uses **ArcGIS Pro 3.7 on Windows** with matching
Esri Deep Learning Libraries. If you're using a different version of ArcGIS Pro, you may encounter different functionality and results.
See the [user guide prerequisites](USER_GUIDE.md#prerequisites) for the recorded
Python/PyTorch/CUDA setup, memory considerations and FFmpeg/ffprobe requirements.

## Quick start

1. Complete the [requirements](#requirements), including the Deep Learning Libraries installation,
   before adding or running the toolbox.
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
- CSV boxes use video pixels. Optional map points require usable FMV metadata
  and assume flat terrain without terrain correction.
- Source video is unchanged; annotated MP4 does not retain KLV/MISB telemetry.
- Validate identity quality and detection accuracy on representative footage.

The native ArcGIS object-tracker adapter is distributed separately in the DLPK.

## License

Independently authored toolbox contributions use [Apache 2.0](licenses/APACHE_LICENSE).
Meta SAM and CLIP components retain their own terms. See
[license scope and third-party notices](licenses/THIRD_PARTY_NOTICES.md) and
[NOTICE](licenses/NOTICE).

Developed by [Mohamed Ahmed](https://mohamedahmed.ca), Esri Canada Education and Research Group.
