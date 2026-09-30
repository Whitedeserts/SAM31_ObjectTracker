# User guide

## Prerequisites

**ArcGIS Pro on Windows and ArcGIS Image Analyst are required for this workflow.**
See [ArcGIS software access options](https://learn.arcgis.com/en/become-a-member/)
and ensure an Image Analyst license is available in ArcGIS Pro.

**Install the Deep Learning Libraries for ArcGIS Pro before using the toolbox.**
Follow [Esri's official installation instructions](https://pro.arcgis.com/en/pro-app/latest/help/analysis/deep-learning/install-deep-learning-frameworks.htm)
and use the libraries matching your ArcGIS Pro version in the ArcGIS Pro Python
environment selected for running the tool. The toolbox does not install these
dependencies automatically. Without them, model loading/inference will fail.

The tested environment is Windows with ArcGIS Pro 3.7, its matching 3.7
deep-learning libraries, Python 3.13, PyTorch 2.9.1 with CUDA 12.9 and
torchvision 0.25.0. Use the matching Esri-supported deep-learning installation
for Pro; the tool does not install dependencies or modify Conda environments.
These package versions describe the recorded test environment, not a request to
install or upgrade PyTorch/CUDA independently of Esri's instructions. Other versions
have not been validated.

A compatible NVIDIA CUDA GPU and driver are required. CPU-only inference is
unsupported. Testing used approximately 12 GB of GPU memory capacity; this is not a verified
minimum. Memory needs depend on frames and active objects. Close other GPU-heavy applications.
Provide system RAM and disk space for model loading, extracted assets and output
tables. One model cache uses approximately 3.3 GiB, plus temporary extraction
space; damaged caches are preserved until manually removed.

Required libraries include arcpy, torch, torchvision, numpy, pandas, scipy,
Pillow, OpenCV, timm, ftfy, regex, iopath, tqdm, einops, huggingface_hub and
psutil for memory diagnostics. FFmpeg/ffprobe are resolved from the active
environment or PATH. No inference-time downloads are performed.

ArcGIS manages this environment; no lockfile is supplied. Avoid general dependency
upgrades in the ArcGIS environment. Optional Triton/FlashAttention paths are guarded
on Windows. Required external dependencies are not bundled or installed by the tool.

## Installation and model selection

Extract the entire ZIP; folder names may contain spaces. In Catalog, right-click
Toolboxes, choose Add Toolbox, and select `SAM31_TextPromptTracker.pyt`.
Do not copy the `.pyt` alone. The release includes the project runtime source,
but still requires the external ArcGIS software and Deep Learning Libraries
listed above. The original development repository and custom PYTHONPATH are unnecessary.

Select **SAM 3.1 Model Package (.dlpk)**. This is a local file selector, not an
upload. This release accepts the pinned SAM 3.1 Object Multiplex checkpoint and
CLIP vocabulary verified by `src/model_package.py`. Fine-tuned or other SAM versions
are rejected rather than loaded with an incompatible architecture. The EMD must
be at the archive root and reference `model/sam3.1_multiplex.pt`. The vocabulary
must be `model/bpe_simple_vocab_16e6.txt.gz`. `config.json` is provenance only;
the pinned builder defines the runtime architecture.

First run: the tool validates the archive, hashes it, and extracts only the
checkpoint and vocabulary into `%LOCALAPPDATA%/SAM31_TextPromptTracker/models/`.
Files are written to a temporary directory and published only after hash/CRC
validation. Existing DLPKs are never modified. Normal inference is offline.

Each package's SHA256 identifies its cache. Renaming an unchanged package reuses
the cache; changed package content selects another cache. Cache files are
rehashed before reuse, adding disk-read time but avoiding repeated extraction.
Incomplete caches are ignored and damaged caches are quarantined under `.invalid-*`.
After closing Pro, those directories and abandoned `.preparing-*` directories may
be removed manually to recover space. Never delete model assets during a run.

The detector and tracker use the same selected assets. Repeated runs in one
Python process reuse loaded models. Switching package fingerprints clears both
model caches before loading. Restart Pro when switching installed runtime releases
or after a CUDA error. Simultaneous runs sharing one Python process are unsupported.

## Tracking stability

New tracks appear after three consecutive valid tracking observations following
their initial seed (five while touching a frame edge). Tentative tracks are
recorded as `TENTATIVE` with null coordinates and use normal tracker capacity.
They expire after ten frames without confirmation. Lost tracks require two valid
observations before boxes reappear. The existing lost/out-of-frame grace controls
continue to govern termination; terminated IDs are not reused.

Mask overlap and containment reject redundant detections. Detector masks seed
the tracker directly. Substantial nearby components within one seeded mask may
contribute to one box; neighboring independent masks are not joined by proximity.
Near-identical live tracks require fifteen consecutive frames of strong overlap
before the newer/unconfirmed duplicate is terminated.

Observation checks reject implausible jumps using the last thirty accepted boxes.
A fresh strongly matched text detection can corroborate a large zoom or recovery.
Abrupt camera changes can therefore temporarily hide a box until re-detection.
These are conservative heuristics: physical tractor/trailer attachment cannot be
guaranteed from masks alone, and sustained vehicle overlap can remain ambiguous.
Coordinates are not smoothed or filled across missing observations.

Confirmation requires tracker confidence of at least 0.50, or a higher configured
minimum. The mask-area continuation threshold is 75% of the configured minimum;
new/recovering observations use the full minimum. These stability defaults are
internal settings in `src/track_state.py`; the toolbox parameter layout is unchanged.

## Inputs and controls

- **Input Video:** local MP4, MOV, AVI, MKV, TS/M2TS, MPEG/PS/VOB, WMV,
  raw H.264/H.265, or local HLS/DASH with its segment files. Actual codec support
  depends on the decoder. True WMV3/VC-1 and discontinuous MISB footage need validation.
- **Text Prompt:** describe the object class, for example `truck` or `person`.
- **Output Folder / Run Name:** generate output paths; optional overrides retain
  custom destinations. Use a new run name to preserve prior results.
- **Short trial:** first 90 frames by default. Cold detector preparation may
  exceed three minutes even for a short clip; this is separate from processing.
- **Detection Interval:** frames between searches for new objects (default 30).
  Existing tracks advance every frame. Larger values delay new-object discovery.
- **Confidence:** detection/tracking threshold; tune with a short representative clip.
- **Max Simultaneous Objects:** maximum live tracks, independent of lifetime ID count.
  Default: 16; higher values are allowed and use additional fixed-capacity SAM sessions.
  Higher limits use more GPU memory and may slow processing. Reduce this value if
  GPU memory runs out or processing is too slow. Lost tracks within their grace
  period and grouped member tracks still count. Try a short trial before a long run;
  a suitable limit depends on the GPU and footage.
- **Lost Grace / Out-of-Frame Grace:** defaults 30/10 frames. IDs and SAM memory
  remain during grace; recovery is possible but not guaranteed. After termination,
  a newly detected object receives a new ID. Camera motion can affect identity quality.

The batch tracker uses bounded SAM sessions sharing a tracker model and one
backbone evaluation per frame. Removed multiplex slots are not reused as fresh
slots; new objects use unused slots or a new session. Surviving memory stays intact.
Fragmented sessions can increase multi-object processing time.

## Outputs

CSV and annotated MP4 are enabled by default. CSV retains track IDs, confidence,
status, frame/time references and pixel bounding-box coordinates. Lost/terminated
observations have null coordinates. A separate pixel-space feature class is no
longer offered in the toolbox because its attributes are already available in CSV.
Optional geospatial detection points provide estimated map locations when usable
FMV metadata and frame timing are available; see the geographic-output section.

Annotated video is a derived visual review product without KLV/MISB or original
telemetry. It is not FMV-compliant. The source remains read-only. `source_video`
and `source_timestamp` support later association with metadata, subject to decoder
timestamp limitations. Optional geographic export can recover missing frame timestamps.

An export error names the failed output; independently completed outputs remain.
A failed output file may be incomplete. Use a new run name after correcting it.

## Troubleshooting

- **Invalid/incompatible DLPK:** obtain the complete compatible package. Do not
  rename a different model or edit hashes to bypass validation.
- **Missing dependency:** activate the supported Pro deep-learning environment;
  install the matching Esri libraries using the supported installation procedure.
- **CUDA unavailable/error:** check the GPU, driver and environment. Restart Pro
  after a CUDA failure before retrying.
- **Insufficient VRAM / severe slowdown:** close other GPU workloads, reduce
  simultaneous objects, and compare a short trial. Do not disable model residency handling.
- **Cache error:** check free space and write permission to the per-user cache.
  Close Pro before removing damaged/incomplete cache directories; retry preparation.
- **Video cannot open:** confirm a local complete file and supported codec.
  Keep original FMV intact when making a separate conversion for diagnosis.
- **Output write failure:** choose a writable folder and new name; release table
  or video locks. The tool reports outputs that succeeded independently.
- **Updated toolbox does not refresh:** restart Pro or remove/re-add it. Positional
  scripts/ModelBuilder must use the current input order: video, text prompt,
  output folder, model package, then the remaining parameters.

## Manual validation on another computer

1. Extract the ZIP into a folder containing spaces. Add the toolbox in a fresh
   Pro process using the supported environment, without the development repository.
2. Select the companion DLPK, a short representative video, prompt `truck`, a
   writable output folder and run name `Trial_A`. Choose Short trial, 90 frames,
   interval 30, confidence 0.35, objects 16, grace 30/10. Leave CSV/video enabled.
3. Expect package SHA256/cache messages, model preparation, per-frame progress
   and successful output paths. Confirm visible boxes follow objects and IDs stay
   stable. Lost objects should not retain frozen boxes in the annotated video.
4. Check CSV statuses and null lost coordinates. With a compatible FMV video, enable
   geographic points in a second run and compare their map positions with the source FMV.
5. Repeat with a new run name in the same Pro process. Expect asset-cache reuse,
   tracker/detector process-cache reuse and skipped detector warm-up.
6. Select a renamed byte-identical DLPK: expect the same fingerprint/cache. Then
   switch between the previous compatible DLPK and the review DLPK: expect different
   fingerprints and both models to reload. Switch back and verify valid outputs.
7. For a 90-frame exit/occlusion/arrival clip, use interval 5 and grace 3/1 as a
   stress test. Confirm termination does not stop survivors, later objects get
   increasing IDs, and recovery within grace retains its ID when SAM reacquires it.
8. Run short MP4, rotated MOV, AVI and TS/M2TS examples; check orientation and
   alignment. Cancel a run, then start a new run to check cleanup.
9. For KLV footage, compare the source SHA256 before/after (`Get-FileHash` in
   PowerShell). It must match. Expect a warning that the derived video has no KLV.
10. Native DLPK validation is separate: select the new DLPK in Configure Object
    Tracker, draw boxes around distinct objects, and track a short segment. Check
    color-sensitive targets, held rows during loss, and object correction. Native
    repeated replacement at full multiplex capacity remains a known limitation;
    the text toolbox's session-pool replacement behavior does not extend to that host.

Long-video identity quality, worst-case performance and real-camera metadata
discontinuities require manual validation. No long inference is part of release tests.

## Group Object Parts

**Group Object Parts** may combine tracks for parts of the same object into one
output box and ID when their shape and movement provide consistent evidence.
It is best suited to connected, elongated parts, such as a truck and trailer.
It can reduce extra boxes, but incorrect grouping is possible. It may increase
processing time, especially with many objects. The option is off by default;
leave it off for separate people or animals.

A group requires accepted visible observations, end-to-end elongated mask geometry,
stable relative motion/position, and recent whole-object evidence. Evidence may
come from a confident whole-vehicle detection or a recent larger mask whose split
parts reconstruct its extent and support. A sudden size change alone is insufficient.
Ambiguous pairs remain separate, including parts without reliable orientation or
whole-object evidence. This is a conservative heuristic, not an attachment classifier.

**Grouping Confirmation Time** defaults to 0.5 seconds;
**Grouping Separation Time** defaults to 1.0 second of observed
contradictory geometry/motion. Both durations are adjustable under Advanced Matching.
Longer confirmation reduces transient grouping but delays it. A missing member does
not establish separation. Timestamp gaps do not count as observed separation.

SAM continues tracking both members. CSV, annotated video and optional geospatial points
use one logical ID and the union of current accepted member boxes. The older ID
survives, even if its original SAM member ends while the other continues. No stale
member box is included. On confirmed separation, surviving members resume their
original IDs. When grouping is enabled, exports add `member_track_ids`; history is
not rewritten retrospectively. Grouped rows replace member rows while grouped.

This option does not reduce SAM slot use or inference cost. Camera changes and
nearby aligned vehicles remain challenging. Review enabled/disabled runs before
using grouped results in downstream analysis. The original video and model package
remain unchanged.


## Multiple object categories

Enter `car; swimming pool` or `car, swimming pool` in **Text Prompts**. Commas and
semicolons separate categories; spaces stay within a phrase. Repeated categories
are searched once, ignoring capitalization. Empty entries are ignored, but at
least one category is required. Commas and semicolons cannot be literal parts of
one category phrase.

Each category is searched on the same scheduled detection frames using the same
loaded detector. Tracking advances once per frame across all categories. More
categories increase detection time. Confidence, detection interval and the maximum
simultaneous-object budget are shared. At capacity, new detections are prioritized
by confidence; there is no reserved capacity per category.

CSV and optional features record the object's initial category in `class_prompt`;
annotated boxes show that category. Existing tracks keep their category and ID.
For near-identical masks found under multiple categories, a new track uses the
higher-confidence detection (first prompt on ties). Existing category matches take
priority. Cross-category containment alone is not duplicate evidence, allowing,
for example, a person inside a swimming pool to remain a separate track. This
heuristic cannot guarantee that synonymous or ambiguous prompts never duplicate.
Prefer distinct categories. Leave Group Object Parts off for mixed scenes unless
its documented attachment assumptions fit; it only groups members of the same
category.

First test: use a short trial and a new run name, with `car; swimming pool`, CSV
and annotated video enabled. Confirm both categories are present when visible,
labels remain stable, and the shared object budget is respected. Repeat with
`car, swimming pool`; the category list should behave identically. Detector
accuracy and runtime depend on the footage and require manual validation.

## Run Name and output protection

Run Name is the base for the generated CSV, annotated MP4 and
optional geographic points. For `Truck_Test_01`, the defaults are:

- `Truck_Test_01.csv`
- `Truck_Test_01_annotated.mp4`
- `SAM31_Tracks.gdb/Truck_Test_01_Detections`

Changing Run Name or Output Folder updates paths still owned by the tool. A
manually edited output path stays unchanged; clear it to restore automatic naming.
Selecting another video changes an automatically generated Run Name, but preserves
a name you entered yourself. If ArcGIS recreates validation state, the tool can
recognize two or more standard output paths sharing a base name in the current
output folder and resume automatic naming. Individually renamed/relocated paths
stay customized. If fewer than two matching generated paths remain, clear the
paths you want to follow Run Name. A deliberately entered path identical to the
standard generated naming pattern cannot be distinguished after state is lost.
Press Tab or click another field after changing Run Name to trigger validation.

Run Names must contain 1-100 letters, numbers or underscores, begin with a letter,
and avoid Windows reserved names. Spaces are supported in folder/video paths;
video stems are sanitized when creating an automatic Run Name. Invalid custom
Run Names produce validation errors instead of silently changing your input.
Existing outputs show a replacement warning when ArcGIS Pro's overwrite existing
outputs setting is enabled (`arcpy.env.overwriteOutput`). With it disabled, choose
a new Run Name/path or enable overwrite. Only selected outputs are replaced;
source-video paths and aliases are always rejected.

CSV is written to a temporary file and published after writing succeeds. Video
is replaced when its writer opens, so an interrupted run can leave a partial video.
The geographic feature class is replaced only after a valid point is available;
if no point qualifies, the prior layer is retained and is not a result of the new
run. A later geographic write failure can leave partial new results. Close any
application, layer or table holding an output lock before retrying. Reuse Run Name
to replace selected results, or change it to preserve the earlier run.

## Optional geographic detection points

Enable **Export Geospatial Detection Points (FMV)** to request a WGS 84 point
feature class. It is off by default and does not change detection or tracking.
The source video is read-only. Track lines are not generated.

The exporter reads a single supported MISB ST 0601 KLV stream once, incrementally,
after tracking. It verifies packet checksums and uses either four full corner
coordinates (tags 82-89) or frame center plus all four corner offsets (tags 23-33).
A metadata record with missing or invalid corners interrupts usable coverage;
older corners are not carried through that record. Sensor position alone is not
sufficient and is never assigned to detected objects.

Each accepted visible box centroid is projected through a homography onto a local
planar footprint defined by the four corners (upper-left, upper-right, lower-right,
lower-left). A local equirectangular plane is used for small footprints. This is
an approximate flat-ground method with no terrain, lens-distortion or object-height
correction. Polar, antimeridian-crossing, large, degenerate and non-convex footprints
are rejected, as are videos with rotation metadata. It is not survey-grade.

KLV packet presentation times are referenced to the video's stream start time and
matched to the decoder's usable `source_timestamp`. The last preceding corner
record must be no more than 0.25 seconds old; this accommodates the inspected
5 Hz metadata stream but deliberately skips slower updates or gaps. Future
metadata is not used. When decoder source timestamps are missing or unusable,
the geographic exporter makes one forward-only ffprobe pass to recover actual
decoded-frame presentation timestamps. These use the same video stream start
as the KLV timeline. It does not estimate timing from frame rate or MISB absolute
time. Each frame timestamp is reused for all observations of that frame.

Recovery assumes sequential decoding from frame zero without skipped frames,
matching the current tracker. It rejects missing/non-increasing timestamps,
decoding errors, mismatched dimensions, an unexpected first-frame timestamp,
and disagreement with available decoder timing. A failure stops geographic
export at that point; previously written points and other outputs are retained.
The fallback buffers only a few timestamp records and closes when export ends,
but the additional video-decoding pass can increase geographic export time.
It runs only when geographic output needs timestamp recovery.

Timestamp recovery does not establish geographic accuracy. Validate point
alignment and metadata compatibility on representative FMV footage.

Point attributes preserve `run_name`, `track_id`, `class_prompt`, `frame_number`,
`timestamp`, `source_timestamp`, `confidence`, `status`, `pixel_x`, `pixel_y`, the
four box coordinates, `longitude`, `latitude`, and `source_video`. Additional fields
record `georef_timestamp` (the time used for map alignment), `timestamp_source`
(`decoder` or `ffprobe_frame_pts`), metadata time/age, raw MISB precision timestamp, sensor and frame-center
coordinates, sensor altitude, method, and grouped member IDs where applicable.
The point layer is a subset of the existing CSV observations; CSV columns and
track IDs are unchanged. CSV and original `source_timestamp` values are not
rewritten when timing is recovered. Lost, tentative, out-of-frame and terminated rows do not
produce points. Precision timestamp is retained as text, not reinterpreted as UTC.

If nothing can be georeferenced, the tool reports: "No usable FMV georeferencing
metadata was found. Geospatial detection output was not created." Existing CSV
and annotated-video outputs continue normally. Partial metadata
coverage yields only the valid subset, with counts reported in the messages.

Validate overlay alignment against independent ground references in ArcGIS Pro
before using the approximate point positions for analysis.

After updating from a version with pixel-feature parameters, open a fresh tool
dialog. Saved History entries and positional ModelBuilder/script calls may need
repair because those two parameters were removed. The toolbox name is unchanged.
