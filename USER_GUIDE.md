# User guide

## Prerequisites

The tested environment is Windows with ArcGIS Pro 3.7, its matching 3.7
deep-learning libraries, Python 3.13, PyTorch 2.9.1 with CUDA 12.9 and
torchvision 0.25.0. Use the matching Esri-supported deep-learning installation
for Pro; the tool does not install dependencies or modify Conda environments.
Other versions have not been validated.

A compatible NVIDIA CUDA GPU and driver are required. CPU-only inference is
unsupported. Testing used approximately 12 GB of GPU memory capacity; memory
needs depend on frames and active objects. Close other GPU-heavy applications.
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
Do not copy the `.pyt` alone. The release includes all project runtime source;
the original development repository and custom PYTHONPATH are unnecessary.

Select **SAM 3.1 Model Package (.dlpk)**. This is a local file selector, not an
upload. This release accepts the pinned SAM 3.1 Object Multiplex checkpoint and
CLIP vocabulary identified in `VERSION.json`. Fine-tuned or other SAM versions
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

CSV and annotated MP4 are enabled by default. Pixel-space feature-class export
is optional and off by default. CSV and feature-class rows contain track IDs,
status, frame/time references and visible coordinates. Lost/terminated observations
have null coordinates; feature-class geometry is also null for those rows.
The feature class is not georeferenced and cannot locate objects on a geographic
map. To inspect it, add it from the output file geodatabase and open its table.

Annotated video is a derived visual review product without KLV/MISB or original
telemetry. It is not FMV-compliant. The source remains read-only. `source_video`
and `source_timestamp` support later association with metadata, subject to decoder
timestamp limitations; this tool does not perform georeferencing or parse MISB.

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
4. Check CSV statuses and null lost coordinates. Enable feature output in a second
   run; open its geodatabase table and confirm visible polygons and null lost geometry.
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

SAM continues tracking both members. CSV, annotated video and optional pixel features
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
