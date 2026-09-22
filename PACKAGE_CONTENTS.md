# Package contents

- `SAM31_TextPromptTracker.pyt` and XML files: ArcGIS toolbox and help metadata.
- `src/`: text detection, streaming tracking, grouping, package validation and exports.
- `SAM31_ObjectTracker/sam3/`: pinned SAM implementation and required inference helpers.
- `SAM31_ObjectTracker/sam31_runtime/`: model loading and tracking session integration.
- Version metadata, user/environment guides, modification notes and third-party licenses.
- `NOTICE`: creator credit and copyright attribution for the original toolbox.

External prerequisites: supported ArcGIS Pro deep-learning environment, NVIDIA CUDA
GPU, compatible separately distributed SAM 3.1 DLPK, and local input video.

Excluded: weights, tokenizer vocabulary archive, DLPK, native adapter distribution,
Python caches, outputs, videos, local environments, private paths and development history.
