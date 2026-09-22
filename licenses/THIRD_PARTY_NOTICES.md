# Third-party notices

| Component | Source and terms | Included |
| --- | --- | --- |
| SAM 3.1 model and SAM source | [Meta SAM License](https://huggingface.co/facebook/sam3.1/blob/main/LICENSE); supplied verbatim in [SAM_LICENSE](SAM_LICENSE) | Runtime source; checkpoint excluded from this repository |
| CLIP tokenizer/vocabulary | [OpenAI CLIP MIT License](https://github.com/openai/CLIP/blob/main/LICENSE); [CLIP_LICENSE](CLIP_LICENSE) | Adapted tokenizer source; vocabulary excluded from this repository |
| Windows EDT compatibility adaptation | Existing Meta-copyright SAM code obtained from the ArcGIS Living Atlas SAM package with its SAM License | Preserved adaptation; provenance in [runtime patch notes](../SAM31_ObjectTracker/PATCHES.md) |
| Original ArcGIS integration | [Apache License 2.0](APACHE_LICENSE); independently authored contributions only, as described below | Custom toolbox and supporting integration code |

SAM source is pinned to commit `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` of
[facebookresearch/sam3](https://github.com/facebookresearch/sam3).
Checkpoint provenance is [facebook/sam3.1](https://huggingface.co/facebook/sam3.1).
The SAM License permits redistribution subject to its conditions, including
providing the license and complying with its restrictions; it is not an unrestricted
permission statement. Review those conditions for the intended publication and use.

Upstream source attribution comments are retained, including references to
GroundingDINO, ConvNeXt, MaskFormer, SAM 2, Detectron2, RITM, TrackEval, rotary
embedding implementations and other ancestors named in the source. They are
included as part of the pinned SAM source, not fetched as separate packages.
No author names or copyright claims have been removed or invented.

PyTorch, torchvision, numpy, pandas, scipy, Pillow, OpenCV, timm, ftfy, regex,
iopath, tqdm, einops, huggingface_hub, psutil, FFmpeg and ArcGIS are external
prerequisites and are not redistributed here. Their own terms apply. No Esri
binaries, SDK, ArcGIS installation, sample media, datasets or standalone CLIP model
are included. This distribution does not grant an ArcGIS license.

The companion checkpoint and vocabulary hashes are recorded in [VERSION.json](../VERSION.json).
This repository contains no model weights or vocabulary archive.

## License scope

The independently authored toolbox, integration and documentation contributions
are licensed under [Apache License 2.0](APACHE_LICENSE). Copyright and project
attribution are preserved in [NOTICE](NOTICE).

This covers independently authored material in `SAM31_TextPromptTracker.pyt`,
its XML help files, `src/`, `SAM31_ObjectTracker/sam31_runtime/`, and project
documentation. It excludes incorporated third-party material and SAM-derived
modifications, which retain their applicable upstream terms.

Apache 2.0 permits commercial and noncommercial use, modification and redistribution
subject to its conditions, including providing the license, preserving applicable
notices and identifying modified files. It does not require publication of
modifications. Its grant applies only to rights contributors are entitled to grant.

Meta SAM materials and SAM-derived modifications retain the SAM License; adapted
CLIP components retain their MIT notice. The separately distributed checkpoint
and vocabulary retain their applicable third-party terms. The complete workflow
is not licensed solely under Apache 2.0. Users must comply with the applicable
upstream and external dependency licenses as well.
