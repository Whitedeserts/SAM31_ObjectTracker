# Third-party notices

| Component | Source and terms | Included |
| --- | --- | --- |
| SAM 3.1 model and SAM source | [Meta SAM License](https://huggingface.co/facebook/sam3.1/blob/main/LICENSE); supplied verbatim in [licenses/SAM_LICENSE](licenses/SAM_LICENSE) | Runtime source; checkpoint excluded from this repository |
| CLIP tokenizer/vocabulary | [OpenAI CLIP MIT License](https://github.com/openai/CLIP/blob/main/LICENSE); licenses/CLIP_LICENSE | Adapted tokenizer source; vocabulary excluded from this repository |
| Windows EDT compatibility adaptation | Existing Meta-copyright SAM code obtained from the ArcGIS Living Atlas SAM package with its SAM License | Preserved adaptation; provenance in PATCHES.md |
| Original ArcGIS integration | [Apache License 2.0](LICENSE); independently authored contributions only, as described in [LICENSE_SCOPE.md](LICENSE_SCOPE.md) | Custom toolbox and supporting integration code |

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

The companion checkpoint and vocabulary hashes are recorded in VERSION.json.
This repository contains no model weights or vocabulary archive.
