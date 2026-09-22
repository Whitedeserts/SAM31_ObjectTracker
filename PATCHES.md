# SAM runtime compatibility patches

Pinned upstream commit: `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`.

- **P1, `sam3/model/edt.py`:** guards the Triton import and uses the CPU fallback
  where Triton is unavailable. The implementation retains its upstream copyright
  and was obtained from the ArcGIS Living Atlas SAM package.
- **P2, `sam3/train/data/collator.py`:** uses `Any` for the dataset annotation,
  avoiding a training dataset dependency during inference imports.
- **P3, `sam3/model/decoder.py`:** permits efficient and math SDPA backends when
  flash attention has no compatible kernel.
- **P4, `sam3/model/maskformer_segmentation.py`:** normalizes multiplex features
  to tensors before segmentation, including wrappers retained across module
  imports. Tensor contents are validated before entering the pixel decoder.
- Optional profiler output uses the system temporary directory in this release.
  Standalone planning comments and private upstream review links are omitted.

The upstream predictor request API forwards `offload_state_to_cpu`, which the
pinned multiplex initializer does not accept. The batch toolbox uses its own
bounded streaming state. The legacy full-video API calls the initializer directly;
it is retained for compatibility and is not used by the toolbox.

The project-specific session adapter is separate from these upstream patches.
See the release's `RUNTIME_NOTES.md` for memory retention and multiplex slot behavior.
