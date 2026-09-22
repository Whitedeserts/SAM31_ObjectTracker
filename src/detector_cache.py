"""One process-wide detector cache, retained across toolbox source refreshes.

Keep only the latest configuration so changing settings does not accumulate
multi-GB models. Predictor sessions belong to runtimes and are closed there.
"""
import os

_key = None
_detector = None


def clear_cache():
    global _key, _detector
    _key, _detector = None, None


def get_detector(*, checkpoint_path, bpe_path, logger, **options):
    global _key, _detector
    key = (os.path.normcase(os.path.realpath(checkpoint_path)),
           os.path.normcase(os.path.realpath(bpe_path)), tuple(sorted(options.items())))
    if key == _key and _detector is not None:
        logger.info("SAM 3.1 detector reused from process cache (no reload)")
        return _detector
    # Release the previous cache reference before constructing a different model.
    _key, _detector = None, None
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    detector = build_sam3_multiplex_video_predictor(
        checkpoint_path=checkpoint_path, bpe_path=bpe_path, **options)
    _key, _detector = key, detector
    return detector
