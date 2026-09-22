"""Switch package-specific model caches together before creating a new runtime."""
_fingerprint = None


def select_package(fingerprint, logger=None):
    global _fingerprint
    if fingerprint != _fingerprint:
        from detector_cache import clear_cache as clear_detector
        from sam31_runtime.model_loader import clear_cache as clear_tracker
        clear_detector()
        clear_tracker()
        _fingerprint = fingerprint
        if logger:
            logger.info('Selected model package SHA256 %s; previous model caches cleared', fingerprint)
