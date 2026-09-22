import logging
import os
import sys
import tempfile

_LOGGER_NAME = "SAM31ObjectTracker"


def default_log_dir():
    d = os.path.join(tempfile.gettempdir(), "SAM31ObjectTracker")
    os.makedirs(d, exist_ok=True)
    return d


def get_logger(debug=None, log_file=None):
    """Return the package logger. `debug=True` switches to DEBUG level.

    The SAM31_DEBUG environment variable (1/0) is honoured when `debug` is None.
    Logs go to stderr AND to a file: `log_file` if given, else SAM31_LOG_FILE, else
    %TEMP%/SAM31ObjectTracker/SAM31ObjectTracker.log. ArcGIS Pro runs the tracker
    Python in-process without a console, so the file is the primary diagnostic.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if debug is None:
        debug = os.environ.get("SAM31_DEBUG", "0") == "1"
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    if not any(getattr(h, "_sam31", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(fmt)
        handler._sam31 = True
        logger.addHandler(handler)
        logger.propagate = False
    path = log_file or os.environ.get("SAM31_LOG_FILE") or os.path.join(default_log_dir(), "SAM31ObjectTracker.log")
    if not any(getattr(h, "_sam31_file", None) == path for h in logger.handlers):
        try:
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(fmt)
            fh._sam31 = True
            fh._sam31_file = path
            logger.addHandler(fh)
        except Exception:
            pass
    return logger
