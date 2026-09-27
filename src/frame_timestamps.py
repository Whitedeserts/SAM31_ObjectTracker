"""Bounded recovery of presentation timestamps for sequential video observations."""
import math
import queue
import subprocess
import threading

from fmv_georeferencing import MetadataUnavailable
from video_reader import FFPROBE, _probe_with_ffprobe


def iter_frame_times(video, width, height):
    """Yield actual decoded-frame PTS relative to the video stream start.

    Frame indices assume the existing forward-only decoder starts at frame zero
    without seeking or dropping frames. Decode errors stop recovery; packet order
    and frame-rate estimates are never substituted for presentation order.
    """
    info = _probe_with_ffprobe(video)
    streams = [s for s in (info or {}).get('streams', []) if s.get('codec_type') == 'video']
    if not FFPROBE or len(streams) != 1:
        raise MetadataUnavailable('Timestamp recovery requires one video stream and ffprobe.')
    stream = streams[0]
    try:
        origin = float(stream['start_time'])
    except (KeyError, ValueError, TypeError):
        raise MetadataUnavailable('Timestamp recovery requires a video start PTS.')
    if not math.isfinite(origin) or (stream.get('width'), stream.get('height')) != (width, height):
        raise MetadataUnavailable('Timestamp recovery stream does not match the decoded video.')
    cmd = [FFPROBE, '-v', 'error', '-err_detect', 'explode', '-select_streams', str(stream['index']),
           '-show_frames', '-show_entries', 'frame=pts_time,width,height:frame_side_data=',
           '-of', 'compact=p=0:nk=0', video]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace',
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    pending = queue.Queue(maxsize=4)
    stopped = threading.Event()

    def read_lines():
        try:
            while not stopped.is_set():
                line = process.stdout.readline(4096)
                while not stopped.is_set():
                    try:
                        pending.put(line, timeout=.1)
                        break
                    except queue.Full:
                        pass
                if not line:
                    break
        except (OSError, ValueError):
            pass

    worker = threading.Thread(target=read_lines, daemon=True)
    worker.start()
    previous = None
    try:
        while True:
            try:
                line = pending.get(timeout=30)
            except queue.Empty:
                raise MetadataUnavailable('Frame timestamp recovery timed out.')
            if not line:
                if process.wait(timeout=5) != 0:
                    raise MetadataUnavailable('Frame timestamp decoding failed.')
                break
            if len(line) >= 4096:
                raise MetadataUnavailable('Invalid frame timestamp record.')
            fields = dict(part.split('=', 1) for part in line.strip().split('|') if '=' in part)
            try:
                stamp = float(fields['pts_time']) - origin
                dimensions = (int(fields['width']), int(fields['height']))
            except (KeyError, ValueError):
                # stderr shares the pipe: decode errors must not shift the frame index silently.
                raise MetadataUnavailable('Missing frame PTS or a video decode error during timestamp recovery.')
            if dimensions != (width, height) or not math.isfinite(stamp):
                raise MetadataUnavailable('Invalid frame dimensions or presentation timestamp.')
            if previous is None:
                if abs(stamp) > .0001:
                    raise MetadataUnavailable('First decoded frame does not match video start PTS.')
                stamp = max(0., stamp)
            elif stamp <= previous:
                raise MetadataUnavailable('Frame presentation timestamps are not strictly increasing.')
            previous = stamp
            yield stamp
    finally:
        stopped.set()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        worker.join(timeout=1)
        process.stdout.close()


class FrameTimestampReader:
    """Forward-only frame lookup; repeated track observations share one timestamp."""
    def __init__(self, video, width, height):
        self.records = iter_frame_times(video, width, height)
        self.index = -1
        self.time = None

    def at(self, frame):
        if frame < 0 or int(frame) != frame or frame < self.index:
            raise MetadataUnavailable('Frame observations are not in sequential order.')
        while self.index < frame:
            self.time = next(self.records, None)
            if self.time is None:
                raise MetadataUnavailable('Video ended before the requested detection frame.')
            self.index += 1
        return self.time

    def close(self):
        self.records.close()
