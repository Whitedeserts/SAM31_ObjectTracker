"""Read-only, bounded ST 0601 corner metadata and approximate ground projection.

Only timestamped, checksum-valid UAS Local Sets with four explicit corners are
accepted. No sensor-pose reconstruction, terrain correction or timestamp guessing.
"""
from __future__ import annotations
import json
import math
import queue
import re
import subprocess
import threading
from dataclasses import dataclass

import numpy as np
from video_reader import FFPROBE, _probe_with_ffprobe

KEY = bytes.fromhex('060e2b34020b01010e01030101000000')
MAX_PACKET_BYTES = 65536
MAX_METADATA_AGE = 0.25  # seconds; accommodates the inspected 5 Hz source, not long gaps


class MetadataUnavailable(ValueError):
    pass


def _ber(data, offset):
    if offset >= len(data):
        raise MetadataUnavailable('Truncated KLV length')
    value = data[offset]; offset += 1
    if value & 128:
        count = value & 127
        if not 1 <= count <= 4 or offset + count > len(data):
            raise MetadataUnavailable('Invalid KLV length')
        value = int.from_bytes(data[offset:offset+count], 'big'); offset += count
    return value, offset


def _signed(fields, tag, length, limit):
    data = fields.get(tag)
    if data is None or len(data) != length:
        return None
    value = int.from_bytes(data, 'big', signed=True)
    denominator = (1 << (8*length-1))-1
    return None if value == -denominator-1 else value*limit/denominator


def _unsigned(fields, tag, length, low, high):
    data = fields.get(tag)
    if data is None or len(data) != length:
        return None
    return low + int.from_bytes(data, 'big')*(high-low)/((1 << (8*length))-1)


@dataclass
class CornerMetadata:
    time: float
    corners: list | None
    values: dict


def decode_local_set(data, time):
    if not math.isfinite(time) or len(data) > MAX_PACKET_BYTES or not data.startswith(KEY):
        raise MetadataUnavailable('Unsupported KLV packet')
    size, offset = _ber(data, len(KEY))
    if offset+size != len(data) or data[-4:-2] != b'\x01\x02':
        raise MetadataUnavailable('Incomplete KLV packet or missing checksum')
    expected = sum(b << (8 if i % 2 == 0 else 0) for i,b in enumerate(data[:-2])) & 0xffff
    if expected != int.from_bytes(data[-2:], 'big'):
        raise MetadataUnavailable('KLV checksum mismatch')
    fields = {}
    while offset < len(data):
        tag = 0
        for _ in range(4):
            if offset >= len(data): raise MetadataUnavailable('Truncated tag')
            b=data[offset]; offset+=1; tag=(tag << 7) | (b & 127)
            if not b & 128: break
        else: raise MetadataUnavailable('Invalid KLV tag')
        length, offset = _ber(data, offset)
        if offset+length > len(data) or tag in fields:
            raise MetadataUnavailable('Truncated or duplicate KLV field')
        fields[tag] = data[offset:offset+length]; offset += length
    values = {
        'sensor_latitude': _signed(fields,13,4,90),
        'sensor_longitude': _signed(fields,14,4,180),
        'sensor_altitude': _unsigned(fields,15,2,-900,19000),
        'frame_latitude': _signed(fields,23,4,90),
        'frame_longitude': _signed(fields,24,4,180),
        'heading': _unsigned(fields,5,2,0,360),
        'pitch': _signed(fields,6,2,20), 'roll': _signed(fields,7,2,50),
        'horizontal_fov': _unsigned(fields,16,2,0,180),
        'vertical_fov': _unsigned(fields,17,2,0,180),
        'sensor_azimuth': _unsigned(fields,18,4,0,360),
        'sensor_elevation': _signed(fields,19,4,180),
        'sensor_roll': _unsigned(fields,20,4,0,360),
        'slant_range': _unsigned(fields,21,4,0,5000000),
    }
    ts = fields.get(2,b'')
    # Preserve MISB precision time as integer text; do not reinterpret it as video PTS.
    values['precision_timestamp'] = str(int.from_bytes(ts,'big')) if len(ts)==8 else None
    corners = []
    if any(tag in fields for tag in range(82,90)):
        for i in range(4):
            corners.append((_signed(fields,83+2*i,4,180),_signed(fields,82+2*i,4,90)))
    elif values['frame_latitude'] is not None and values['frame_longitude'] is not None:
        for i in range(4):
            lat=_signed(fields,26+2*i,2,.075); lon=_signed(fields,27+2*i,2,.075)
            corners.append((None if lon is None else values['frame_longitude']+lon,
                            None if lat is None else values['frame_latitude']+lat))
    if len(corners)!=4 or any(v is None for corner in corners for v in corner):
        corners=None
    return CornerMetadata(time,corners,values)


def decode_hex_dump(text):
    data=bytearray()
    for line in text.replace(r'\n','\n').splitlines():
        if ':' not in line: continue
        hexpart=line.split(':',1)[1].lstrip().split('  ',1)[0]
        if not re.fullmatch(r'[0-9a-fA-F ]*',hexpart):
            raise MetadataUnavailable('Invalid ffprobe KLV payload')
        data.extend(bytes.fromhex(hexpart))
        if len(data)>MAX_PACKET_BYTES: raise MetadataUnavailable('KLV packet too large')
    return bytes(data)


def iter_metadata(video, stop_time=None):
    """One forward-only ffprobe pass; at most four small packet lines buffered."""
    info=_probe_with_ffprobe(video)
    if not info or not FFPROBE: raise MetadataUnavailable('ffprobe metadata reader is unavailable')
    streams=info.get('streams',[])
    videos=[s for s in streams if s.get('codec_type')=='video']
    klvs=[s for s in streams if s.get('codec_name') in ('klv','smpte_klv')]
    if len(videos)!=1 or len(klvs)!=1:
        raise MetadataUnavailable('A single video and a single supported KLV stream are required')
    stream=videos[0]
    rotation=float(stream.get('tags',{}).get('rotate',0) or 0)
    if rotation or any(float(s.get('rotation',0) or 0) for s in stream.get('side_data_list',[])):
        raise MetadataUnavailable('Rotated video requires a calibrated pixel-to-corner mapping')
    try: origin=float(stream['start_time'])
    except (KeyError,TypeError,ValueError): raise MetadataUnavailable('No video PTS origin is available')
    if not math.isfinite(origin): raise MetadataUnavailable('Invalid video PTS origin')
    cmd=[FFPROBE,'-v','error','-select_streams',str(klvs[0]['index']),
         '-show_packets','-show_data','-show_entries','packet=pts_time,data:packet_side_data=',
         '-of','compact=p=0:nk=0']
    if stop_time is not None: cmd += ['-read_intervals', '%+'+str(max(0,stop_time)+1)]
    cmd.append(video)
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,
                          encoding='utf-8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    items=queue.Queue(maxsize=4); done=threading.Event()
    def read_lines():
        try:
            while not done.is_set():
                line=proc.stdout.readline(1024*1024)
                item=line if line and len(line)<1024*1024 else None
                while not done.is_set():
                    try: items.put(item,timeout=.1);break
                    except queue.Full: pass
                if item is None: break
        except (OSError,ValueError):
            pass
    worker=threading.Thread(target=read_lines,daemon=True);worker.start()
    last=-math.inf
    try:
        while True:
            try: line=items.get(timeout=30)
            except queue.Empty: raise MetadataUnavailable('KLV metadata read timed out')
            if line is None: break
            match=re.search(r'(?:^|\|)pts_time=([^|]+)',line)
            if not match: raise MetadataUnavailable('KLV packet lacks presentation time')
            try: time=float(match.group(1))-origin
            except ValueError: raise MetadataUnavailable('Invalid KLV presentation time')
            if not math.isfinite(time) or time<last:
                raise MetadataUnavailable('Non-monotonic KLV presentation timestamps')
            last=time
            if stop_time is not None and time>stop_time: break
            try:
                record=decode_local_set(decode_hex_dump(line.split('|data=',1)[1]),time)
            except (MetadataUnavailable,IndexError):
                # An invalid observation is a barrier: never carry earlier corners through it.
                record=CornerMetadata(time,None,{})
            yield record
        if proc.poll() not in (None,0): raise MetadataUnavailable('ffprobe metadata extraction failed')
    finally:
        done.set()
        if proc.poll() is None: proc.terminate()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill();proc.wait(timeout=5)
        worker.join(timeout=1)
        proc.stdout.close()


class MetadataMatcher:
    def __init__(self, records, max_age=MAX_METADATA_AGE):
        self.records=iter(records); self.current=None;self.next=None
        self.max_age=max_age;self.last=-math.inf;self.ended=False

    def at(self, time):
        if time is None or not math.isfinite(time) or time<self.last:
            return None
        self.last=time
        while True:
            if self.next is None and not self.ended:
                self.next=next(self.records,None);self.ended=self.next is None
            if self.next is None or self.next.time>time: break
            self.current,self.next=self.next,None
        if self.current and self.current.corners and 0<=time-self.current.time<=self.max_age:
            return self.current
        return None

    def close(self):
        close=getattr(self.records,'close',None)
        if close:close()


def project_centroid(corners, x, y, width, height):
    """Homography onto a small, flat ground footprint; TL, TR, BR, BL order.

    Uses a local equirectangular plane, rejects large/polar/dateline footprints
    and degenerate/non-convex quadrilaterals. No terrain or object-height correction.
    """
    if width<2 or height<2 or not (0<=x<=width-1 and 0<=y<=height-1):
        raise MetadataUnavailable('Pixel outside unrotated frame')
    c=np.asarray(corners,dtype=float)
    if c.shape!=(4,2) or not np.isfinite(c).all() or (np.abs(c[:,0])>180).any() or (np.abs(c[:,1])>=85).any():
        raise MetadataUnavailable('Invalid ground corners')
    if np.ptp(c[:,0])>1 or np.ptp(c[:,1])>1:
        raise MetadataUnavailable('Footprint is too large or crosses the antimeridian')
    center=c.mean(axis=0);scale=np.array([math.cos(math.radians(center[1])),1.])*111319.49079327358
    ground=(c-center)*scale
    edges=np.roll(ground,-1,axis=0)-ground
    cross=edges[:,0]*np.roll(edges,-1,axis=0)[:,1]-edges[:,1]*np.roll(edges,-1,axis=0)[:,0]
    if not ((cross>1e-6).all() or (cross< -1e-6).all()):
        raise MetadataUnavailable('Degenerate or non-convex frame footprint')
    norm=max(np.abs(ground).max(),1.)
    dest=ground/norm
    a=[];b=[]
    for (u,v),(gx,gy) in zip(((0,0),(1,0),(1,1),(0,1)),dest):
        a.extend(([u,v,1,0,0,0,-gx*u,-gx*v],[0,0,0,u,v,1,-gy*u,-gy*v]));b.extend((gx,gy))
    if np.linalg.cond(a)>1e8:raise MetadataUnavailable('Ill-conditioned footprint transform')
    h=np.append(np.linalg.solve(a,b),1.).reshape(3,3)
    denominators=[(h@np.array([u,v,1.]))[2] for u,v in ((0,0),(1,0),(1,1),(0,1))]
    if min(denominators)<=1e-8: raise MetadataUnavailable('Projective horizon intersects the frame')
    q=h@np.array([x/(width-1),y/(height-1),1.])
    lon,lat=center+(q[:2]/q[2])*norm/scale
    return float(lon),float(lat)
