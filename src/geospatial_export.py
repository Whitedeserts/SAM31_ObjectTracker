"""Optional geographic detection points; independent of tracker state and CSV schema."""
import math
import os
from collections import Counter
from fmv_georeferencing import MetadataMatcher, MetadataUnavailable, iter_metadata, project_centroid
from frame_timestamps import FrameTimestampReader
from output_naming import require_output_writable, validate_feature_path, overwrite_allowed, output_exists

FIELDS = [
    ('run_name','TEXT',100),('track_id','LONG'),('class_prompt','TEXT',255),
    ('frame_number','LONG'),('timestamp','DOUBLE'),('source_timestamp','DOUBLE'),
    ('confidence','DOUBLE'),('status','TEXT',24),('pixel_x','DOUBLE'),('pixel_y','DOUBLE'),
    ('xmin','DOUBLE'),('ymin','DOUBLE'),('xmax','DOUBLE'),('ymax','DOUBLE'),
    ('longitude','DOUBLE'),('latitude','DOUBLE'),('source_video','TEXT',2048),
    ('georef_timestamp','DOUBLE'),('timestamp_source','TEXT',24),
    ('metadata_time','DOUBLE'),('metadata_age','DOUBLE'),('precision_timestamp','TEXT',24),
    ('sensor_latitude','DOUBLE'),('sensor_longitude','DOUBLE'),('sensor_altitude','DOUBLE'),
    ('frame_latitude','DOUBLE'),('frame_longitude','DOUBLE'),('method','TEXT',40),
    ('member_track_ids','TEXT',128),
]
NO_GEO = 'No usable FMV georeferencing metadata was found. Geospatial detection output was not created.'


def export_geospatial_points(rows, path, *, video, run_name, probe, logger=None, records=None):
    """Write VISIBLE observations with decoder or recovered PTS and a recent footprint.

    Missing timing/metadata is not an inference failure. No stale position is
    created for lost/tentative/terminated tracks, and no frame-rate timing estimates are used.
    """
    import arcpy
    validate_feature_path(path);require_output_writable(path)
    counts=Counter();cursor=None;matcher=None;created=None
    timing=None;last_source=None
    def warn(message):
        if logger:logger.warning('%s',message)
    if not probe.has_klv and records is None:
        warn(NO_GEO);return None,dict(counts)
    try:
        for row in rows.itertuples(index=False):
            counts['observations']+=1
            if row.status!='VISIBLE':counts['not_visible']+=1;continue
            ts=getattr(row,'source_timestamp',None)
            usable_source = ts is not None and math.isfinite(ts) and ts >= 0
            timestamp_source = 'decoder'
            aligned_time = ts
            if not usable_source:
                if timing is None:
                    timing = FrameTimestampReader(video, probe.width, probe.height)
                    if last_source is not None:
                        recovered = timing.at(last_source[0])
                        if abs(recovered - last_source[1]) > .002:
                            raise MetadataUnavailable('Recovered timing disagrees with decoder timing.')
                aligned_time = timing.at(row.frame_number)
                timestamp_source = 'ffprobe_frame_pts'
                counts['recovered_timestamps'] += 1
            else:
                if timing is not None and abs(timing.at(row.frame_number) - ts) > .002:
                    raise MetadataUnavailable('Recovered timing disagrees with decoder timing.')
                last_source = (row.frame_number, ts)
            box=[row.xmin,row.ymin,row.xmax,row.ymax]
            if any(v is None or not math.isfinite(v) for v in box) or box[2]<=box[0] or box[3]<=box[1]:
                counts['invalid_box']+=1;continue
            if matcher is None:matcher=MetadataMatcher(records if records is not None else iter_metadata(video))
            metadata=matcher.at(aligned_time)
            if metadata is None:counts['missing_or_stale_metadata']+=1;continue
            x=(box[0]+box[2])/2;y=(box[1]+box[3])/2
            try:lon,lat=project_centroid(metadata.corners,x,y,probe.width,probe.height)
            except MetadataUnavailable:counts['invalid_geometry']+=1;continue
            if cursor is None:
                gdb,name=os.path.split(path)
                if not arcpy.Exists(gdb):arcpy.management.CreateFileGDB(os.path.dirname(gdb),os.path.basename(gdb))
                require_output_writable(path)
                if output_exists(path) and not arcpy.TestSchemaLock(path):
                    raise RuntimeError("Cannot replace geospatial output " + path + ". Close layers or tables using it and try again.")
                with arcpy.EnvManager(overwriteOutput=overwrite_allowed(), overwriteOutputOptions="Delete"):
                    arcpy.management.CreateFeatureclass(gdb,name,'POINT',spatial_reference=arcpy.SpatialReference(4326))
                created=path
                for spec in FIELDS:
                    arcpy.management.AddField(path,spec[0],spec[1],**({'field_length':spec[2]} if len(spec)>2 else {}))
                cursor=arcpy.da.InsertCursor(path,[f[0] for f in FIELDS]+['SHAPE@XY'])
            values=dict(run_name=run_name,track_id=int(row.track_id),class_prompt=str(row.class_prompt),
                        frame_number=int(row.frame_number),timestamp=float(row.timestamp),source_timestamp=ts if usable_source else None,
                        confidence=row.confidence,status=row.status,pixel_x=x,pixel_y=y,
                        xmin=box[0],ymin=box[1],xmax=box[2],ymax=box[3],longitude=lon,latitude=lat,
                        source_video=video,metadata_time=metadata.time,metadata_age=aligned_time-metadata.time,
                        georef_timestamp=aligned_time,timestamp_source=timestamp_source,
                        method='ST0601_planar_projective',member_track_ids=str(getattr(row,'member_track_ids','')))
            values.update(metadata.values)
            cursor.insertRow([values.get(f[0]) for f in FIELDS]+[(lon,lat)])
            counts['written']+=1
    except MetadataUnavailable as exc:
        warn('Geospatial metadata unavailable: '+str(exc))
        counts['metadata_read_stopped']+=1
    finally:
        if cursor is not None:del cursor
        if matcher is not None:matcher.close()
        if timing is not None:timing.close()
    if not created:
        warn(NO_GEO)
        if output_exists(path):warn("The previous geospatial output was retained; it does not contain results from this run.")
    elif logger:logger.info('Geospatial points: %s. Method assumes a small planar footprint; no terrain correction.',dict(counts))
    if counts['recovered_timestamps'] and logger:
        logger.info('Recovered source frame timestamps for %s observations; CSV timestamps are unchanged.', counts['recovered_timestamps'])
    return created,dict(counts)
