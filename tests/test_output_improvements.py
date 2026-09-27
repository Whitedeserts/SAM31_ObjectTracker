"""Short naming, non-overwrite and geographic-output regressions; no SAM inference."""
import csv
import logging
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import arcpy
import numpy as np
import pandas as pd
import output_naming as naming
import arcgis_export
from fmv_georeferencing import (KEY, CornerMetadata, MetadataMatcher, MetadataUnavailable,
                               decode_local_set, project_centroid)
from geospatial_export import export_geospatial_points

CORNERS=[(-105.,40.),(-104.999,40.),(-104.999,39.999),(-105.,39.999)]


def packet(fields):
    body=b''.join(bytes([tag,len(value)])+value for tag,value in fields)+b'\x01\x02\0\0'
    size=len(body);prefix=bytes([size]) if size<128 else b'\x81'+bytes([size])
    data=KEY+prefix+body
    checksum=sum(b<<(8 if i%2==0 else 0) for i,b in enumerate(data[:-2]))&65535
    return data[:-2]+checksum.to_bytes(2,'big')


def signed(value,limit,length):
    return round(value/limit*((1<<(length*8-1))-1)).to_bytes(length,'big',signed=True)


def observations():
    return pd.DataFrame([dict(frame_number=i,timestamp=t,source_timestamp=t,source_video='sample.ts',
        track_id=tid,class_prompt='truck',confidence=.9,xmin=x,ymin=10.,xmax=x+20.,ymax=30.,status=status)
        for i,t,tid,x,status in [(0,.1,4,10.,'VISIBLE'),(0,.1,7,60.,'VISIBLE'),
                               (1,.2,4,15.,'VISIBLE'),(1,.2,7,60.,'OUT_OF_FRAME')]])


class NamingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='SAM naming ');self.addCleanup(self.temp.cleanup)
        self.tool=runpy.run_path(str(ROOT/'SAM31_TextPromptTracker.pyt'))['SAM31TextPromptTrackingTool']()
        self.params=self.tool.getParameterInfo();self.p={p.name:p for p in self.params}
        self.p['output_folder'].value=self.temp.name
        self.p['in_video'].value=os.path.join(self.temp.name,'Video A.ts')

    def update(self):self.tool.updateParameters(self.params)

    def test_initial_defaults_and_auto_video_change(self):
        self.update();first=self.p['run_name'].valueAsText
        self.assertIn('video_a',first.lower())
        self.assertEqual(self.p['csv_path'].valueAsText,naming.build_output_names(first,self.temp.name)['csv_path'])
        self.p['in_video'].value=os.path.join(self.temp.name,'Video B.ts');self.update()
        self.assertNotEqual(first,self.p['run_name'].valueAsText)
        self.assertIn('video_b',self.p['csv_path'].valueAsText.lower())

    def test_manual_run_updates_all_generated_outputs_and_survives_video_change(self):
        self.update();self.p['run_name'].value='Truck_Test_02';self.update()
        expected=naming.build_output_names('Truck_Test_02',self.temp.name)
        for name,path in expected.items():self.assertEqual(naming.canonical(self.p[name].valueAsText),naming.canonical(path))
        self.p['in_video'].value=os.path.join(self.temp.name,'Video B.ts');self.update()
        self.assertEqual(self.p['run_name'].valueAsText,'Truck_Test_02')

    def test_custom_output_preserved_and_clear_returns_to_auto(self):
        self.update();custom=os.path.join(self.temp.name,'custom.csv');self.p['csv_path'].value=custom
        self.update();self.p['run_name'].value='Other';self.update()
        self.assertEqual(self.p['csv_path'].valueAsText,custom)
        self.p['csv_path'].value=None;self.update()
        self.assertEqual(self.p['csv_path'].valueAsText,os.path.join(self.temp.name,'Other.csv'))

    def test_folder_change_updates_generated_not_manual(self):
        self.update();self.p['csv_path'].value=os.path.join(self.temp.name,'manual.csv');self.update()
        self.p['output_folder'].value=os.path.join(self.temp.name,'space folder');self.update()
        self.assertIn('space folder',self.p['annotated_video_path'].valueAsText)
        self.assertNotIn('space folder',self.p['csv_path'].valueAsText)

    def test_path_normalization(self):
        self.update();self.p['csv_path'].value=self.p['csv_path'].valueAsText.replace('\\','/').upper()
        self.p['run_name'].value='Normalized';self.update()
        self.assertEqual(naming.canonical(self.p['csv_path'].valueAsText),naming.canonical(os.path.join(self.temp.name,'Normalized.csv')))

    def test_invalid_names(self):
        for name in ['has spaces','bad/name','bad:name','x'*101,'123test','CON','NUL','']:
            with self.subTest(name=name),self.assertRaises(ValueError):naming.build_output_names(name,self.temp.name)
        self.assertTrue(naming.build_output_names('A_'+'x'*98,self.temp.name))

    def test_enabled_output_collision_validation(self):
        self.update();path=self.p['csv_path'].valueAsText;Path(path).write_text('keep')
        with patch('model_package.inspect_package'):
            self.tool.updateMessages(self.params)
        self.assertIn('already exists',self.p['csv_path'].message)
        self.p['export_csv'].value=False
        with patch('model_package.inspect_package'):self.tool.updateMessages(self.params)
        self.assertFalse(self.p['csv_path'].hasError())

    def test_feature_names_and_parameter_positions(self):
        expected=naming.build_output_names('Truck_Test_01',self.temp.name)
        naming.validate_feature_path(expected['geospatial_path'])
        with self.assertRaises(ValueError):naming.validate_feature_path(os.path.join(self.temp.name,'wrong.shp'))
        self.assertEqual([p.name for p in self.params[:4]],['in_video','text_prompt','output_folder','model_package'])
        self.assertFalse(self.p['export_geospatial'].value)
        self.assertNotIn('export_features', self.p)
        self.assertNotIn('out_feature_class', self.p)


class SpatialTests(unittest.TestCase):
    def test_decode_offset_and_absolute_corners(self):
        fields=[(2,(123456789).to_bytes(8,'big')),(23,signed(40,90,4)),(24,signed(-105,180,4))]
        for i,(lon,lat) in enumerate(CORNERS):
            fields.extend([(26+2*i,signed(lat-40,.075,2)),(27+2*i,signed(lon+105,.075,2))])
        r=decode_local_set(packet(fields),.1)
        self.assertEqual(r.values['precision_timestamp'],'123456789')
        self.assertTrue(np.allclose(r.corners,CORNERS,atol=.000003))
        full=[(82+2*i,signed(lat,90,4)) for i,(lon,lat) in enumerate(CORNERS)]
        full += [(83+2*i,signed(lon,180,4)) for i,(lon,lat) in enumerate(CORNERS)]
        self.assertTrue(np.allclose(decode_local_set(packet(full),.1).corners,CORNERS))

    def test_corrupt_checksum_and_missing_fields(self):
        data=bytearray(packet([(23,signed(40,90,4))]));data[-1]^=1
        with self.assertRaises(MetadataUnavailable):decode_local_set(bytes(data),0)
        self.assertIsNone(decode_local_set(packet([(23,signed(40,90,4))]),0).corners)

    def test_invalid_full_corner_does_not_fall_back(self):
        fields=[(82,b'\x80\0\0\0'),(23,signed(40,90,4)),(24,signed(-105,180,4))]
        self.assertIsNone(decode_local_set(packet(fields),0).corners)

    def test_projection_corners_and_center(self):
        for (x,y),expected in zip([(0,0),(100,0),(100,100),(0,100)],CORNERS):
            np.testing.assert_allclose(project_centroid(CORNERS,x,y,101,101),expected,atol=1e-10)
        np.testing.assert_allclose(project_centroid(CORNERS,50,50,101,101),(-104.9995,39.9995),atol=1e-10)
        self.assertNotEqual(project_centroid(CORNERS,10,50,101,101),project_centroid(CORNERS,90,50,101,101))

    def test_projective_not_bilinear(self):
        corners=[(0.,.002),(.002,.002),(.0015,0.),(.0005,0.)]
        lon,lat=project_centroid(corners,50,50,101,101)
        self.assertAlmostEqual(lon,.001,places=8)
        self.assertAlmostEqual(lat,.002/3,places=8)

    def test_invalid_geometry(self):
        for corners in [[CORNERS[0]]*4,[CORNERS[i] for i in (0,2,1,3)],[(179,1),(-179,1),(-179,0),(179,0)]]:
            with self.assertRaises(MetadataUnavailable):project_centroid(corners,50,50,101,101)
        with self.assertRaises(MetadataUnavailable):project_centroid(CORNERS,-1,5,101,101)

    def test_timestamp_gaps_partial_and_no_future(self):
        m=MetadataMatcher([CornerMetadata(.1,CORNERS,{}),CornerMetadata(.3,None,{}),CornerMetadata(1.,CORNERS,{})])
        self.assertIsNone(m.at(0));self.assertIsNotNone(m.at(.2));self.assertIsNone(m.at(.31))
        self.assertIsNone(m.at(.9));self.assertIsNotNone(m.at(1.));self.assertIsNone(m.at(1.3))
        self.assertIsNone(m.at(None));self.assertIsNone(m.at(.5))

    def test_real_feature_output_schema_and_csv_agreement(self):
        with tempfile.TemporaryDirectory(prefix='SAM geo ') as temp:
            rows=observations();out=os.path.join(temp,'result.gdb','Run_Detections')
            path,counts=export_geospatial_points(rows,out,video='sample.ts',run_name='Run',
                probe=SimpleNamespace(has_klv=True,width=101,height=101),records=[CornerMetadata(0,CORNERS,{'precision_timestamp':'123'})])
            self.assertEqual(counts['written'],3);self.assertEqual(arcpy.Describe(path).spatialReference.factoryCode,4326)
            data=list(arcpy.da.SearchCursor(path,['track_id','frame_number','pixel_x','longitude','latitude','status']))
            self.assertEqual([r[0] for r in data],[4,7,4]);self.assertNotEqual(data[0][3],data[1][3])
            csvpath=os.path.join(temp,'Run.csv');arcgis_export.export_csv(rows,csvpath)
            visible=pd.read_csv(csvpath).query("status == 'VISIBLE'")
            self.assertEqual(list(visible.track_id),[r[0] for r in data])
            for r,row in zip(data,visible.itertuples()):self.assertEqual(r[2],(row.xmin+row.xmax)/2)
            with self.assertRaises(FileExistsError):export_geospatial_points(rows,out,video='sample.ts',run_name='Run',probe=SimpleNamespace(has_klv=True),records=[])
            arcpy.management.ClearWorkspaceCache()

    def test_missing_timestamps_and_plain_video_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            rows=observations();rows['source_timestamp']=None
            for klv in [False,True]:
                with patch('geospatial_export.iter_metadata',side_effect=AssertionError('Must not parse unalignable metadata')):
                    path,counts=export_geospatial_points(rows,os.path.join(temp,'result.gdb','Run_Detections'),
                        video='sample.mp4',run_name='Run',probe=SimpleNamespace(has_klv=klv,width=101,height=101))
                self.assertIsNone(path);self.assertFalse(os.path.exists(os.path.join(temp,'result.gdb')))


class OverwriteTests(unittest.TestCase):
    def test_existing_csv_video_and_pixel_features_survive(self):
        with tempfile.TemporaryDirectory() as temp:
            for suffix,call in [('csv',lambda p:arcgis_export.export_csv(observations(),p)),
                                ('mp4',lambda p:arcgis_export.AnnotatedVideoWriter(p,32,32,30))]:
                p=Path(temp)/('existing.'+suffix);p.write_bytes(b'original')
                with self.assertRaises(FileExistsError):call(str(p))
                self.assertEqual(p.read_bytes(),b'original')
            gdb=os.path.join(temp,'test.gdb');arcpy.management.CreateFileGDB(temp,'test.gdb')
            arcpy.management.CreateFeatureclass(gdb,'pixels','POINT')
            with arcpy.EnvManager(overwriteOutput=True),self.assertRaises(FileExistsError):
                arcgis_export.export_feature_class(observations(),gdb,'pixels')
            self.assertTrue(arcpy.Exists(os.path.join(gdb,'pixels')))
            arcpy.management.ClearWorkspaceCache()

    def test_pixel_output_defaults_to_centroid_points(self):
        import inspect
        import pipeline
        self.assertEqual(inspect.signature(pipeline.run_pipeline_and_export).parameters['geometry'].default, 'point')
        with tempfile.TemporaryDirectory() as temp:
            rows = observations()
            rows.loc[rows.status != 'VISIBLE', ['xmin','ymin','xmax','ymax']] = None
            path = arcgis_export.export_feature_class(rows, os.path.join(temp, 'points.gdb'), 'pixels')
            self.assertEqual(arcpy.Describe(path).shapeType, 'Point')
            self.assertEqual(arcpy.Describe(path).spatialReference.name, 'Unknown')
            data = list(arcpy.da.SearchCursor(path, ['track_id','xmin','ymin','xmax','ymax','SHAPE@XY']))
            arcpy.management.ClearWorkspaceCache()
            self.assertEqual(len(data), len(rows))
            for actual, expected in zip(data, rows.itertuples()):
                self.assertEqual(actual[0], expected.track_id)
                if expected.status == 'VISIBLE':
                    self.assertEqual(actual[1:5], (expected.xmin,expected.ymin,expected.xmax,expected.ymax))
                    # Unknown-reference geodatabase geometry is quantized at storage resolution.
                    self.assertAlmostEqual(actual[5][0], (expected.xmin+expected.xmax)/2, delta=.001)
                    self.assertAlmostEqual(actual[5][1], -(expected.ymin+expected.ymax)/2, delta=.001)
                else:
                    self.assertTrue(all(v is None or math.isnan(v) for v in actual[5]))
            arcpy.management.ClearWorkspaceCache()

    def test_disk_history_csv_still_streams(self):
        from result_store import DiskHistory
        with tempfile.TemporaryDirectory() as temp:
            frame=observations();rows=DiskHistory(list(frame.columns))
            try:
                for record in frame.to_dict('records'):rows.append(record)
                target=os.path.join(temp,'new.csv');arcgis_export.export_csv(rows,target)
                self.assertEqual(len(pd.read_csv(target)),4)
            finally:rows.close()

class PipelineTests(unittest.TestCase):
    def test_output_collision_precedes_video_and_model_loading(self):
        import pipeline
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'keep.csv';output.write_text('existing')
            with patch.object(pipeline,'validate_video') as validate, patch.object(pipeline,'SAM31VideoRuntime') as model:
                with self.assertRaises(FileExistsError):
                    pipeline.run_pipeline_and_export('source.ts','truck',export_feature_class_flag=False,
                        save_annotated_video=False,out_csv_path=str(output))
            validate.assert_not_called();model.assert_not_called()
            self.assertEqual(output.read_text(),'existing')

    def test_geo_only_and_missing_metadata_keep_processing_successful(self):
        import pipeline
        tm=Mock();tm.rows=observations();tm.performance_summary.return_value={};tm.peak_rss_mb=0
        probe=SimpleNamespace(has_klv=False,width=101,height=101)
        with tempfile.TemporaryDirectory() as temp, patch.object(pipeline,'validate_video',return_value=probe), \
             patch.object(pipeline,'run_full_pipeline',return_value=(tm,[])) as process:
            result=pipeline.run_pipeline_and_export('plain.mp4','truck',export_feature_class_flag=False,
                save_annotated_video=False,export_csv_flag=False,runtime=Mock(),warm_up_detector=False,
                return_results=False,geospatial_path=os.path.join(temp,'outputs.gdb','Trial_Detections'),run_name='Trial')
            self.assertTrue(result['success']);self.assertIsNone(result['geospatial_path'])
            self.assertTrue(process.call_args.kwargs['bounded_history']);tm.close_history.assert_called_once()

    def test_missing_geo_does_not_discard_csv(self):
        import pipeline
        tm=Mock();tm.rows=observations();tm.performance_summary.return_value={};tm.peak_rss_mb=0
        with tempfile.TemporaryDirectory() as temp, patch.object(pipeline,'validate_video',return_value=SimpleNamespace(has_klv=False)), \
             patch.object(pipeline,'run_full_pipeline',return_value=(tm,[])):
            result=pipeline.run_pipeline_and_export('plain.mp4','truck',export_feature_class_flag=False,
                save_annotated_video=False,out_csv_path=os.path.join(temp,'Trial.csv'),
                runtime=Mock(),warm_up_detector=False,return_results=False,
                geospatial_path=os.path.join(temp,'outputs.gdb','Trial_Detections'),run_name='Trial')
            self.assertTrue(result['success']);self.assertTrue(Path(result['csv_path']).exists())
            self.assertIsNone(result['geospatial_path'])

    def test_decoder_loop_and_model_cache_sources_unchanged(self):
        import ast,subprocess
        before=ast.parse(subprocess.check_output(['git','show','HEAD:src/pipeline.py'],cwd=ROOT).decode('utf-8-sig'))
        after=ast.parse((ROOT/'src/pipeline.py').read_text(encoding='utf-8-sig'))
        old=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name=='run_full_pipeline')
        new=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='run_full_pipeline')
        self.assertEqual(ast.dump(old),ast.dump(new))
        for name in ['src/track_manager.py','src/sam31_video_runtime.py','src/video_reader.py',
                     'src/detector_cache.py','src/package_model_cache.py','src/streaming_session_pool.py',
                     'SAM31_ObjectTracker/sam31_runtime/sam31_session.py']:
            old=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).replace(b'\r\n',b'\n')
            self.assertEqual(old,(ROOT/name).read_bytes().replace(b'\r\n',b'\n'))


class ToolboxLoadingTests(unittest.TestCase):
    def test_callbacks_after_arcgis_restores_import_path(self):
        module = runpy.run_path(str(ROOT / 'SAM31_TextPromptTracker.pyt'))
        original_path = list(sys.path)
        cached = sys.modules.pop('output_naming', None)
        try:
            sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or '.'))
                           != os.path.normcase(str(ROOT / 'src'))]
            tool = module['SAM31TextPromptTrackingTool']()
            parameters = tool.getParameterInfo()
            # getParameterInfo can temporarily add src again; callbacks must not need it.
            sys.path[:] = [p for p in original_path if os.path.normcase(os.path.abspath(p or '.'))
                           != os.path.normcase(str(ROOT / 'src'))]
            tool.updateParameters(parameters)
            tool.updateMessages(parameters)
            self.assertEqual(len(parameters), 27)
            self.assertEqual(tool.label, 'Track Objects by Text Prompt (SAM 3.1)')
            toolbox = module['Toolbox']()
            self.assertEqual(toolbox.label, 'SAM 3.1 Text-Prompt Tracking')
            self.assertEqual(toolbox.alias, 'sam31TextPromptTracking')
        finally:
            sys.path[:] = original_path
            if cached is not None:
                sys.modules['output_naming'] = cached


class FrameTimingTests(unittest.TestCase):
    def test_repeated_frames_and_gaps_use_actual_times(self):
        from frame_timestamps import FrameTimestampReader
        with patch('frame_timestamps.iter_frame_times', return_value=iter([0., .04, .11, .15])):
            reader = FrameTimestampReader('v', 101, 101)
            self.assertEqual(reader.at(0), 0.)
            self.assertEqual(reader.at(0), 0.)
            self.assertEqual(reader.at(2), .11)
            with self.assertRaises(MetadataUnavailable):reader.at(1)
            with self.assertRaises(MetadataUnavailable):reader.at(4)

    def test_recovered_points_preserve_original_timestamps(self):
        rows = observations();rows['source_timestamp'] = None
        with tempfile.TemporaryDirectory() as temp, patch('geospatial_export.FrameTimestampReader') as timing:
            timing.return_value.at.side_effect = lambda frame: [.1, .2][frame]
            path, counts = export_geospatial_points(rows, os.path.join(temp,'test.gdb','points'),
                video='sample.ts', run_name='Run', probe=SimpleNamespace(has_klv=True,width=101,height=101),
                records=[CornerMetadata(0,CORNERS,{})])
            with arcpy.da.SearchCursor(path,['track_id','source_timestamp','georef_timestamp','timestamp_source']) as cursor:
                values=list(cursor)
            self.assertEqual(values,[(4,None,.1,'ffprobe_frame_pts'),(7,None,.1,'ffprobe_frame_pts'),(4,None,.2,'ffprobe_frame_pts')])
            self.assertEqual(counts['written'],3)
            timing.return_value.close.assert_called_once()
            self.assertTrue(rows.source_timestamp.isna().all())
            arcpy.management.ClearWorkspaceCache()

    def test_timing_disagreement_stops_export(self):
        rows=observations();rows.loc[1:,'source_timestamp']=None
        with tempfile.TemporaryDirectory() as temp, patch('geospatial_export.FrameTimestampReader') as timing:
            timing.return_value.at.return_value=8.
            path, counts=export_geospatial_points(rows,os.path.join(temp,'test.gdb','points'),
                video='sample.ts',run_name='Run',probe=SimpleNamespace(has_klv=True,width=101,height=101),
                records=[CornerMetadata(0,CORNERS,{})])
            self.assertEqual(counts['written'],1)
            self.assertEqual(counts['metadata_read_stopped'],1)
            timing.return_value.close.assert_called_once()
            arcpy.management.ClearWorkspaceCache()

    def test_frame_parser_rejects_invalid_pts_and_decode_errors(self):
        import io
        from frame_timestamps import iter_frame_times
        info={'streams':[{'codec_type':'video','index':0,'start_time':'100','width':101,'height':101}]}
        for lines in ['pts_time=100|width=101|height=101\npts_time=100|width=101|height=101\n',
                      'pts_time=101|width=101|height=101\n',
                      'pts_time=N/A|width=101|height=101\n',
                      '[h264] decode error\n', 'pts_time=100|width=99|height=101\n']:
            proc=Mock();proc.stdout=io.StringIO(lines);proc.poll.return_value=0;proc.wait.return_value=0
            with patch('frame_timestamps._probe_with_ffprobe',return_value=info),patch('frame_timestamps.subprocess.Popen',return_value=proc):
                with self.assertRaises(MetadataUnavailable):list(iter_frame_times('v',101,101))
                self.assertTrue(proc.stdout.closed)

    def test_frame_parser_nonzero_origin_and_variable_intervals(self):
        import io
        from frame_timestamps import iter_frame_times
        info={'streams':[{'codec_type':'video','index':0,'start_time':'100','width':101,'height':101}]}
        proc=Mock();proc.stdout=io.StringIO('pts_time=100|width=101|height=101\npts_time=100.04|width=101|height=101\npts_time=100.11|width=101|height=101\n');proc.poll.return_value=0;proc.wait.return_value=0
        with patch('frame_timestamps._probe_with_ffprobe',return_value=info),patch('frame_timestamps.subprocess.Popen',return_value=proc):
            np.testing.assert_allclose(list(iter_frame_times('v',101,101)),[0,.04,.11])


class NamingRecoveryTests(unittest.TestCase):
    setUp = NamingTests.setUp
    update = NamingTests.update

    def test_recreated_validator_follows_new_run_name(self):
        self.update()
        self.p['run_name'].value = 'Renamed_Run'
        self.tool._naming = self.tool._naming.__class__()
        self.update()
        for key, path in naming.build_output_names('Renamed_Run', self.temp.name).items():
            self.assertEqual(naming.canonical(self.p[key].valueAsText), naming.canonical(path))

    def test_recreated_validator_preserves_single_custom_path(self):
        self.update()
        custom = os.path.join(self.temp.name, 'my_custom.csv')
        self.p['csv_path'].value = custom
        self.p['run_name'].value = 'Renamed_Run'
        self.tool._naming = self.tool._naming.__class__()
        self.update()
        self.assertEqual(self.p['csv_path'].valueAsText, custom)
        self.assertTrue(self.p['annotated_video_path'].valueAsText.endswith('Renamed_Run_annotated.mp4'))
        self.assertTrue(self.p['geospatial_path'].valueAsText.endswith('Renamed_Run_Detections'))

    def test_overwrite_warning_respects_environment(self):
        self.update()
        Path(self.p['csv_path'].valueAsText).write_text('existing')
        with arcpy.EnvManager(overwriteOutput=True):
            self.tool.updateMessages(self.params)
            self.assertFalse(self.p['csv_path'].hasError())
            self.assertIn('will be replaced', self.p['csv_path'].message)
        with arcpy.EnvManager(overwriteOutput=False):
            self.tool.updateMessages(self.params)
            self.assertTrue(self.p['csv_path'].hasError())


class ReplacementTests(unittest.TestCase):
    def test_csv_replacement_and_failed_write_preserve_previous(self):
        with tempfile.TemporaryDirectory() as temp, arcpy.EnvManager(overwriteOutput=True):
            path=Path(temp)/'results.csv';path.write_text('previous')
            broken=Mock();broken.to_csv.side_effect=RuntimeError('write failed')
            with self.assertRaises(RuntimeError):arcgis_export.export_csv(broken,str(path))
            self.assertEqual(path.read_text(),'previous')
            arcgis_export.export_csv(observations(),str(path))
            self.assertEqual(len(pd.read_csv(path)),4)
            self.assertEqual(len(list(Path(temp).iterdir())),1)

    def test_video_replacement_writes_readable_video(self):
        import cv2
        with tempfile.TemporaryDirectory() as temp, arcpy.EnvManager(overwriteOutput=True):
            path=Path(temp)/'annotated.mp4';path.write_bytes(b'previous')
            writer=arcgis_export.AnnotatedVideoWriter(str(path),32,32,10)
            writer.write(np.zeros((32,32,3),dtype=np.uint8));writer.close()
            cap=cv2.VideoCapture(str(path))
            try:self.assertTrue(cap.read()[0])
            finally:cap.release()

    def test_geospatial_replaces_instead_of_appending(self):
        with tempfile.TemporaryDirectory() as temp, arcpy.EnvManager(overwriteOutput=True):
            path=os.path.join(temp,'results.gdb','points')
            kwargs=dict(video='sample.ts',run_name='Run',probe=SimpleNamespace(has_klv=True,width=101,height=101))
            export_geospatial_points(observations(),path,records=[CornerMetadata(0,CORNERS,{})],**kwargs)
            export_geospatial_points(observations().iloc[:1],path,records=[CornerMetadata(0,CORNERS,{})],**kwargs)
            self.assertEqual(int(arcpy.management.GetCount(path)[0]),1)
            result,_=export_geospatial_points(observations(),path,records=[],**kwargs)
            self.assertIsNone(result)
            self.assertEqual(int(arcpy.management.GetCount(path)[0]),1)
            arcpy.management.ClearWorkspaceCache()

if __name__=='__main__':unittest.main()
