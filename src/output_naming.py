"""Run-name ownership and output paths shared by validation and execution."""
import os
import re
from datetime import datetime

OUTPUT_TOGGLES = {
    'csv_path': 'export_csv',
    'annotated_video_path': 'save_annotated_video',
    'geospatial_path': 'export_geospatial',
}


def canonical(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(path))) if path else None


def validate_run_name(name):
    if not name or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,99}', name):
        raise ValueError('Use 1-100 letters, numbers or underscores; start with a letter.')
    if name.upper() in {'CON', 'PRN', 'AUX', 'NUL', *('COM'+str(i) for i in range(1,10)),
                       *('LPT'+str(i) for i in range(1,10))}:
        raise ValueError('Run Name cannot be a Windows reserved filename.')
    return name


def build_output_names(run_name, output_folder):
    validate_run_name(run_name)
    folder = os.path.abspath(output_folder)
    gdb = os.path.join(folder, 'SAM31_Tracks.gdb')
    return dict(csv_path=os.path.join(folder, run_name + '.csv'),
                annotated_video_path=os.path.join(folder, run_name + '_annotated.mp4'),
                geospatial_path=os.path.join(gdb, run_name + '_Detections'))


def output_exists(path):
    if not path:
        return False
    if os.path.lexists(path):
        return True
    if '.gdb' + os.sep in os.path.normpath(path).lower():
        import arcpy
        return bool(arcpy.Exists(path))
    return False


def overwrite_allowed():
    try:
        import arcpy
        return bool(arcpy.env.overwriteOutput)
    except ImportError:
        return False


def require_output_writable(path):
    if output_exists(path) and not overwrite_allowed():
        raise FileExistsError('Output already exists: ' + str(path) +
                              '. Enable ArcGIS Pro overwrite existing outputs, or choose a new Run Name or path.')


def require_new_output(path):
    """Legacy export API: callers requiring a fresh dataset keep this guard."""
    if output_exists(path):
        raise FileExistsError('Output already exists: ' + str(path))


def validate_feature_path(path):
    import arcpy
    parent, name = os.path.split(os.path.normpath(path))
    if not parent.lower().endswith('.gdb') or not name:
        raise ValueError('Choose a feature class directly inside a file geodatabase (.gdb).')
    if arcpy.ValidateTableName(name, parent) != name:
        raise ValueError('Invalid geodatabase feature-class name: ' + name)


class OutputNamingState:
    """Remember only values actually generated here; overrides remain user-owned.

    Canonical path comparison handles ArcGIS case/separator normalization. A new
    tool instance can recover a shared generated naming pattern from its paths.
    """
    def __init__(self):
        self.video = None
        self.run_name = None
        self.paths = {}
        self.initialized = False

    def recover(self, parameters):
        """Recover only a matching cohort of at least two standard output paths.

        ArcGIS can recreate validation objects. A single arbitrary path is not
        evidence of automatic ownership; differently named/located overrides stay.
        """
        folder = parameters['output_folder'].valueAsText
        if not folder:
            return
        candidates = {}
        suffixes = {'csv_path': '.csv', 'annotated_video_path': '_annotated.mp4',
                    'geospatial_path': '_Detections'}
        for key, suffix in suffixes.items():
            value = parameters[key].valueAsText
            if not value:
                continue
            filename = os.path.basename(os.path.normpath(value))
            if not filename.lower().endswith(suffix.lower()):
                continue
            name = filename[:-len(suffix)]
            try:
                defaults = build_output_names(name, folder)
            except ValueError:
                continue
            if canonical(value) == canonical(defaults[key]):
                candidates.setdefault(name.lower(), []).append(key)
        for keys in candidates.values():
            if len(keys) >= 2:
                for key in keys:
                    self.paths[key] = parameters[key].valueAsText

    def update(self, parameters):
        p = parameters
        if not self.initialized:
            self.recover(p)
            self.initialized = True
        video = canonical(p['in_video'].valueAsText)
        name = p['run_name'].valueAsText
        if video and (not name or (name == self.run_name and video != self.video)):
            stem = re.sub(r'[^A-Za-z0-9_]', '_', os.path.splitext(os.path.basename(video))[0])[:60]
            name = 'Tracks_' + stem + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
            p['run_name'].value = name
            self.run_name = name
        elif name != self.run_name:
            self.run_name = None
        self.video = video
        defaults = {}
        folder = p['output_folder'].valueAsText
        if folder and name:
            try:
                defaults = build_output_names(name, folder)
            except ValueError:
                pass  # updateMessages reports the invalid Run Name without mutating paths.
        for key, toggle in OUTPUT_TOGGLES.items():
            if key not in p:
                continue
            parameter = p[key]
            parameter.enabled = bool(p[toggle].value)
            current = parameter.valueAsText
            if key in defaults and (not current or canonical(current) == canonical(self.paths.get(key))):
                parameter.value = defaults[key]
                self.paths[key] = defaults[key]
            elif current and canonical(current) != canonical(self.paths.get(key)):
                self.paths.pop(key, None)
