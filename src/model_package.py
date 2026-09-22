"""Validate local DLPKs and cache verified assets without importing package code."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import uuid
import zipfile

ASSETS = {
    'model/sam3.1_multiplex.pt': (3502755717, '0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6'),
    'model/bpe_simple_vocab_16e6.txt.gz': (None, '924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a'),
}


class ModelPackageError(ValueError):
    """An incompatible archive or unusable model cache."""


@dataclass(frozen=True)
class ModelAssets:
    checkpoint: str
    vocabulary: str
    fingerprint: str


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cache_root():
    base = os.environ.get('LOCALAPPDATA')
    return (Path(base) if base else Path.home() / '.cache') / 'SAM31_TextPromptTracker' / 'models'


def inspect_package(path):
    """Check structure cheaply; full asset hashing occurs during preparation."""
    path = Path(path)
    if not path.is_file():
        raise ModelPackageError('Model package not found. Select an existing local .dlpk file.')
    try:
        with zipfile.ZipFile(path) as archive:
            names = set()
            for info in archive.infolist():
                # ZipInfo normalizes Windows separators in filename; validate raw names.
                name = info.orig_filename
                parts = name.rstrip('/').split('/')
                if (not parts or '\\' in name or name.startswith('/') or ':' in name
                        or any(not p or p in ('.', '..') or p.endswith((' ', '.'))
                               or any(ord(c) < 32 for c in p)
                               or p.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                                   *('COM' + str(i) for i in range(1, 10)),
                                   *('LPT' + str(i) for i in range(1, 10))} for p in parts)
                        or stat.S_ISLNK(info.external_attr >> 16)):
                    raise ModelPackageError(f'Unsafe archive entry: {name!r}. Select a trusted DLPK.')
                key = name.casefold().rstrip('/')
                if key in names:
                    raise ModelPackageError(f'Duplicate archive entry: {name!r}.')
                names.add(key)
            emds = [n for n in archive.namelist() if n.endswith('.emd') and '/' not in n]
            if len(emds) != 1:
                raise ModelPackageError('Expected one EMD at the DLPK root. Select the SAM31_ObjectTracker package.')
            if archive.getinfo(emds[0]).file_size > 1024 * 1024:
                raise ModelPackageError('EMD metadata is too large.')
            emd = json.loads(archive.read(emds[0]))
            if (not isinstance(emd, dict) or emd.get('ModelVersion') != '3.1' or emd.get('ModelType') != 'ObjectTracking'
                    or emd.get('ModelFile') != 'model/sam3.1_multiplex.pt'):
                raise ModelPackageError('Incompatible model. This release requires the pinned SAM 3.1 Object Multiplex package.')
            for name, (size, _) in ASSETS.items():
                if name not in archive.namelist():
                    raise ModelPackageError(f'Missing model asset: {name}. Select a complete SAM 3.1 DLPK.')
                info = archive.getinfo(name)
                if (info.flag_bits & 1 or info.file_size <= 0 or
                        (size is not None and info.file_size != size) or
                        (size is None and info.file_size > 16 * 1024 * 1024)):
                    raise ModelPackageError(f'Unsupported or damaged asset: {name}.')
    except ModelPackageError:
        raise
    except (OSError, zipfile.BadZipFile, ValueError, RuntimeError) as exc:
        raise ModelPackageError(f'Cannot read DLPK: {exc}. Select an intact local package.') from exc


def _valid_cache(folder, fingerprint):
    try:
        marker = folder / 'complete.json'
        if folder.is_symlink() or (folder / 'model').is_symlink() or marker.is_symlink() or marker.stat().st_size > 4096:
            return False
        if json.loads(marker.read_text()) != {'package_sha256': fingerprint}:
            return False
        return all(not (folder / name).is_symlink() and sha256(folder / name) == digest
                   for name, (_, digest) in ASSETS.items())
    except (OSError, ValueError):
        return False


def _remove_staging(folder, root):
    # Only delete this operation's staging directory below its resolved cache root.
    if folder.parent.resolve() != root.resolve() or not folder.name.startswith('.preparing-'):
        raise ModelPackageError('Refusing to remove a path outside the model cache.')
    shutil.rmtree(folder)


def prepare_package(path, *, root=None, logger=None):
    """Content-addressed extraction; incomplete directories are never used."""
    path = Path(path).resolve()
    inspect_package(path)
    staging = None
    root = Path(root) if root is not None else cache_root()
    try:
        fingerprint = sha256(path)
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve()
        target = root / fingerprint
        if _valid_cache(target, fingerprint):
            if logger:
                logger.info('Model assets reused from cache; package SHA256 %s', fingerprint)
        else:
            staging = Path(tempfile.mkdtemp(prefix='.preparing-', dir=root))
            with zipfile.ZipFile(path) as archive:
                for name, (_, digest) in ASSETS.items():
                    destination = staging / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    # Names come from the fixed asset allowlist, never archive-controlled paths.
                    with archive.open(name) as source, destination.open('xb') as output:
                        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                    if sha256(destination) != digest:
                        raise ModelPackageError(f'Incompatible or corrupt {name}. Expected the pinned SAM 3.1 asset.')
            if sha256(path) != fingerprint:
                raise ModelPackageError('DLPK changed during preparation. Retry with a stable local file.')
            (staging / 'complete.json').write_text(json.dumps({'package_sha256': fingerprint}))
            if target.exists():
                if _valid_cache(target, fingerprint):
                    _remove_staging(staging, root)
                    staging = None
                else:
                    # Preserve damaged caches for diagnosis; never overwrite open model mappings.
                    target.rename(root / ('.invalid-' + fingerprint + '-' + uuid.uuid4().hex))
            if staging is not None:
                try:
                    staging.rename(target)
                    staging = None
                except FileExistsError:
                    if not _valid_cache(target, fingerprint):
                        raise
            if logger:
                logger.info('Model assets prepared; package SHA256 %s', fingerprint)
        return ModelAssets(str(target / 'model/sam3.1_multiplex.pt'),
                           str(target / 'model/bpe_simple_vocab_16e6.txt.gz'), fingerprint)
    except ModelPackageError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ModelPackageError(f'Model preparation failed: {exc}. Check cache permissions, free disk space, and package integrity.') from exc
    finally:
        if staging is not None and staging.exists():
            _remove_staging(staging, root)
