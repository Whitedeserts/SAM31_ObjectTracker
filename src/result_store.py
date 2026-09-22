"""Append-only temporary result history with bounded resident memory."""
import csv
import pickle
import sqlite3
import tempfile
from collections import namedtuple


class DiskHistory:
    def __init__(self, columns=None):
        self.columns = columns
        self._directory = tempfile.TemporaryDirectory(prefix="sam31-results-")
        self._db = sqlite3.connect(self._directory.name + "/history.sqlite")
        self._db.execute("PRAGMA cache_size=-1024")
        self._db.execute("CREATE TABLE history (id INTEGER PRIMARY KEY, value BLOB)")
        self._count = 0

    def append(self, value):
        self._db.execute("INSERT INTO history VALUES (?, ?)",
                         (self._count, pickle.dumps(value)))
        self._count += 1
        if self._count % 256 == 0:
            self._db.commit()

    def __len__(self):
        return self._count

    def __iter__(self):
        for (value,) in self._db.execute("SELECT value FROM history ORDER BY id"):
            yield pickle.loads(value)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            if step != 1:
                raise ValueError("history only supports contiguous slices")
            return [pickle.loads(row[0]) for row in self._db.execute(
                "SELECT value FROM history WHERE id >= ? AND id < ? ORDER BY id", (start, stop))]
        if index < 0:
            index += self._count
        row = self._db.execute("SELECT value FROM history WHERE id = ?", (index,)).fetchone()
        if row is None:
            raise IndexError(index)
        return pickle.loads(row[0])

    def itertuples(self, index=False):
        row_type = namedtuple("TrackRow", self.columns)
        for row in self:
            yield row_type(*(row.get(key) for key in self.columns))

    def to_csv(self, path, index=False):
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self)

    def close(self):
        self._db.close()
        self._directory.cleanup()
