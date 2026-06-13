import os
import os.path as osp
import time

from .common import Color, colorize


def lock_path_for(h5_path: str) -> str:
    return h5_path + ".lock"


def acquire_lock_or_none(h5_path: str) -> str | None:
    """
    Atomically create a lock file next to the H5 output.
    Returns the lock path, or None when another process owns it.
    """
    lock_path = lock_path_for(h5_path)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} time={time.time()}".encode("utf-8"))
        os.close(fd)
        return lock_path
    except FileExistsError:
        return None


def release_lock(lock_path: str | None) -> None:
    if not lock_path:
        return
    try:
        os.remove(lock_path)
    except OSError:
        pass


def make_out_path_with_index(base_path: str, run_id: str, index: int) -> str:
    root, ext = osp.splitext(base_path)
    return f"{root}_{run_id}_{int(index):04d}{ext}"


def ensure_parent_dir(path: str) -> None:
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def list_input_h5_files(input_dir: str, fallback: str | None = None) -> list[str]:
    files = []
    if input_dir and os.path.isdir(input_dir):
        for name in os.listdir(input_dir):
            if name.lower().endswith(".h5"):
                files.append(osp.join(input_dir, name))

    files.sort()
    if files:
        return files

    if fallback and osp.exists(fallback):
        return [fallback]

    raise FileNotFoundError(
        f"Не найдено ни одного .h5 в '{input_dir}', "
        f"и резервный файл '{fallback}' тоже отсутствует."
    )


def open_input_h5(h5_path: str):
    import h5py

    if not osp.exists(h5_path):
        raise FileNotFoundError(f"Не найден HDF5-файл: {h5_path}")

    db = h5py.File(h5_path, "r")
    print("[H5] open:", h5_path)
    print("[H5] top-level keys:", list(db.keys()))
    return db


def pick_group(db, candidates):
    """Return (group, chosen_name) from candidate top-level H5 group names."""
    for name in candidates:
        if name in db:
            return db[name], name
    raise KeyError(f"Не нашёл ни одну из групп {candidates}. Доступны: {list(db.keys())}")


def read_depth_to_hw_float(depth_item):
    """Convert depth dataset to a float32 (H, W) array."""
    import numpy as np

    depth = np.array(depth_item[:])
    if depth.ndim == 2:
        return depth.astype(np.float32)
    if depth.ndim == 3:
        if depth.shape[0] in (1, 2, 3) and depth.shape[0] != depth.shape[-1]:
            depth = np.moveaxis(depth, 0, -1)
        if depth.shape[2] == 1:
            return depth[..., 0].astype(np.float32)
        if depth.shape[2] >= 3:
            return depth[..., 1].astype(np.float32)
        return depth.mean(axis=2).astype(np.float32)
    raise ValueError(f"Неожиданная форма depth: {depth.shape}")


def seg_with_attrs(seg_ds):
    """Return (seg_float32, area, label), computing attrs when absent."""
    import numpy as np

    seg = np.array(seg_ds[:]).astype("float32")
    if "area" in seg_ds.attrs and "label" in seg_ds.attrs:
        area = seg_ds.attrs["area"]
        label = seg_ds.attrs["label"]
    else:
        labels, counts = np.unique(seg.astype(np.int32), return_counts=True)
        area = counts.astype(np.float32)
        label = labels.astype(np.int32)
    return seg, area, label


def add_res_to_db(imgname, res, db) -> None:
    import h5py
    import numpy as np

    dt = h5py.string_dtype(encoding="utf-8")

    for i, item in enumerate(res):
        dname = f"{imgname}_{i}"
        dset = db["data"].create_dataset(dname, data=item["img"])

        dset.attrs["charBB"] = item["charBB"]
        dset.attrs["wordBB"] = item["wordBB"]

        txt_list = [str(t) for t in item.get("txt", [])]
        dset.attrs.create("txt", np.array(txt_list, dtype=dt))

        lang_list = [str(x) for x in item.get("lang", [])]
        if lang_list:
            dset.attrs.create("lang", np.array(lang_list, dtype=dt))


class H5ResultWriter:
    """Output writer with lock-file protection and file rollover."""

    def __init__(self, base_path: str, run_id: str, viz: bool, max_size_gb: float):
        self.base_path = base_path
        self.run_id = run_id
        self.viz = bool(viz)
        self.max_size_gb = float(max_size_gb)
        self.db = None
        self.path = None
        self.index = 0
        self.lock_path = None

    def __enter__(self):
        if self.viz:
            self.db, self.path, self.lock_path = self._open_locked(self.base_path)
            self.index = 0
        else:
            self.db, self.path, self.index, self.lock_path = self._open_next_free(0)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.db is not None:
                self.db.close()
        finally:
            release_lock(self.lock_path)

    def write(self, imgname, res) -> None:
        add_res_to_db(imgname, res, self.db)
        if not self.viz:
            self._maybe_roll()

    def _open_locked(self, path: str):
        import h5py

        ensure_parent_dir(path)
        lock_path = acquire_lock_or_none(path)
        if lock_path is None:
            raise RuntimeError(f"Output H5 is already in use by another process: {path}")

        try:
            db = h5py.File(path, "w")
            db.create_group("/data")
            print(colorize(Color.GREEN, f"[H5] Opened output file: {path}", bold=True))
            return db, path, lock_path
        except Exception:
            release_lock(lock_path)
            raise

    def _open_next_free(self, start_index: int):
        import h5py

        idx = int(start_index)
        while True:
            path = make_out_path_with_index(self.base_path, self.run_id, idx)
            ensure_parent_dir(path)

            lock_path = acquire_lock_or_none(path)
            if lock_path is None:
                idx += 1
                continue

            try:
                db = h5py.File(path, "w")
                db.create_group("/data")
                print(colorize(Color.GREEN, f"[H5] Opened output file: {path}", bold=True))
                return db, path, idx, lock_path
            except Exception:
                release_lock(lock_path)
                raise

    def _maybe_roll(self) -> None:
        try:
            self.db.flush()
        except Exception:
            pass

        try:
            size_bytes = os.path.getsize(self.path) if osp.exists(self.path) else 0
        except Exception:
            size_bytes = 0

        if size_bytes < self.max_size_gb * (1024 ** 3):
            return

        try:
            self.db.close()
        except Exception:
            pass
        release_lock(self.lock_path)

        self.db, self.path, self.index, self.lock_path = self._open_next_free(self.index + 1)
