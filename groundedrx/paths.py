"""Vector-store path resolution — Colab vs. Kaggle vs. local, auto-detected.

Split into a pure filesystem search (`find_qdrant_store`, fully unit-testable
against a plain temp directory, no real Kaggle/Colab environment needed) and
the environment-dependent resolution with side effects (`resolve_store_path`,
which actually copies/unzips into a writable location).

Ported from GroundedRx_Colab.ipynb's Setup cell. The notebook's Colab branch
used IPython's `!unzip` shell magic, which only exists inside a notebook;
here it's Python's stdlib `zipfile` instead, so this module has no shell
dependency and works as plain importable code.
"""

import glob
import os
import shutil
import zipfile
from typing import Optional


def find_qdrant_store(search_root: str) -> Optional[str]:
    """Return the directory containing meta.json under search_root, or None.

    Pure filesystem search -- no Kaggle/Colab dependency, fully testable
    against a plain temp directory.
    """
    matches = glob.glob(os.path.join(search_root, "**", "meta.json"), recursive=True)
    return os.path.dirname(matches[0]) if matches else None


def is_kaggle() -> bool:
    """True inside a Kaggle notebook session (Colab has no /kaggle/input)."""
    return os.path.exists("/kaggle/input")


def resolve_store_path(
    work_dir: Optional[str] = None,
    kaggle_input: str = "/kaggle/input",
    colab_zip: str = "/content/qdrant_db_archive.zip",
    on_kaggle: Optional[bool] = None,
    explicit_path: Optional[str] = None,
) -> str:
    """
    Resolve (and materialize, if needed) a writable Qdrant store directory.

    Docker / on-premises deployment: pass `explicit_path`, or set the
    GROUNDEDRX_QDRANT_PATH environment variable, to a directory containing
    an already-extracted store (mounted in as a volume). This short-circuits
    all Colab/Kaggle detection entirely -- there is no `/kaggle/input` or
    `/content` inside a container, so without this, resolution would
    silently fall through to the Colab branch and fail looking for a zip
    that was never there.

    On Kaggle: /kaggle/input/ is read-only but Qdrant's local client needs to
    write a .lock file to open the store, and Kaggle auto-extracts any .zip
    uploaded as a Dataset (there's never a raw zip to unzip, unlike Colab) --
    so this searches for meta.json (the store's own marker file) under
    kaggle_input, then copies the whole containing folder into work_dir,
    skipping the copy if it already exists (idempotent across re-runs in the
    same session).

    On Colab: colab_zip is a raw zip that genuinely needs unzipping into
    work_dir.

    `on_kaggle` is exposed as a parameter (defaults to real auto-detection
    via `is_kaggle()`) purely so this function is testable without an actual
    Kaggle environment.
    """
    if explicit_path is None:
        explicit_path = os.environ.get("GROUNDEDRX_QDRANT_PATH")
    if explicit_path is not None:
        if not os.path.exists(os.path.join(explicit_path, "meta.json")):
            raise FileNotFoundError(
                f"GROUNDEDRX_QDRANT_PATH={explicit_path} has no meta.json -- "
                "mount the extracted qdrant_db_archive store there."
            )
        return explicit_path

    if on_kaggle is None:
        on_kaggle = is_kaggle()
    if work_dir is None:
        work_dir = "/kaggle/working" if on_kaggle else "/content"
    storage = os.path.join(work_dir, "qdrant_storage")

    if on_kaggle:
        source_dir = find_qdrant_store(kaggle_input)
        if source_dir is None:
            raise FileNotFoundError(
                "Qdrant store (meta.json) not found under "
                f"{kaggle_input} -- attach the qdrant_db_archive Dataset to "
                "this notebook first (Add Data -> search your dataset -> Add)."
            )
        if not os.path.exists(storage):
            shutil.copytree(source_dir, storage)
    else:
        if not os.path.exists(storage):
            with zipfile.ZipFile(colab_zip) as zf:
                zf.extractall(storage)

    return storage
