"""Path resolution logic -- pure filesystem operations, no real Kaggle/Colab
environment needed. Uses pytest's tmp_path fixture throughout."""

import json
import zipfile

from groundedrx.paths import find_qdrant_store, resolve_store_path


def _make_fake_store(root):
    """A minimal fake Qdrant store directory: just needs meta.json to exist
    for find_qdrant_store's marker-file search."""
    store = root / "some" / "nested" / "dataset_dir"
    store.mkdir(parents=True)
    (store / "meta.json").write_text(json.dumps({"vectors": {"size": 1024}}))
    (store / "collection").mkdir()
    (store / "collection" / "dummy.txt").write_text("payload data")
    return store


def test_find_qdrant_store_locates_meta_json(tmp_path):
    store = _make_fake_store(tmp_path)
    found = find_qdrant_store(str(tmp_path))
    assert found == str(store)


def test_find_qdrant_store_returns_none_when_absent(tmp_path):
    assert find_qdrant_store(str(tmp_path)) is None


def test_resolve_store_path_kaggle_copies_into_work_dir(tmp_path):
    kaggle_input = tmp_path / "input"
    work_dir = tmp_path / "working"
    _make_fake_store(kaggle_input)

    result = resolve_store_path(
        work_dir=str(work_dir), kaggle_input=str(kaggle_input), on_kaggle=True
    )

    assert result == str(work_dir / "qdrant_storage")
    assert (work_dir / "qdrant_storage" / "meta.json").exists()
    assert (work_dir / "qdrant_storage" / "collection" / "dummy.txt").exists()


def test_resolve_store_path_kaggle_is_idempotent(tmp_path):
    kaggle_input = tmp_path / "input"
    work_dir = tmp_path / "working"
    _make_fake_store(kaggle_input)

    first = resolve_store_path(work_dir=str(work_dir), kaggle_input=str(kaggle_input), on_kaggle=True)
    # second call must not raise (shutil.copytree would raise FileExistsError
    # if the "skip if it already exists" guard were missing)
    second = resolve_store_path(work_dir=str(work_dir), kaggle_input=str(kaggle_input), on_kaggle=True)
    assert first == second


def test_resolve_store_path_kaggle_missing_store_raises(tmp_path):
    work_dir = tmp_path / "working"
    empty_input = tmp_path / "input"
    empty_input.mkdir()
    try:
        resolve_store_path(work_dir=str(work_dir), kaggle_input=str(empty_input), on_kaggle=True)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_resolve_store_path_colab_extracts_zip(tmp_path):
    work_dir = tmp_path / "content"
    zip_path = tmp_path / "qdrant_db_archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("meta.json", json.dumps({"vectors": {"size": 1024}}))
        zf.writestr("collection/dummy.txt", "payload data")

    result = resolve_store_path(work_dir=str(work_dir), colab_zip=str(zip_path), on_kaggle=False)

    assert result == str(work_dir / "qdrant_storage")
    assert (work_dir / "qdrant_storage" / "meta.json").exists()
