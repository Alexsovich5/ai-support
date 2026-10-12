import os
import stat

import pytest

from src import keygen


def test_creates_both_key_files_read_only(tmp_path):
    assert keygen.main([str(tmp_path)]) == 0
    ingest = tmp_path / "ingest_api_key"
    openai = tmp_path / "openai_api_key"
    assert stat.S_IMODE(os.stat(ingest).st_mode) == 0o400
    assert len(ingest.read_text().strip()) >= 43
    assert openai.read_text().strip() == keygen.STUB_OPENAI_KEY


def test_keys_are_random_per_install(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    keygen.ensure_keys(tmp_path / "a")
    keygen.ensure_keys(tmp_path / "b")
    assert (tmp_path / "a" / "ingest_api_key").read_text() != (tmp_path / "b" / "ingest_api_key").read_text()


def test_existing_files_are_kept(tmp_path):
    keygen.ensure_keys(tmp_path)
    before = (tmp_path / "ingest_api_key").read_text()
    assert keygen.ensure_keys(tmp_path) == []
    assert (tmp_path / "ingest_api_key").read_text() == before


def test_values_are_never_printed(tmp_path, capsys):
    keygen.main([str(tmp_path)])
    out = capsys.readouterr().out
    assert (tmp_path / "ingest_api_key").read_text().strip() not in out
    assert "ingest_api_key" in out


def test_symlinked_directory_or_file_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    assert keygen.main([str(link)]) == 1
    assert not any(real.iterdir())

    target = tmp_path / "elsewhere"
    os.symlink(target, real / "ingest_api_key")
    with pytest.raises(ValueError):
        keygen.ensure_keys(real)
    assert not target.exists()
