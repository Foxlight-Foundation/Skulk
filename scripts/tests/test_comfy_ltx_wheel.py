"""Reproducible wheel assembly and exact-byte publication checks."""

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.build_comfy_ltx_wheel import FILENAME, assemble_wheel
from scripts.verify_comfy_ltx_promotion import verify_artifact

SOURCE = Path(__file__).resolve().parents[2] / "packaging/skulk-comfy-ltx"


def test_builds_repeat_exactly_and_manifest_probe_initializes_no_gpu(tmp_path: Path) -> None:
    first = assemble_wheel(SOURCE, tmp_path / "first")
    second = assemble_wheel(SOURCE, tmp_path / "second")
    assert first.read_bytes() == second.read_bytes()
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    assert verify_artifact(first.parent, digest) == first
    subprocess.run([sys.executable, "-I", "-B", "-c", "import sys;sys.path.insert(0,sys.argv[1]);from skulk_comfy_ltx import manifest_json;assert manifest_json();assert 'torch' not in sys.modules and 'comfy' not in sys.modules", str(first)], check=True, capture_output=True)
    with pytest.raises(FileExistsError):
        assemble_wheel(SOURCE, first.parent)
    with zipfile.ZipFile(first) as archive:
        assert "skulk_comfy_ltx/LICENSE-LTX-2_x" in archive.namelist()
        assert "skulk_comfy_ltx/LICENSE-APACHE-2.0" in archive.namelist()
        assert "skulk_comfy_ltx/NOTICE" in archive.namelist()


@pytest.mark.parametrize("mutation", ["extra", "symlink"])
def test_unreviewed_source_files_are_never_packaged(tmp_path: Path, mutation: str) -> None:
    source = tmp_path / "source"
    shutil.copytree(SOURCE, source, ignore=shutil.ignore_patterns("__pycache__"))
    package = source / "skulk_comfy_ltx"
    if mutation == "extra":
        (package / "extra.py").write_text("raise RuntimeError('unreviewed')")
    else:
        (package / "nodes.py").unlink()
        (package / "nodes.py").symlink_to(SOURCE / "skulk_comfy_ltx/nodes.py")
    with pytest.raises(ValueError, match="reviewed"):
        assemble_wheel(source, tmp_path / "output")


@pytest.mark.parametrize("digest", ["", "../other", "a" * 63, "A" * 64])
def test_promotion_requires_an_explicit_complete_lowercase_digest(tmp_path: Path, digest: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        verify_artifact(tmp_path, digest)


def test_unselected_archive_cannot_be_promoted(tmp_path: Path) -> None:
    path = assemble_wheel(SOURCE, tmp_path)
    with pytest.raises(ValueError, match="selected descriptor"):
        verify_artifact(tmp_path, "a" * 64)
    (tmp_path / "other.whl").write_bytes(b"not selected")
    with pytest.raises(ValueError, match="one bounded"):
        verify_artifact(tmp_path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert path.name == FILENAME


def test_launcher_limits_custom_search_after_operator_path_overrides(tmp_path: Path) -> None:
    """A folder-name whitelist cannot admit a same-named plugin in another root."""
    wheel = assemble_wheel(SOURCE, tmp_path / "candidate")
    library = tmp_path / "lib"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(library)
    checkout = tmp_path / "comfy"
    (checkout / "comfy").mkdir(parents=True)
    (checkout / "comfy/__init__.py").write_text("")
    (checkout / "comfy/options.py").write_text("args_parsing = False\ndef enable_args_parsing():\n global args_parsing\n args_parsing = True\n")
    (checkout / "folder_paths.py").write_text("import comfy.options\nassert comfy.options.args_parsing\npaths={'custom_nodes':['untrusted'], 'models':['models']}\ndef get_folder_paths(folder):\n return paths[folder]\n")
    (checkout / "main.py").write_text("import folder_paths,sys\nfrom pathlib import Path\nfolder_paths.paths['custom_nodes'].append('another-untrusted')\nactual=folder_paths.get_folder_paths('custom_nodes')\nassert len(actual)==1\nassert (Path(actual[0])/'skulk_comfy_ltx/launcher.py').is_file()\nassert folder_paths.get_folder_paths('models')==['models']\nassert sys.argv[1:]==['--disable-all-custom-nodes','--whitelist-custom-nodes','skulk_comfy_ltx']\n")
    subprocess.run([sys.executable, "-I", "-B", str(library / "skulk_comfy_ltx/launcher.py"), str(checkout), "--disable-all-custom-nodes", "--whitelist-custom-nodes", "skulk_comfy_ltx"], check=True, capture_output=True)


def test_other_source_revision_cannot_promote_even_with_matching_descriptor(tmp_path: Path) -> None:
    """A successful attested build from another branch is not source approval."""
    source = tmp_path / "other-source"
    shutil.copytree(SOURCE, source, ignore=shutil.ignore_patterns("__pycache__"))
    nodes = source / "skulk_comfy_ltx/nodes.py"
    nodes.write_text(nodes.read_text() + "\n# Different source revision.\n")
    candidate = assemble_wheel(source, tmp_path / "candidate")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="reviewed checkout"):
        verify_artifact(candidate.parent, digest)
