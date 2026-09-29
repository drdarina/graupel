import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest


LICENSE_EXPRESSION = "GPL-3.0-or-later AND MPL-2.0"


def test_license_files_exist_in_root():
    root = Path(__file__).resolve().parent.parent
    license_file = root / "LICENSE"
    mpl_file = root / "LICENSES" / "MPL-2.0.txt"
    frontend_license = root / "react" / "LICENSE.md"
    built_frontend_notice = root / "react" / "public" / "FRONTEND_LICENSE.txt"
    third_party_file = root / "THIRD_PARTY_LICENSES.md"

    assert license_file.is_file(), "LICENSE file must exist in repository root"
    assert mpl_file.is_file(), "The complete MPL-2.0 text must exist"
    assert frontend_license.is_file(), "React sources need an adjacent notice"
    assert built_frontend_notice.is_file(), "Built frontend needs a notice"
    assert third_party_file.is_file(), "THIRD_PARTY_LICENSES.md must exist in repository root"

    content = license_file.read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in content
    assert "GPL-3.0-or-later" in content or "either version 3 of the License, or" in content
    assert "react/src/" in content
    assert "MPL-2.0" in content

    mpl_content = mpl_file.read_text(encoding="utf-8")
    assert "Mozilla Public License Version 2.0" in mpl_content
    assert "Exhibit A - Source Code Form License Notice" in mpl_content

    frontend_content = frontend_license.read_text(encoding="utf-8")
    assert "MPL-2.0" in frontend_content
    assert "Icons and image assets" in frontend_content

    built_notice_content = built_frontend_notice.read_text(encoding="utf-8")
    assert "Source Code Form is available" in built_notice_content
    assert "GPL-3.0-or-later" in built_notice_content
    assert built_notice_content == (
        root / "graupel" / "react" / "FRONTEND_LICENSE.txt"
    ).read_text(encoding="utf-8")

    tp_content = third_party_file.read_text(encoding="utf-8")
    assert "Third-Party Licenses" in tp_content
    assert "react" in tp_content
    assert "maplibre-gl" in tp_content
    assert "lucide-react" in tp_content


def test_license_files_included_in_sdist_and_wheel(tmp_path):
    root = Path(__file__).resolve().parent.parent
    if not (root / "graupel" / "react" / "index.html").is_file():
        pytest.skip("frontend not built; run `make react` first")

    # Build from a copy so a failed build can never leave staging
    # directories (e.g. graupel-<version>/) in the working tree.
    src = tmp_path / "src"
    shutil.copytree(
        root,
        src,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "dist",
            "build",
            "*.egg-info",
            "__pycache__",
            ".pytest_cache",
            "log",
            "*.db",
        ),
    )
    dist_dir = tmp_path / "dist"

    uv_bin = shutil.which("uv") or "uv"
    cmd = [
        uv_bin,
        "build",
        "--out-dir",
        str(dist_dir),
    ]
    res = subprocess.run(cmd, cwd=src, capture_output=True, text=True)
    assert res.returncode == 0, f"uv build failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    sdists = list(dist_dir.glob("*.tar.gz"))
    wheels = list(dist_dir.glob("*.whl"))

    assert len(sdists) >= 1, "At least one sdist (.tar.gz) must be built"
    assert len(wheels) >= 1, "At least one wheel (.whl) must be built"

    # Verify sdist
    with tarfile.open(sdists[0], "r:gz") as tar:
        names = tar.getnames()
        expected_suffixes = (
            "/LICENSE",
            "/LICENSES/MPL-2.0.txt",
            "/THIRD_PARTY_LICENSES.md",
            "/react/LICENSE.md",
            "/react/public/FRONTEND_LICENSE.txt",
            "/react/src/App.tsx",
            "/graupel/react/FRONTEND_LICENSE.txt",
        )
        for suffix in expected_suffixes:
            assert any(n.endswith(suffix) for n in names), (
                f"{suffix} not found in sdist"
            )

    # Verify wheel
    with zipfile.ZipFile(wheels[0], "r") as zf:
        wheel_names = zf.namelist()
        expected_suffixes = (
            ".dist-info/licenses/LICENSE",
            ".dist-info/licenses/LICENSES/MPL-2.0.txt",
            ".dist-info/licenses/THIRD_PARTY_LICENSES.md",
            "graupel/react/FRONTEND_LICENSE.txt",
        )
        for suffix in expected_suffixes:
            assert any(n.endswith(suffix) for n in wheel_names), (
                f"{suffix} not found in wheel"
            )

        metadata_name = next(
            n for n in wheel_names if n.endswith(".dist-info/METADATA")
        )
        metadata = zf.read(metadata_name).decode("utf-8")
        assert f"License-Expression: {LICENSE_EXPRESSION}" in metadata
        assert "License-File: LICENSE" in metadata
        assert "License-File: LICENSES/MPL-2.0.txt" in metadata
        assert "License-File: THIRD_PARTY_LICENSES.md" in metadata
