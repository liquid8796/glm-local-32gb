"""Release metadata and checked-in evidence links must describe the same release."""

from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlsplit

from glm_local import __version__

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: no third-party TOML parser is required.
    tomllib = None


ROOT = Path(__file__).resolve().parents[1]


def project_version(text):
    if tomllib is not None:
        return tomllib.loads(text)["project"]["version"]
    project = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", text)
    if project is None:
        raise ValueError("Missing [project] table")
    version = re.search(r'^version\s*=\s*"([^"]+)"\s*$', project.group(1), re.MULTILINE)
    if version is None:
        raise ValueError("Missing simple quoted project version")
    return version.group(1)


class ReleaseConsistencyTests(unittest.TestCase):
    def test_package_and_pyproject_release_match(self):
        self.assertEqual(project_version((ROOT / "pyproject.toml").read_text("utf-8")),
                         __version__)

    def test_current_document_versions_match_package(self):
        expected = {
            "README.md": f"**Trạng thái {__version__}:",
            "docs/CHECKPOINT-METADATA.md": f"# Checkpoint metadata audit — {__version__}",
            "docs/ARCHITECTURE-MAPPER.md": f"# GLM Architecture Mapper v{__version__}",
            "docs/CURRENT-MEMORY.md": f"Current release: v{__version__}.",
        }
        for relative, marker in expected.items():
            with self.subTest(document=relative):
                self.assertIn(marker, (ROOT / relative).read_text("utf-8"))

    def test_metadata_historical_verification_links_resolve(self):
        document = ROOT / "docs" / "CHECKPOINT-METADATA.md"
        links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text("utf-8"))
        checked = []
        for target in links:
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path.startswith("verification/"):
                continue
            artifact = (document.parent / unquote(url.path)).resolve()
            with self.subTest(link=target):
                self.assertTrue(artifact.is_file(), f"Missing checked-in evidence: {target}")
            checked.append(target)
        self.assertGreaterEqual(len(checked), 4, "Metadata evidence links were lost")


if __name__ == "__main__":
    unittest.main()
