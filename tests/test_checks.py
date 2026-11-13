from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from atlas.checks import fingerprint, typecheck, validate_sources


class QualityChecksTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_syntax_duplicate_json_and_html_ids_are_reported(self):
        files = [self.write("broken.py", "def invalid(\n"), self.write("duplicate.json", '{"key":1,"key":2}'),
                 self.write("duplicate.html", '<div id="value"></div><p id="value"></p>')]
        issues = validate_sources(files, self.root)
        self.assertEqual(len(issues), 3)
        self.assertTrue(any("duplicate JSON" in issue for issue in issues))
        self.assertTrue(any("duplicate HTML" in issue for issue in issues))

    def test_python_minimum_version_and_source_changes_are_checked(self):
        source = self.write("future.py", "type NewAlias = int\n")
        self.assertEqual(len(validate_sources([source], self.root)), 1)
        before = fingerprint([source], self.root)
        source.write_text("value = 1\n")
        self.assertNotEqual(before, fingerprint([source], self.root))
        self.assertEqual(validate_sources([source], self.root), [])

    def test_checker_is_required_and_warnings_fail_the_gate(self):
        with self.assertRaisesRegex(ValueError, "Pyright is missing"):
            typecheck(self.root, "node")
        self.write("package.json", json.dumps({"devDependencies": {"pyright": "1.1.414"}}))
        self.write("node_modules/pyright/index.js", "")
        self.write("node_modules/pyright/package.json", '{"version":"1.1.414"}')
        result = SimpleNamespace(returncode=0, stdout=json.dumps({"summary": {"errorCount": 0, "warningCount": 1}}))
        with patch("atlas.checks.subprocess.run", return_value=result):
            self.assertFalse(typecheck(self.root, "node")["passed"])


    def test_nonstandard_numeric_constants_fail_source_validation(self):
        files = [self.write(str(index) + ".json", '{"value":' + value + '}')
                 for index, value in enumerate(("NaN", "Infinity", "-Infinity"))]
        issues = validate_sources(files, self.root)
        self.assertEqual(len(issues), 3)
        self.assertTrue(all("nonfinite JSON" in issue for issue in issues))


    def test_reencoded_windows_text_is_blocked_and_nepali_unicode_is_preserved(self):
        correct = "range 1\u201380; \u0928\u0947\u092a\u093e\u0932\u0940"
        broken = "range 1\u201380".encode("utf-8").decode("cp1252")
        files = [self.write("source.py", "message = " + repr(broken)), self.write("plan.md", broken),
                 self.write("names.json", json.dumps({"name": correct}, ensure_ascii=False))]
        issues = validate_sources(files, self.root)
        self.assertEqual(len(issues), 2)
        self.assertTrue(all("encoding error" in issue for issue in issues))
        self.assertEqual(json.loads(files[-1].read_text(encoding="utf-8"))["name"], correct)
