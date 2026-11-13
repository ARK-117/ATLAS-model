"""Reproduce editor diagnostics and run the project's offline quality checks."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .json_inputs import reject_constant

ROOT = Path(__file__).resolve().parents[1]


def source_files(root: Path) -> list[Path]:
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=root, capture_output=True, check=True)
    return sorted({root / name.decode("utf-8") for name in result.stdout.split(b"\0")
                   if name and (root / name.decode("utf-8")).is_file()})


def fingerprint(files: list[Path], root: Path) -> str:
    return sha256(json.dumps({str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in files},
                             sort_keys=True).encode()).hexdigest()


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


class _HTMLIds(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name == "id" and value:
                if value in self.ids:
                    raise ValueError(f"duplicate HTML id: {value}")
                self.ids.add(value)


def validate_sources(files: list[Path], root: Path) -> list[str]:
    issues = []
    for path in files:
        try:
            content = None
            if path.suffix in {".py", ".json", ".html", ".js", ".md", ".csv", ".txt", ".yml", ".yaml", ".toml"}:
                content = path.read_text(encoding="utf-8-sig")
                if any(marker in content for marker in ("\u00e2\u20ac", "\u00c3\u0192", "\u00c3\u00a2")):
                    raise ValueError("text contains a UTF-8/Windows-1252 encoding error; restore the original Unicode text")
            if path.suffix == ".py":
                assert content is not None
                ast.parse(content, filename=str(path), feature_version=(3, 11))
            elif path.suffix == ".json":
                assert content is not None
                json.loads(content, object_pairs_hook=_unique_json, parse_constant=reject_constant)
            elif path.suffix == ".html":
                parser = _HTMLIds()
                assert content is not None
                parser.feed(content)
                parser.close()
        except (OSError, ValueError, SyntaxError) as error:
            issues.append(f"{path.relative_to(root)}: {error}")
    return issues


def typecheck(root: Path, node: str) -> dict:
    installed = root / "node_modules" / "pyright"
    if not (installed / "index.js").is_file():
        raise ValueError("Pyright is missing. Run npm ci --ignore-scripts first.")
    expected = json.loads((root / "package.json").read_text())["devDependencies"]["pyright"]
    actual = json.loads((installed / "package.json").read_text())["version"]
    if actual != expected:
        raise ValueError("Installed Pyright differs from the pinned version. Run npm ci --ignore-scripts.")
    result = subprocess.run([node, str(installed / "index.js"), "--outputjson"], cwd=root,
                            capture_output=True, text=True, encoding="utf-8")
    try:
        report = json.loads(result.stdout)
    except ValueError as error:
        raise ValueError("Pyright did not return a diagnostic report: " + result.stderr.strip()) from error
    summary = report["summary"]
    report["passed"] = result.returncode == 0 and summary["errorCount"] == 0 and summary["warningCount"] == 0
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Check all active source files, editor diagnostics and tests before a milestone or push.")
    parser.add_argument("--report", type=Path, default=ROOT / "audit" / "code-quality" / "latest.json")
    args = parser.parse_args()
    report: dict[str, Any] = {"checked_at": datetime.now(timezone.utc).isoformat(), "passed": False}
    try:
        files = source_files(ROOT)
        before = fingerprint(files, ROOT)
        issues = validate_sources(files, ROOT)
        node = shutil.which("node")
        if node is None:
            raise ValueError("Node.js is required for development checks; the ATLAS runtime remains standard-library Python.")
        for path in files:
            if path.suffix in {".js", ".mjs", ".cjs"}:
                checked = subprocess.run([node, "--check", str(path)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
                if checked.returncode:
                    issues.append(f"{path.relative_to(ROOT)}: {checked.stderr.strip()}")
        report["source_issues"] = issues
        print(f"Source checks: {len(issues)} issue(s)", flush=True)
        for issue in issues[:20]:
            print(issue, flush=True)
        diagnostics = typecheck(ROOT, node)
        report["pyright"] = diagnostics
        summary = diagnostics["summary"]
        print(f"Pyright: {summary['filesAnalyzed']} files, {summary['errorCount']} errors, {summary['warningCount']} warnings", flush=True)
        for item in diagnostics.get("generalDiagnostics", [])[:20]:
            print(f"{item['file']}:{item['range']['start']['line'] + 1}: {item['message']}", flush=True)
        tests = subprocess.run([sys.executable, "-B", "-S", "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
        report["tests"] = {"returncode": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr}
        print((tests.stdout + tests.stderr).strip(), flush=True)
        browser_tests = sorted((ROOT / "tests").glob("*.test.cjs"))
        javascript_passed = True
        if browser_tests:
            checked_js = subprocess.run([node, "--test", "--test-reporter=tap", *map(str, browser_tests)], cwd=ROOT,
                                        capture_output=True, text=True, encoding="utf-8", errors="replace")
            javascript_passed = checked_js.returncode == 0
            report["javascript_tests"] = {"returncode": checked_js.returncode, "stdout": checked_js.stdout, "stderr": checked_js.stderr}
            counts = [line[2:] for line in checked_js.stdout.splitlines()
                      if line.startswith(("# tests ", "# pass ", "# fail ", "# skipped "))]
            print("JavaScript: " + ", ".join(counts), flush=True)
            if not javascript_passed:
                print((checked_js.stdout + checked_js.stderr).strip(), flush=True)
        report["sources_unchanged"] = before == fingerprint(source_files(ROOT), ROOT)
        if not report["sources_unchanged"]:
            print("Sources changed while checks were running. Run the checks again before pushing.", flush=True)
        report["source_fingerprint"] = before
        report["passed"] = not issues and diagnostics["passed"] and tests.returncode == 0 and javascript_passed and report["sources_unchanged"]
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        report["check_error"] = str(error)
        print(f"Checks failed: {error}", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Quality report: {args.report}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
