# ATLAS Model workflow

- This is the public mathematics and model package. Preserve its standalone Python 3.11+ runtime.
- Keep model behavior, time ordering, numerical checks and paper-only safeguards intact.
- Run npm ci --ignore-scripts and python -B -S -m atlas.checks before publishing changes.
- Resolve all type errors and warnings without suppressing diagnostics.
- Use synthetic fixtures for public examples and tests.
- Do not add private application data, portfolios, credentials, research snapshots, dashboards or provider collectors.
- Keep exports traceable through EXPORT_MANIFEST.json. Public edits need review before an upstream re-export.
