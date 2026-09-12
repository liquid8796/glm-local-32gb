# Project workflow

- After each completed patch, run relevant checks, commit the changes, and push to `origin/master` without asking for confirmation again. Never force-push; fetch and preserve remote work before integration.
- During C# GUI/CLI work, keep the Python/native inference core unchanged unless the user explicitly requests core changes.
- Do not commit model weights, Python environments, temporary download payloads, build caches, or local run folders. Track model profiles and baseline metadata under `config/models/` and `docs/models/`.
