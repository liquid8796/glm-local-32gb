# ModelDesk startup fix — 1.0.1

The startup log stopped after `Services composed; initializing workspace`. The main window had not been created yet.

`ShellViewModel.ApplyAsync` called the real synchronous `PythonCoreService.GetProfiles` on the WPF dispatcher. That method waited on `LocalFiles.ReadBoundedAsync().GetAwaiter().GetResult()`. An incomplete file read captured the same dispatcher for its continuation, which the synchronous wait had blocked. `ReportService.List` contained the same unsafe pattern, although the normal reports view already invokes it on a worker thread.

Synchronous services now use a genuinely synchronous bounded reader. Async file operations avoid capturing the caller's synchronization context. Profile discovery runs off the UI thread, and the application shows its window before awaiting workspace initialization. A loading status remains visible until initialization finishes.

Three new regressions use real files and real profile/report/settings services on a pumped STA dispatcher. Before the fix all three timed out at five seconds. After the fix all three passed; the complete suite passed **102/102 tests**, with no skips. Both Debug and Release built with zero warnings/errors. Existing headless WPF rendering tests also passed.

The former GUI tests used a fake profile provider, which did not exercise the real blocking file read. These regressions cover that missing boundary. They do not claim interactive Windows 10 testing or visible-window automation.

Local logs: `reports/studio/startup-regression-results/startup-before-fix.trx`, `startup-after-fix.trx`, and `reports/studio/startup-fix-build.log`. All 62 Python/native source hashes remained unchanged.
