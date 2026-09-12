# ModelDesk Hub and download queue — 1.0.2

Selecting a Hub search result loads its model details automatically. Requests use cancellation and generation checks so a slow previous result cannot replace a newer selection. Each model remembers its destination folder during the session; matching queue entries can restore a previous destination.

The local inventory scans filesystem metadata on a worker thread when a model opens, its destination changes, the Hub tab is revisited, or a matching download completes. A regular final file with the expected size is marked downloaded and its checkbox is hidden. Partial transfers, mismatched sizes, and inaccessible paths remain distinct. The inventory does not read model payloads or revalidate content hashes. External filesystem changes are picked up by **Quét lại** or revisiting the tab.

File filters support all, downloaded, and not downloaded. Shift selection follows the visible sorted/filtered order, supports deselection, and skips downloaded or unavailable files. The destination is rescanned before enqueueing to avoid adding files that have just completed.

**Xóa danh sách** stops active transfers, waits for their cleanup, saves an empty queue, and resets selection, errors, the status filter, and aggregate counters. It preserves completed files and partial-transfer data. New downloads can be enqueued afterward. Filename and model headers sort the queue; status filtering changes the visible rows while aggregate statistics still describe the entire queue.

The complete Release solution builds with zero warnings/errors, and **144/144 tests pass**, with no skips. Validation uses real temporary files for inventory and file-preservation checks, controlled asynchronous adapters for stale-result/cancellation/queue races, and a pumped WPF dispatcher for view-model behavior. Regression cases include cancelling a scan after its result completes and clearing during startup queue restoration. Headless rendering covers all seven tabs at two sizes and both themes (28 frames), using the actual XAML with fixture data. These checks do not interact with the user's running window or download model weights.

Local evidence: `reports/studio/hub-queue-v1.0.2-tests.log`, `reports/studio/tests/studio-tests.trx`, and `reports/studio/ui/headless-*.png`. All 62 Python/native source hashes remain unchanged. The user's running Debug application locks its output DLLs; Release builds use a separate output directory and do not interrupt that session.
