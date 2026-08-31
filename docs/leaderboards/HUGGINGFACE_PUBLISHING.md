# Publishing the static leaderboard Spaces

Motius keeps each Hugging Face Space's static interface under
`docs/leaderboards/hf_space_*`. The publisher derives every local source and
remote Space pair from `docs/leaderboards/catalog.json` and
`docs/tasks/taxonomy.json`; there is no second, hand-maintained repo-ID list.

## Preview a publish plan

The command is a local dry run by default. With no selection flags, it checks
changes since `HEAD`, including eligible untracked files, across all catalogued
Spaces:

```powershell
python tools/publish_hf_spaces.py
```

Choose a different comparison commit or limit the plan to one or more Spaces:

```powershell
python tools/publish_hf_spaces.py --since origin/main
python tools/publish_hf_spaces.py --space sequential_text_to_motion_babel
python tools/publish_hf_spaces.py --space ZeyuLing/motion-edit-leaderboard
```

`--space` accepts a benchmark ID, full Space repo ID, catalog source path, or
the `hf_space_*` source-directory name. It is repeatable. To deliberately
re-upload every eligible current interface file, use `--all` (optionally with
`--space`):

```powershell
python tools/publish_hf_spaces.py --all --space hf_space_motion_repair
```

## Apply the reviewed plan

Authenticate through a standard Hugging Face mechanism such as `hf auth login`
or the `HF_TOKEN` environment variable. Never put a token in the command line.
After reviewing the dry-run output, repeat the command with `--apply`:

```powershell
python tools/publish_hf_spaces.py --since origin/main --apply
```

`--apply` publishes one Space at a time. Each commit is pinned to the remote
SHA observed immediately before that commit, so a concurrent remote update
causes a conflict instead of being silently overwritten. Successful output
includes the before SHA, after SHA, and immutable Hugging Face commit URL.

## Safety boundary

The publisher only uploads current, regular, UTF-8 text files with one of these
interfaces: `.html`, `.css`, `.js`, or the exact name `README.md`. JSON
payloads, media, binary files, ignored files, symlinks, and every other path are
excluded. Deleted and renamed-away paths are preserved remotely: the publisher
never creates a delete operation and does not replace the whole Space tree.

The file digest recorded during planning is checked again when each upload
operation is constructed. If a file changes between review and apply, the run
stops and asks for a fresh plan. A failure also stops the remaining Spaces so a
partial batch is visible and auditable.
