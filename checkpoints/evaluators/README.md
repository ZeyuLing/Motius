# Local Evaluator Snapshots

Evaluator checkpoints normally load directly from the Hugging Face artifact
linked in the [Evaluator Zoo](../../README.md#evaluator-zoo). For offline
use, place a complete snapshot under
`checkpoints/evaluators/<evaluator-name>/` and pass that directory to the
evaluator's `from_pretrained(...)` method.

Downloaded weights, tokenizer caches, and Hugging Face cache directories are
ignored. Keep lightweight evaluation protocols and metric code in `motius/`
or `docs/`; this directory is only for runtime checkpoint artifacts.
