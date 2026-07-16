# Local Model Snapshots

Model Zoo pipelines normally resolve complete artifacts from Hugging Face:

```python
from motius.pipelines import MDMPipeline

pipe = MDMPipeline.from_pretrained("ZeyuLing/hftrainer-mdm-humanml3d")
```

For offline use, download a snapshot into
`checkpoints/models/<method>/<artifact>/` and pass that directory to the same
`from_pretrained` API. Follow the method's
[Model Card](../../README.md#model-zoo) for its exact repository and any
additional licensed dependency.

Downloaded weights and caches are intentionally ignored. A small,
redistributable model asset may be committed only after its license and purpose
are documented alongside it and its path is explicitly allowed by `.gitignore`.
