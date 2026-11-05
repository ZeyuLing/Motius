# FlowHMR, integrated into Motius for Blender

[Watch the 1080p product film](motius-flowhmr-product-release-1080p.mp4) · [图文快速开始](../../../../docs/tutorials/flowhmr-blender.md) · [中文字幕](motius-flowhmr-product-release.zh.srt) · [Recording and frame audit](../../../../docs/architecture/validation/flowhmr-product-release-20261027.json)

![One performance, three Blender characters](motius-flowhmr-product-release-poster.png)

A 2-minute 14-second product film, recorded in actual Blender 4.5.3. It opens with the complete supplied 25-second performance: the source and three grounded characters play together at 30 fps, showing all 750 frames. Then it follows video selection, person review, real GPU inference, original reconstruction inspection, character import, reversible grounding and FBX export.

The selected-person review stays on screen for about 18 seconds. The raw reconstruction plays for 12 seconds. Processing waits are shortened by cuts. The current film adds only operation labels: slogans, timing explanations, performance cards and closing links have been removed. Measurements and upstream references remain in this documentation.

## What was actually run

The detector, full-frame VGGT camera estimation, per-frame body and both hand features, and motion generation execute on the supplied video. No previous neural frontend results are reused. The source input SHA matches the supplied attachment. Result and detail pickups use the same generated capture and actions, with revised viewport framing and lighting.

Both visible source and viewport frame counters were read from the **final encoded film**. The opening contains counters 1–750 in order, matching in every frame. All three characters' deformed bounds were checked across all 750 frames against the actual release viewport: no character crosses the viewport boundary. The supplemental uninterrupted operation recording and original outputs remain in `outputs/validation/flowhmr/product-release-20261027/`.

Playback uses the plugin's explicitly labelled viewport/source display cache. This preserves original-speed presentation and does not modify editable motion or export data. Cache preparation is a separate waiting stage.

## Performance boundaries

This recording measures **96.43 seconds of inference after person selection**, including model loading, on H200. Driver-sampled peak backend-process memory is **6432 MiB / 6.28 GiB**. Detection, surface conversion/import, character retargeting/grounding, export and display-cache preparation are separate. This is not a whole-application 96-second claim, a consumer-GPU measurement, or an under-60-second result.

The recording uses a Mesa software viewport. The shadowed release-result display cache takes about 367 seconds to prepare in that environment; this cost is retained in the raw recording and disclosed separately. It must not be confused with neural GPU inference or advertised as a consumer graphics benchmark. See the [guide](../../../../docs/tutorials/flowhmr-blender.md) and [complete provenance](../../../../docs/architecture/validation/flowhmr-product-release-20261027.json).

Earlier benchmarks and promotional cuts are archived in [HISTORY.md](HISTORY.md). Their timing scopes and frontend reuse differ; they are not substituted for this recording.

## Credits

Integrated method: [FlowHMR](https://github.com/flowhmr/flowhmr). Blender workflow reference: [Motion2MixamoRig](https://github.com/MaxLiu22/Motion2MixamoRig). Integration does not imply upstream endorsement or change the original licenses.

The opening audio comes from the supplied video. The quiet ambient bed is an original synthesized composition. Source footage, character and model usage remain subject to their respective permissions and licenses.
