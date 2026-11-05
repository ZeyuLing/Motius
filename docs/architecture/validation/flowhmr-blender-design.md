# FlowHMR Blender workflow design

The requested taste skill is written for public websites and explicitly excludes app screens. Apply its composition, real imagery, hierarchy and screenshot critique to the native Blender UI; do not invent a website or claim a browser inspection of Blender. No direction draw or website motion quota applies.

Subject inventory: the supplied dance video, its 750 actual frames, native 52-joint body and hand motion, selectable person detections, the source frame and its corresponding pose, reversible grounded actions, editable Mixamo characters, physically meaningful floor contacts, standard and accelerated inference, external backend with a responsive Blender front end.

Idea: the same performance becomes the visual thread through selection, capture and a cast of characters. The live interaction is choosing an actual detected person and reviewing that trajectory.

Composition: the source occupies a dedicated preview; the stage is dominant; the sidebar contains four plain-language groups. Charcoal ground, light text, mint selection accent. Native Blender controls retain native conventions. Detector outlines and the timeline communicate data, not decoration. Settings are progressively disclosed. No fake progress or simulated inference.

Waiting: verified preprocessing reuse retains exact cached ROIs and validates selected identity; true sampling evaluations and surface-frame counters are visible. Character import remains available while the external worker runs. CPU runtimes are not labelled GPU speedups.

Playback: Play every frame starts at the range beginning, uses sync NONE and a native modal operator that waits for draw acknowledgments from both the character viewport and source clip before advancing, then stops at the range end. Evaluated and actually drawn frame sequences are retained separately for verification. Slow devices slow down instead of dropping animation frames.

Selection: click any interior part of an observed box, with smallest box first and Tab/repeated click for overlaps. Review sampled frames using arrows or wheel. No nearest-person guess for background clicks. Video SHA binds selection to source. Missing endpoints and long detection gaps prevent inference; short gaps interpolate only within one track. ByteTrack is not an appearance-reidentification guarantee through arbitrary occlusion; review the chosen trajectory, and trim or re-detect ambiguous segments.

Rendered critique and correction: actual screen captures revealed overlapping raw/cast geometry, overly loose cast framing, and expensive software viewport shadows. The final UI isolates raw reconstruction and restores previous character visibility, frames the cast more tightly, and uses a lightweight scene floor grid for the software-recorded demo. The source image and selected bbox remain visible. Both viewport draw sequences are audited; software rendering speed is reported honestly, rather than hiding waiting time or speeding up the recording. Final artifacts: `outputs/validation/flowhmr/blender-workflow-v2/person-picker.png`, `native-output.png`, `three-character-ui.png`.
