# clips/

Drop your raw AI-generated clips here. Cappycat loads this folder automatically and
orders the clips by their **filenames**, so name them in story order.

Any of these naming styles works (mix-and-match is not recommended within one folder):

| style | example |
|---|---|
| leading number | `01_intro.mp4`, `02_kitchen.mp4`, `10_finale.mp4` |
| clip / shot / part markers | `clip1.mp4`, `clip2.mp4`, `clip10.mp4` · `Part II.mp4` |
| scene + shot | `scene1_shot1.mp4`, `scene1_shot2.mp4`, `scene2_shot1.mp4` |
| episode codes | `S01E01.mp4`, `S01E02.mp4` |
| words | `first.mp4`, `second.mp4` · `chapter one.mp4` · `1st.mp4`, `2nd.mp4` |
| trailing number | `raccoon_kitchen_3.mp4`, `raccoon_roof_12.mp4` |
| letter sub-shots | `shot 3a.mp4`, `shot 3b.mp4`, `shot 4.mp4` |

Numbers are compared as numbers (2 comes before 10). Files named only `intro`,
`opening` or `prologue` go first. Files named `outro`, `finale` or `credits` go last.
Resolutions (`1080p`, `4k`), versions (`v2`), frame rates (`24fps`) and random
generator IDs in the name are ignored.

The app shows why each clip got its position and warns about gaps
(for example, "numbering skips 4") or two clips claiming the same slot.
Check the order from a terminal with:

```bash
pipeline/.venv/Scripts/python -m cappycat_pipeline order clips
```

`.cube` LUTs placed here are imported too. Media files in this folder are not committed to git.
