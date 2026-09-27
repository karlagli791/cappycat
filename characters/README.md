# characters/

Reference images for the main cast, used by the AI pipeline to recognise each
character by name. `characters.json` lists every character with its detection
prompts, identifying features, and which reference images show them.

Every character marked `"unique": true` should appear at most once in a frame.
When the pipeline sees the same unique character twice in one frame, it reports a
duplicate-character artifact by name (for example "Bunny appears twice") and the
reframer crops the extra copy out.

| id | name | references |
|---|---|---|
| suzie | Suzie (squirrel) | `suzie_sheet.webp`, group lineup #6 |
| raccoon | Raccoon | `raccoon_sheet.webp`, group lineup #3 |
| turtle | Turtle | `turtle_sheet.webp`, group lineup #4 |
| felix | Felix (dog, black backwards cap, skateboard tee) | `felix_sheet.webp`, `fox_sheet.webp` (earlier design, same outfit), group lineup #5 |
| bunny | Bunny | `bunny_sheet.webp`, `bunny_ref.webp` (cardigan and dress outfit), group lineup #1 |
| otter | Otter | `otter_sheet.webp`, group lineup #2 |

Felix and Fox are the same character. `fox_sheet.webp` is an earlier design of Felix,
kept as an extra reference, and `fox` is listed as an alias in `characters.json`.

## Adding a character

1. Put one or more images of the character in this folder. Character sheets work well.
2. Add an entry to `characters.json` with an `id`, `name`, `prompts` and `references`.
   For a group picture, add the image under `lineups` with the characters listed left to right.
3. Run `pipeline/.venv/Scripts/python -m cappycat_pipeline characters build` to refresh the
   cached reference embeddings. The pipeline also rebuilds them automatically when an image changes.
   The chosen crops are written to `.cache/crops/` for review (`*_ref.jpg`: reference only, e.g. a
   back view; `*_dropped.jpg`: a near-duplicate of another crop, not used).

## Optional manifest fields

* **Manual crop boxes** - when the automatic crops miss a view (a back view, a pose), give the boxes
  in the image's pixels. Detection is then skipped for that image unless `"detect": true`:

  ```json
  { "image": "bunny_sheet.webp", "boxes": [[40, 290, 250, 760], [555, 300, 775, 760]], "detect": true }
  ```

* **`zeroShot`** (per character) - short descriptions that tell this character apart from look-alikes,
  e.g. `"zeroShot": ["a young cartoon turtle wearing a navy baseball cap backwards"]`. Default:
  `"a cartoon <species> character: <features>"`.
* **`negatives`** (top level) - figures that are *not* main cast, as text and / or image crops. A
  detection that looks more like a negative than like its best cast member gets no name:

  ```json
  "negatives": [
    { "text": "a cartoon deer" },
    { "image": "grandpa_turtle.webp", "boxes": [[0, 0, 400, 600]] }
  ]
  ```

  Every character's `notLike` text is used as a negative too.

Supporting characters who can legitimately appear more than once, or who aren't in
the cast (such as the elderly turtle), need no character entry. Anything not matched to a cast
member is treated as an unnamed extra; list look-alikes under `negatives` (the deer, the
elephant and the elderly turtle are listed) so they are never named as a cast member.
