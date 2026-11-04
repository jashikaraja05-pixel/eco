# Dataset Attribution and Licence Record

EcoLens AI / ECO-LENSAI trains its optional local waste classifier on public
image datasets. **No dataset images are stored in this repository**, and no
trained checkpoint is committed. This file records, for each source, what it is,
where it comes from, which licence the source *declares*, what would have to be
credited if its images are redistributed, what preprocessing this project
applies, and which questions are still open.

The licences below are the **declared metadata published by each source**. They
are not a legal opinion and not a redistribution clearance. Nothing here may be
quoted as "cleared for use"; the open questions in the last section must be
answered by the dataset owner or a responsible person before any public
redistribution of images.

---

## 1. TrashNet (default training source)

| Field | Value |
|---|---|
| Name | TrashNet — "A Dataset for Trash Classification" (paper title: *Classification of Trash for Recyclability Status*) |
| Source | https://github.com/garythung/trashnet |
| Also distributed via | https://huggingface.co/datasets/garythung/trashnet (used by `--source hf`) |
| Declared licence | MIT — `LICENSE` file in the GitHub repository, stating *Copyright (c) 2017 Gary Thung* |
| Licence file / card to re-check | `LICENSE` at the repository root; the Hugging Face dataset card |
| Class folders | `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash` |
| Size as published | 2,527 images |
| Used for app category | `recyclable` (cardboard, glass, metal, paper, plastic), `residual` (`trash`) |
| Attribution to display | "TrashNet — Gary Thung & Mindy Yang, MIT License (c) 2017 Gary Thung" |

The MIT text is verified in the repository's own `LICENSE` file. Any copy of the
images must keep the copyright and permission notice that the MIT licence
requires.

## 2. Garbage Classification (12 classes) — "garbage12"

| Field | Value |
|---|---|
| Name | Garbage Classification (12 classes), Kaggle dataset by user `mostafaabla` |
| Source | https://www.kaggle.com/datasets/mostafaabla/garbage-classification |
| Licence file / card to re-check | The **Licence** block on the Kaggle dataset page (and any `LICENSE`/`README` inside the downloaded archive) |
| Declared licence | **ODbL 1.0 — Database: Open Database; Contents: © Original Authors** |
| Class folders | `battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass` |
| Size as published | 15,150 images |
| Used for app category | `hazardous` (`battery`), `organic` (`biological`), `recyclable` (the glass/cardboard/metal/paper/plastic folders), `residual` (`clothes`, `shoes`, `trash`) |
| Attribution to display | "Garbage Classification (12 classes) — mostafaabla, Kaggle. Database: ODbL 1.0; Contents: © Original Authors" |

Important corrections recorded here:

- **This dataset contains no e-waste class.** Its 12 folders are the ones listed
  above. Earlier documentation in this repository implied that garbage12
  supplied e-waste; that claim was wrong and has been removed. The e-waste
  category comes only from source 3.
- Under ODbL the *database* is open but the *contents* (the individual images)
  remain © their original authors, and the dataset page does not restate who
  those authors are. Any redistribution therefore needs the attribution wording
  the ODbL requires plus a check of the individual image rights — see the open
  questions.
- The archive is not committed; it is downloaded by the developer via the Kaggle
  CLI (`--kaggle`) after accepting Kaggle's terms in their own account.

## 3. GIZ e-waste dataset with YOLO labels

| Field | Value |
|---|---|
| Name | GIZ e-waste dataset (YOLO-format bounding-box labels) |
| Source | https://huggingface.co/datasets/GIZ/e-waste-dataset-yolo-labels |
| Licence file / card to re-check | The **Licence** metadata field and the dataset card text on Hugging Face, plus any `LICENSE`/`README` inside the archive |
| Declared licence metadata | **CC BY 4.0** (as declared in the dataset card metadata) |
| Label format | object detection: one `.txt` file of normalised bounding boxes per image, with many object classes (e.g. ACs, compressors, computers, fridges) |
| Used for app category | `ewaste` — every accepted image is copied once into a single `e-waste/` folder |
| Attribution to display | "GIZ e-waste dataset (YOLO labels) — GIZ, https://huggingface.co/datasets/GIZ/e-waste-dataset-yolo-labels, CC BY 4.0 (declared)" |

**Unconfirmed items (do not treat as cleared):**

- The **exact attribution string** CC BY 4.0 requires (creator name, title, and
  any copyright notice the card specifies) is not yet recorded here, because the
  dataset card does not state an unambiguous author/copyright line that this
  repository can copy.
- CC BY 4.0 covers the *dataset*, but the individual photographs inside it may
  originate from third parties with their own rights. Whether GIZ holds the
  rights needed to license them under CC BY 4.0 has **not** been confirmed.
- Whether derivative use in a trained model, and redistribution of images, is
  permitted in the intended deployment is therefore an **open question**.

---

## Modifications applied by this project

Applied by `backend/prepare_dataset.py` at preparation time; original files are
left untouched.

1. **Class-folder selection.** Only class folders that contain usable images
   (extension `.jpg`/`.jpeg`/`.png`/`.webp`, non-empty file) become output
   classes. A folder named in a mapping but missing from the dataset is dropped
   and reported, never turned into an untrained output neuron.
2. **Folder-name normalisation and aliases.** Folder names are matched
   case- and separator-insensitively, and a small declared alias table maps
   obvious synonyms; unknown folders are reported, never guessed.
3. **Optional per-class cap.** `--max-per-class` limits how many images a single
   class contributes, to reduce class imbalance (a seeded subset is selected;
   the full dataset is unchanged on disk).
4. **Manifest split.** Images are assigned once to `train`/`val`/`test`
   (70 % / 15 % / remainder) with a fixed seed, so an image appears in exactly
   one split and the split is reproducible.
5. **Image-level relabelling for the box-labelled GIZ source.** Because the
   classifier predicts one label per photo, each accepted GIZ image is copied
   once into `e-waste/`; the bounding boxes are **not** cropped and the `.txt`
   label files are ignored. Images with unclear content are excluded rather than
   guessed. This is an **aggregation, not a re-annotation**: no new bounding
   boxes are produced, and no licence is changed by the copy.
6. **No visual modification** of image pixels is performed by the preparation
   step other than the JPEG re-encode done when exporting TrashNet from Hugging
   Face (quality 90, RGB).
7. **Licence recording.** `split.json` contains a `licences` block naming only
   the sources whose folders actually contributed an output class, so a run can
   be traced back to its inputs. Sources that were not present are not claimed.

## Duplicate handling

Duplicate images across the merged sources are detected by SHA-256 of the file
contents before merging (see the acquisition checklist), and only the first copy
is kept. This project checks for *file-level* duplicates; near-duplicate or
re-encoded copies of the same photo are **not** currently detected.

## Trading / redistribution notes

- Nothing here permits redistribution of the images. Do not commit dataset
  archives, extracted images or raw Kaggle/Hugging Face downloads to this
  repository. `backend/data/` and model weights stay untracked.
- A trained model's weights are a derived artefact. Whether ODbL (contents
  clause) and CC BY 4.0 attach obligations to the weights or to their
  distribution is **not resolved here**; treat weight distribution as subject to
  the same open questions.

## Open questions — required before any redistribution

1. **GIZ**: who exactly must be credited under CC BY 4.0 (creator name, title,
   copyright line, link, and any modification notice), and does GIZ hold
   sufficient rights in the third-party photographs to license them this way?
2. **GIZ**: is the dataset card's `CC BY 4.0` the intended licence for the
   *images*, or only for the annotations/derived labels?
3. **garbage12**: the dataset page's Licence block must be re-read and archived
   verbatim (ODbL 1.0, "Contents: © Original Authors") — were the individual
   images licensed by the original authors for redistribution, or only
   aggregated?
4. **garbage12**: what attribution string satisfies the ODbL "Contents"
   requirement for the images, given that the dataset page does not name the
   original authors individually?
5. **TrashNet**: is the MIT `LICENSE` in the repository still the applicable
   licence for the Hugging Face mirror used by `--source hf`?
6. Whether a distributed model checkpoint and/or a public demo that displays
   predictions counts as a derivative database under ODbL.

Until these are answered, this repository claims **no legal clearance and no
redistribution rights** for any of the above datasets.

## Dataset acquisition log (to be completed when a dataset is actually obtained)

This template is intentionally unfilled — no dataset has been downloaded as
part of this change, so any value here would be invented.

| Dataset | Source URL | Declared licence (verbatim) | Licence file path in archive | Download date | Archive SHA-256 | Image count | Duplicates removed | Attribution recorded |
|---|---|---|---|---|---|---|---|---|
| garbage12 | | | | | | | | |
| giz_ewaste | | | | | | | | |
| trashnet | | | | | | | | |

## Code references

- `backend/prepare_dataset.py` — mapping tables, aliases, licence metadata
  (`DATASET_LICENCES`, `dataset_licences()`), split manifest writer.
- `backend/manifest.py` — dataset/model contract validation.
- `backend/README.md` — dataset table and documented limitations.
