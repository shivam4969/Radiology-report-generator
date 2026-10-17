# samples/

Images in this folder show up under **Try the sample image** in the UI, and
`npm run bench` runs every one of them through the configured engine.

Committed: `cxr-pneumonia-cc0.jpg` (lobar pneumonia, Mikael Häggström, CC0, see `../CREDITS.md`).

Everything else you put here is gitignored (`samples/*.png`, `samples/*.jpg` other than the CC0 file,
`samples/labels.csv`). Public research datasets have their own terms; do not redistribute them
through this repo, and never put identifiable patient images here.

## A 30 to 50 image labelled set for the demo and the bench

`labels.csv` sits next to the images with two columns:

```csv
filename,label
00000013_005.png,Pneumonia
00000032_001.png,No Finding
rsna-0004cfab-14fd-4e49-80ba-63a80b6bddd6.png,Pneumonia
```

Labels are free text; `scripts/bench.mjs` lower-cases them and accepts a few synonyms
(Pneumonia / Consolidation / Lung Opacity / Infiltration count as agreement with each other,
Mass / Nodule / Lung Lesion likewise, "No Finding" or "Normal" agree when nothing reaches the
threshold). NIH multi-labels such as `Effusion|Infiltration` are read as their first label.

### Option A: the script (Kaggle CLI configured)

```bash
pip install kaggle                       # token from kaggle.com/settings -> ~/.kaggle/kaggle.json
python scripts/fetch_samples.py --n 40   # NIH ChestX-ray14 "sample" dataset, balanced across 10 labels
python scripts/fetch_samples.py --rsna   # add the RSNA pneumonia challenge subset (needs: pip install pydicom)
```

The script downloads into `data/downloads/` (gitignored), picks single-label images spread across
No Finding, Pneumonia, Effusion, Cardiomegaly, Pneumothorax, Consolidation, Nodule, Mass, Atelectasis
and Infiltration, copies them here and writes or extends `labels.csv`.

### Option B: by hand

1. **NIH ChestX-ray14 sample** (5,606 PNGs with `sample_labels.csv`):
   Kaggle <https://www.kaggle.com/datasets/nih-chest-xrays/sample>, or the full set from the
   NIH Box link <https://nihcc.app.box.com/v/ChestXray-NIHCC> (one `images_0xx.tar.gz` and
   `Data_Entry_2017.csv` is enough). Unzip somewhere, then:
   `python scripts/fetch_samples.py --from-dir /path/to/unzipped --n 40`
2. **RSNA Pneumonia Detection Challenge** (DICOM; accept the competition rules first):
   <https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data>, then:
   `python scripts/fetch_samples.py --rsna --from-dir /path/to/unzipped --n 20`
   DICOMs are converted to 8-bit PNG under `samples/_rsna_png/` and copied here.
3. Or copy any PNG/JPG chest films here and write `labels.csv` yourself.

### Then

```bash
ENGINE=local npm run bench
```

prints, after the reads, a per-label table of how often the classifier's top finding agreed with
the label. Treat it as a sanity check of the pipeline, not a validation: the labels are NLP-mined
and noisy, the classifier is a research model, and 40 images is not a study.
