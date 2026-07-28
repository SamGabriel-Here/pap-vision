# Dataset — Mendeley LBC

Everything here was read directly from the published dataset's file listing on
2026-07-29, not from the accompanying paper. Where the two disagree, this
document says so, because a couple of the widely-cited numbers are wrong.

| | |
|---|---|
| Name | Liquid based cytology pap smear images for multi-class diagnosis of cervical cancer |
| Contributor | Elima Hussain |
| DOI | [10.17632/zddtpgzv63.4](https://doi.org/10.17632/zddtpgzv63.4) |
| Version used | 4 (published 18 November 2019) |
| Licence | **CC BY 4.0** — redistribution of derived works, including trained weights, is permitted with attribution |
| Download size | 2.23 GB (962 JPEGs, 2048×1536) |
| Capture | Leica ICC50 HD microscope at 400× |

## What is actually in it

The four class folders are named descriptively, not by their Bethesda
abbreviation:

| Folder on Mendeley | Class |
|---|---|
| `High squamous intra-epithelial lesion` | HSIL |
| `Low squamous intra-epithelial lesion` | LSIL |
| `Negative for Intraepithelial malignancy` | NILM |
| `Squamous cell carcinoma` | SCC |

## Three corrections to the commonly-cited description

**1. The class distribution in the literature is wrong.** Papers citing this
dataset almost universally report 963 images split NILM 613 / LSIL 163 /
HSIL 113 / SCC 74. The actual published folders contain:

| Class | Images listed | Unique images | Commonly cited |
|---|---:|---:|---:|
| HSIL | 173 | **163** | 113 |
| LSIL | 113 | **113** | 163 |
| NILM | 613 (incl. `Results.csv`) | **612** | 613 |
| SCC | 74 | **74** | 74 |
| **Total** | 973 | **962** | 963 |

HSIL and LSIL appear to have been transposed somewhere and then copied
forward. The HSIL folder also lists ten filenames twice — `HSIL_2 (10).jpg`
through `HSIL_2 (20).jpg` — which accounts for 173 listed against 163 unique.
The NILM folder contains a stray `Results.csv` that is not an image.

**2. There are 61 slides, not 460 independent samples.** The paper describes
specimens collected from 460 patients, and that figure gets repeated as though
the images are 460 independent samples. They are not. The filenames carry a
slide identifier, and there are only 61 distinct ones:

| Class | Slides | Images | Images per slide (min–max) |
|---|---:|---:|---|
| HSIL | 10 | 163 | 5–28 |
| LSIL | **4** | 113 | 21–37 |
| NILM | 43 | 612 | 2–25 |
| SCC | **4** | 74 | 11–27 |
| **Total** | **61** | 962 | |

This is the single most important fact about the dataset, and it is why the
[splitting rules](splitting.py) matter so much. With roughly 16 images per
slide, a random per-image split puts images from the same slide on both sides
of the train/test boundary almost every time. The model then only has to
recognise a slide's staining and illumination signature, not its pathology.

**3. LSIL and SCC have four slides each.** A grouped 70/10/20 split therefore
gives one LSIL slide and one SCC slide in the test set. Per-class recall for
those two classes is measured on a single slide, and no amount of careful
methodology downstream fixes that. Any LSIL or SCC number from this dataset
should be read as an anecdote with a confidence interval spanning most of the
unit interval, not as a measurement.

## Filename conventions

Inconsistent, and worth handling explicitly:

```
HSIL_2 (10).jpg     slide HSIL_2,  image 10
LSIL_4 (37).jpg     slide LSIL_4,  image 37
NL_10_ (5).jpg      slide NL_10,   image 5     — trailing underscore
scc_1 (21).jpg      slide scc_1,   image 21    — lowercase
SCC_3 (11).jpg      slide SCC_3,   image 11    — uppercase, same folder
```

The slide id is everything before the parenthesised image number. Both the
trailing underscore and the inconsistent casing have to be normalised, or one
slide fragments into two groups and leaks across the split. `splitting.py`
lowercases and strips trailing underscores for exactly this reason.

## Fetching it

`inspect_dataset.py` recognises this naming scheme and will report the slide
structure for whatever layout you end up with. Mendeley's "Download All" link
is an S3 URL that requires signing, but each file exposes a stable public
download URL through the dataset API:

```
https://data.mendeley.com/public-api/datasets/zddtpgzv63/files?folder_id=<folder>&version=4
https://data.mendeley.com/public-files/datasets/zddtpgzv63/files/<file_id>/file_downloaded
```

## Attribution

> Hussain, Elima (2019), "Liquid based cytology pap smear images for
> multi-class diagnosis of cervical cancer", Mendeley Data, V4,
> doi: 10.17632/zddtpgzv63.4 — licensed CC BY 4.0.
