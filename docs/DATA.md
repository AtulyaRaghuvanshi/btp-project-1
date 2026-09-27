# Dataset acquisition

Everything goes under `data/raw/<name>/`. `scripts/01_build_manifest.py` scans
whatever is present and skips what is not, so you can start with one dataset and
add more later without rebuilding anything.

Target layout:

```
data/raw/
├── artifact/          <- primary, 32 GB
├── forensynths/       <- GAN arm
├── genimage/          <- diffusion arm
├── cocoglide/         <- localization arm (has masks)
└── selftest/          <- synthetic, auto-generated
```

---

## 1. ArtiFact — primary source *(open, no form)*

2.5 M images from 25 generators (13 GAN, 7 diffusion, 5 other) plus ~965 k real
photographs, organised one folder per generator with a `metadata.csv` carrying
the 0/1 target. That per-generator structure is what makes the cross-generator
transfer matrix cheap to build.

```bash
pip install kaggle
# put your kaggle.json in %USERPROFILE%\.kaggle\
kaggle datasets download -d awsaf49/artifact-dataset -p data/raw/artifact --unzip
```

**32 GB.** If disk or bandwidth is tight, pull a subset of generator folders —
the manifest builder does not care how many are present. A workable minimum is
one real source (`ffhq` or `coco`), two GANs (`progan`, `stylegan`) and two
diffusion models (`stable_diffusion`, `glide`).

## 2. ForenSynths / CNNDetection — GAN arm *(open)*

The standard GAN-generalisation test set: ProGAN, StyleGAN, BigGAN, CycleGAN,
StarGAN, GauGAN, Deepfake.

```bash
git clone https://github.com/PeterWang512/CNNDetection
cd CNNDetection/dataset/test && bash download_testset.sh
# then move the extracted folders into data/raw/forensynths/
```

Mirror if the script's link is down: Kaggle `chandlertimm/forensynths`.

Expected layout: `data/raw/forensynths/<generator>/{0_real,1_fake}/*.png`

## 3. CocoGlide — localization arm *(open, small)*

512 GLIDE text-guided inpainted images from COCO-2017-val crops, **each with a
binary localization mask**. Small (~200 MB) and the only source here that
supports plausibility metrics.

```bash
curl -L -o cocoglide.zip https://grip.unina.it/download/prog/TruFor/CocoGlide.zip
unzip cocoglide.zip -d data/raw/cocoglide
```

Mirror: `huggingface.co/datasets/nebula/CocoGlide`

Expected layout: `data/raw/cocoglide/{fake,real}/*.png` and
`data/raw/cocoglide/masks/*.png` (mask stem must match the image stem).

## 4. GenImage — diffusion arm at scale *(open, large)*

8 generators (SD1.4, SD1.5, Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN).
Full set is ~1 TB — **download one or two generator subsets only.**

```bash
huggingface-cli download Yuchuan-Wang/GenImage --repo-type dataset \
    --include "stable_diffusion_v_1_4/*" --local-dir data/raw/genimage
```

Expected layout: `data/raw/genimage/<generator>/{train,val}/{nature,ai}/*.png`

## 5. FaceForensics++ / DF40 — face axis *(request form, submit day 1)*

Not needed for the core result, but they add the faceswap and reenactment
families and turnaround is 1–3 days, so send the requests immediately.

- FF++: https://github.com/ondyari/FaceForensics (form)
- DF40: https://github.com/YZY-stack/DF40 (form; 40 forgery methods)

Extract frames into flat per-method folders, then register them in
`scripts/01_build_manifest.py` with `M.scan_generic(...)`:

```python
rows += M.scan_generic(raw / "ffpp" / "Deepfakes", "ffpp", "deepfakes", 1,
                       family="faceswap")
rows += M.scan_generic(raw / "ffpp" / "original", "ffpp", "real", 0)
```

---

## Adding any other dataset

Write one `scan_*` function in `src/xgf/data/manifest.py` returning rows with
these columns, and nothing downstream changes:

| column | meaning |
|---|---|
| `path` | absolute image path |
| `label` | 1 = fake/synthetic, 0 = real |
| `generator` | concrete model, lowercased (`"stylegan2"`, `"sd14"`) |
| `family` | synthesis paradigm from `config.FAMILIES` |
| `source` | dataset name |
| `split` | `train`/`val`/`test`, or `""` to auto-assign |
| `mask_path` | GT manipulation mask, or `""` |

Splits are assigned **within** each `(generator, label)` group. That matters: if
a generator straddled the train/test boundary unevenly, the cross-generator
transfer matrix would leak and the whole experiment would be meaningless.

---

## Licensing

ArtiFact, ForenSynths, GenImage, CocoGlide and DeepfakeBench carry
research/non-commercial terms. Fine for an academic BTP — note it if any result
is later commercialised. FF++ and DF40 additionally require signed agreements;
do not redistribute the images, only your manifest CSVs and derived metrics.
