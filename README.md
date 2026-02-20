# Memes Sentiment Analysis

Streamlined, runnable pipeline for multimodal meme sentiment/intention/topic classification using ViT (images) + XLM-R (text).

## Repo Map
- `config.yml` — single source of paths, Azure creds, model settings, download link slots.
- `environment.yml` — conda env spec (CPU by default; swap `cpuonly` for `pytorch-cuda=11.8` if on NVIDIA GPU).
- `cleaning_data.py` — moves bad/empty memes to `rejected_stuff/` and saves cleaned Excel.
- `Azure_OCR_meme.py` — fills missing `Description` via Azure Read API; rejects images with no text.
- `Visual_features_extraction.py` — ViT feature extraction pipeline for train/test splits.
- `xlmr_text_processor.py` — text cleaning + XLM-R feature extraction for train/test splits.
- `extract_features.py` — convenience runner that calls both visual + text pipelines.
- `multimodal_fusion.py` — fusion model training/validation using extracted features.
- `features/` — precomputed training features already present (ViT, XLM-R, fused).
- `dataset/` — train/test CSVs + images (`dataset/train|test/images`).

## Quickstart
1) Create env
```bash
conda env create -f environment.yml
conda activate memes-sentiment
```
2) Configure `config.yml`
- Update `links.dataset_download_url`, `links.images_download_url`, `links.feature_pack_download_url` (placeholders provided).
- Azure creds already filled per request; adjust `azure.endpoint/region/key` if needed.
- Paths are repo-relative; adjust only if you move folders.
3) (Optional) OCR fill & clean
```bash
python Azure_OCR_meme.py    # writes cleaned Excel with OCR descriptions
python cleaning_data.py     # removes rows/files with missing labels -> saves cleaned Excel
```
4) Extract features (train + test)
```bash
python extract_features.py
```
- Outputs: `features/train_vit_features.csv`, `features/test_vit_features.csv`, `features/train_xlmr_text_features.csv`, `features/test_xlmr_text_features.csv`.
5) Train fusion model
```bash
python multimodal_fusion.py
```
- Saves best weights to `checkpoints/best_fusion.pt` and `checkpoints/best_classifier.pt`.

## Remaining Tasks / Notes
- Test split embeddings are not precomputed; step 4 will generate them.
- Fused features for test set (and an inference script) are still TODO.
- Add evaluation script to report metrics on held-out test once embeddings exist.
- Consider class-weighting or focal loss if class imbalance hurts performance.
- If running on GPU, remove `cpuonly` in `environment.yml` and add `pytorch-cuda` per CUDA version.

## Links (placeholders)
- Dataset: `<paste-public-dataset-link-here>`
- Images: `<paste-public-images-link-here>`
- Feature pack: `<paste-precomputed-features-link-here>`

## Tips
- Keep `Meme Number` filenames as `meme_<id>.jpg` etc. to match loaders.
- If Azure OCR rate limits, add small `time.sleep` inside the polling loop.
- Set a different `project.seed` in `config.yml` for reproducible splits if needed.
