# Data

The HAM10000 dataset is not bundled in this repo (~2.7GB). Download it yourself from the [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T):

1. Download `HAM10000_metadata.csv`, `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip`
2. Unzip both image archives into the same parent folder as the metadata CSV, e.g.:

```
HAM10000/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
```

3. Verify the layout:

```bash
python data_prep.py --data_dir HAM10000
```

4. Train:

```bash
python src/train.py --data_dir HAM10000 --config configs/gpu_8gb.json
```

## Alternative: Kaggle

Kaggle hosts a direct HAM10000 mirror, useful if you want to skip the Dataverse download and attach the dataset directly to a Kaggle or Colab notebook without a manual download step.

## Dataset summary

10,015 dermatoscopic images across seven diagnostic categories (akiec, bcc, bkl, df, mel, nv, vasc). See the main [README](../README.md#the-dataset) and the [wiki Dataset page](https://github.com/motazalqaoud/Oncology-diagnosis-using-CNN/wiki/Dataset) for the full class breakdown and why the pipeline splits by lesion ID rather than image ID.

## Citation

> Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Sci Data* 5, 180161 (2018). https://doi.org/10.7910/DVN/DBW86T
