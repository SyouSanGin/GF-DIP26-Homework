# Assignment 2 - DIP with PyTorch

本仓库为高凡(SA25001019) DIP HW2 DIP with PyTorch 作业代码仓

## Requirements

You can create the environment with the following command (conda is required!):

```bash
conda env create -f environment.yml
```

## Task 1: Poisson Editing with PyTorch

### Evaluation

Run the following command to start the gradio UI:

```bash
python ./run_blending_gradio.py
```

> Images for test are stored in 'data_poisson' folder

### Results

<img title="" src="images/README/cbb31f148cc4e533e955bb9bed618839fe4083a6.gif" alt="" data-align="center"><img title="" src="images/README/f3a3fee90da852d58b093b5c3322559140a8fb9f.gif" alt="" data-align="center"><img src="images/README/26f92a0b8e37eebf56c1ab257a2552cb50f0aadf.gif" title="" alt="" data-align="center">

## Task 2：Pix2Pix

### Data Preperation

1. Make sure your current working directory is `Assignments/02_DIPwithPyTorch/Pix2Pix`

2. Download the dataset (run the following command). We use `edges2shoes` dataset for training & validation.
   
   ```bash
   ./download_dataset.sh edges2shoes
   ```

### Training

Run the training script:

```bash
CUDA_VISIBLE_DEVICES=<YOUR_DEVICE_ID> python ./train.py
```

The checkpoints will be stored in `checkpoints` folder. Results on `train` & `valid` sets will be saved in `*_results`

### Evaluation

**Prepare the model** :  You can train the model by yourself or download the pretrained model at [Release HW-2 Model CKPT · SyouSanGin/GF-DIP26-Homework](https://github.com/SyouSanGin/GF-DIP26-Homework/releases/tag/HW-2) and put the file into `checkpoints` folder.

**Inference**： Run the following command to generate results:

```bash
python ./eval.py\
 --test_dir datasets/edges2shoes/val \
 --model_path <Where you store the checkpoint> 
```

Outputs will be saved in "test_results" folder.

### Results

The following results were all generated from the “val” portion of the edges2shoes dataset. For more results, see [HERE](./test_results)

**Average SSIM**: 0.915

From left to right: **edge map**, **generated result**, **ground truth**

<img src="images/README/2026-04-13-22-37-06-image.png" title="" alt="" data-align="center">

<img src="images/README/2026-04-13-22-37-21-image.png" title="" alt="" data-align="center">

<img src="images/README/2026-04-13-22-37-34-image.png" title="" alt="" data-align="center">

<img src="images/README/2026-04-13-22-37-43-image.png" title="" alt="" data-align="center">

<img src="images/README/2026-04-13-22-38-52-image.png" title="" alt="" data-align="center">
