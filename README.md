
# Depth-guided Texture Diffusion for Image Semantic Segmentation

![Main Image](path/to/your/image.png)

This repository contains the official implementation of the TCSVT journal paper **"Depth-guided Texture Diffusion for Image Semantic Segmentation"**. We will continue to maintain and update this repository.

## Requirements

- Python 3.x
- PyTorch
- Other dependencies can be installed using:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

To run the code, you first need to download the dataset. Please download it from the following [link](https://pan.baidu.com/s/1qu4UpCG_g6BKUv73B1Rs1w?pwd=erzq) and extract the data to the `data/` directory.


```bash
mkdir -p data
# Assume the dataset is downloaded and extracted here
```

## Pretrained Weights

You also need to download the pretrained weights. Place the downloaded weights into the `pretrain/` folder. Download them from the following [link](https://pan.baidu.com/s/1qu4UpCG_g6BKUv73B1Rs1w?pwd=erzq).

```bash
mkdir -p pretrain
# Place pretrained weights here
```

## Training

To train the model, you can use the `train.sh` script provided in the `scripts` folder:

```bash
bash scripts/train.sh
```

## Testing

To test the model, use the `test.sh` script:

```bash
bash scripts/test.sh
```

## Citation

If you find our work useful, please consider citing:
