# HWDiff: Hierarchical Wavelet-Guided Diffusion Model for Single Image Deblurring

## Datasets

| Dataset                          | Description          | Link                                        |
|----------------------------------|----------------------|---------------------------------------------|
| GoPro                            | Training + Testing   | [Download](https://github.com/subeeshvasu/Awesome-Deblurring) |
| HIDE                             | Testing              | [Download](https://github.com/subeeshvasu/Awesome-Deblurring) |
|RealBlur-R                        | Testing              | [Download](https://github.com/subeeshvasu/Awesome-Deblurring) |
|RealBlur-J                        | Testing              | [Download](https://github.com/subeeshvasu/Awesome-Deblurring) |

## Models
note: The trained models will be opened after the article is accepted.

## Requirements

- CUDA 10.1 (or later)
- Python 3.9 (or later)
- Pytorch 1.8.1 (or later)
- Torchvision 0.19
- OpenCV 4.7.0
- tensorboard, skimage, scipy, lmdb, tqdm, yaml, einops, natsort

## Training

1. In the coarse training pipeline, modify the comments on lines 143-152 and 225-269 in the `/basicsr/models/image_restoration_model.py` file, then run

    ```bash
    python  basicsr/train.py -opt Options/Deblurring_HFG.yml
    ```
2. In the fine training pipeline, modify the comments on lines 156-170 and 272-322 in the `/basicsr/models/image_restoration_model.py` file, then run

    ```bash
    python  basicsr/train.py -opt Options/Deblurring_HFG_fine.yml
    ```

## Testing

1. Download the pre-trained model and place it in ./pretrained_models/
2. Testing
   
   For GoPro dataset and HIDE dataset, run
   ```bash
   python test.py
   ```

   For RealBlur-R dataset and RealBlur-J dataset, run
   ```bash
   python test_real.py
   ```
3.Calculating PSNR/SSIM scores

   For GoPro dataset and HIDE dataset, run
   ```bash
   python calculate_psnr_ssim.py
   ```

   For RealBlur-R dataset and RealBlur-J dataset, run
   ```bash
   python evaluate_realblur.py
   ```

## Acknowledgements

This code is built on BasicSR, Restormer, and C2F-DFT.
