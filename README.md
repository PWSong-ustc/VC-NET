# VC-NET
Implementation for paper **Label-free Prediction of Connectivity in Perfused Microvascular Networks in Vitro based on Deep Learning**
## Installation
### Requirements
* timm>=0.4.12
* einops>=0.4.1
* numpy>=1.22.0
* apex>=0.1
* pytorch>=1.10.1
* torchvision>=0.11.2
* scipy>=1.8.0
* yaml>=0.2.5
* pyyaml>=6.0
* yacs>=0.1.8
* matplotlib>=3.5.1
* opencv-python>=4.5.5.62
* pandas>=1.4.0
* pillow>=8.4.0
* argparse>=1.4.0
* tensorboardx>=2.4
* medpy>=0.4.0
* scikit-learn>=1.1.2
* imgaug>=0.4.0

## Training
If you want to train your own model, please run:
```
python train.py \ 
  --batch_size  8 \ 
  --num_classes 2 \ 
  --lr 0.01 \
  --device 'cuda' \  
  --data-path 'your_dataset_path' \ 
  --output-dir 'your_output_path'
```

## Prediction
Once you have the trained model, you can obtain the prediction image for a single image.  
Simpliy run： `python predict_multi_class.py `

## Results
### Weights
To get the best model and the best center weights, [please download in this link](https://drive.google.com/drive/folders/15aGK6R0rNf_ARf3sgIAVS6wS_3W-jJeu?usp=sharing)  

### Dataset
For the MVNs_connectivity_dataset, please contact us(xul666@mail.ustc.edu.cn).

## References
* [MPSCL](https://github.com/TFboys-lzz/MPSCL)
* [MoCo](https://github.com/facebookresearch/moco)
