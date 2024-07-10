# VC-NET
Implementation for paper Label-free prediction of connectivity in perfused microvascular networks in vitro based on deep learning 
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
`python train.py  --data-path 'your_dataset_path'  --device 'cuda'  --b 8  --num_classes 2  --lr 0.01  --output-dir 'your_output_path'`

## Test
`python predict_multi_class.py `

## References
* [MPSCL](https://github.com/TFboys-lzz/MPSCL)
