# MedSeg: Medical Segmentation

## Introduction
This repository contains code to train and evaluate 3D Convolutional Neural Networks for semantic segmentation on medical images.
The architectures developed in this framework are a combination of auto-encoder [UNet](https://arxiv.org/abs/1505.04597) with shortcut connections as in [ResNet](https://arxiv.org/abs/1512.03385), densely connections for deep supervision as in [DensetNet](https://arxiv.org/abs/1608.06993) and Merge-And-Run mapping for attention focusing as in [MRGE](https://arxiv.org/abs/1611.07718).

## Credits
Many thanks to all [contributors](##Contributors) of this repository. If you like it, please click on Star!<br/>

If you use this package for your research, please cite the paper:<br/>

Küstner T, Hepp T, Fischer M, Schwartz M, Fritsche A, Häring HU, Nikolaou K, Bamberg F, Yang B, Schick F, Gatidis S, Machann J <br/>
"Fully automated and standardized segmentation of adipose tissue compartments via deep learning in 3D whole-body MRI of epidemiological cohort studies" Radiology Artificial Intelligence 2020.<br/>
[[BibTeX](readme/kuestner_ryai2020.bib)]&nbsp;&nbsp;&nbsp;[[Endnote](readme/kuestner_ryai2020.ris)] 

```bibtex
@article{Kuestner2020,
    title={Fully automated and standardized segmentation of adipose tissue compartments via deep learning in 3D whole-body MRI of epidemiological cohort studies},
    author={Thomas K\"ustner and Tobias Hepp and Marc Fischer and Martin Schwartz and Andreas Fritsche and Hans-Ulrich Häring and Konstantin Nikolaou and Fabian Bamberg and Bin Yang and Fritz Schick and Sergios Gatidis and J\"urgen Machann},
    journal={Radiology Artificial Intelligence},
    year={2020},
}
```

## Documentation

### Installation
Clone the repository and install the requirements
```shell
$ git clone https://gitlab.com/iss_mia/cnn_segmentation/ desired_directory
$ python3 -m pip install -r requirements.txt
```

### Usage
Set all parameters in the [configuration file](./config/config_default.yaml). Check call arguments:
```shell
$ python3 main.py -h 
```

#### Preprocessing
Conversion of input data (DICOM, NiFTY, Matlab/HDF5) to TFRecords
```shell
$ python3 main.py --preprocess -c config/config_default.yml
```

#### Training
Network training on specified databases
```shell
$ python3 main.py --train -c config/config_default.yml -e experiment_name
```

#### Evaluation
Evaluate metrics of trained network
```shell
$ python3 main.py --evaluate -c config/config_default.yml -e experiment_name
```

#### Prediction
Predict segmentation mask for test dataset with trained network
```shell
$ python3 main.py --predict -c config/config_default.yml -e experiment_name
```

#### Database information
Get database information stored in `Patient` class
```shell
$ python3 readme/Patient_example.py -c config/config_default.yml
```

### Networks/Architectures 
Architectures are implemented in [/models/ModelSet.py](./models/ModelSet.py).

#### DCNet
densely connected with merge-and-run mapping network

<img src=readme/MRGE.png width="70%"></img>

#### Dilated dense convolution
DCNet: Densely connected networks (including modifications: dilated convs, ...)

<img src=readme/dilatedDense.png width="70%"></img>

#### UNet
vanilla UNet and extension for inclusion of positional input

#### Dilated DenseNet
Densely connected network with dilations

### Loss functions
Custom loss functions are defined in [/models/loss_function.py](./models/loss_function.py).

### Metrics
Custom evaluation metrics are defined in [/models/metrics.py](./models/metrics.py).

### Applications
- Whole-body semantic organ segmentation
- Whole-body adipose tissue segmentation

## License
This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

##Contributors 
Thanks to Marc Fischer for providing the med_io pipeline around which this framework was structured.  