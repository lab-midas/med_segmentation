# MedSeg: Medical Segmentation
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-10-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

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
Densely connected networks (including modifications: dilated convs, ...)

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

## Contributors 
Thanks to Marc Fischer for providing the med_io pipeline around which this framework was structured.  

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/marcfi"><img src="https://avatars2.githubusercontent.com/u/48595245?v=4?s=100" width="100px;" alt=""/><br /><sub><b>marcfi</b></sub></a><br /><a href="https://github.com/lab-midas/med_segmentation/commits?author=marcfi" title="Code">💻</a> <a href="#ideas-marcfi" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-marcfi" title="Maintenance">🚧</a> <a href="#tool-marcfi" title="Tools">🔧</a></td>
    <td align="center"><a href="https://sites.google.com/site/kspaceastronauts"><img src="https://avatars1.githubusercontent.com/u/15344655?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Thomas Kuestner</b></sub></a><br /><a href="https://github.com/lab-midas/med_segmentation/commits?author=thomaskuestner" title="Code">💻</a> <a href="#ideas-thomaskuestner" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-thomaskuestner" title="Maintenance">🚧</a> <a href="#projectManagement-thomaskuestner" title="Project Management">📆</a> <a href="https://github.com/lab-midas/med_segmentation/commits?author=thomaskuestner" title="Documentation">📖</a> <a href="https://github.com/lab-midas/med_segmentation/pulls?q=is%3Apr+reviewed-by%3Athomaskuestner" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/lab-midas/med_segmentation/commits?author=thomaskuestner" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://www.is.mpg.de/de/people/thepp"><img src="https://avatars1.githubusercontent.com/u/30172495?v=4?s=100" width="100px;" alt=""/><br /><sub><b>tobiashepp</b></sub></a><br /><a href="https://github.com/lab-midas/med_segmentation/commits?author=tobiashepp" title="Code">💻</a> <a href="#ideas-tobiashepp" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-tobiashepp" title="Maintenance">🚧</a> <a href="#tool-tobiashepp" title="Tools">🔧</a> <a href="#security-tobiashepp" title="Security">🛡️</a></td>
    <td align="center"><a href="https://github.com/a-doering"><img src="https://avatars1.githubusercontent.com/u/35858164?v=4?s=100" width="100px;" alt=""/><br /><sub><b>a-doering</b></sub></a><br /><a href="https://github.com/lab-midas/med_segmentation/commits?author=a-doering" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/KaijieMo1"><img src="https://avatars3.githubusercontent.com/u/69183027?v=4?s=100" width="100px;" alt=""/><br /><sub><b>KaijieMo1</b></sub></a><br /><a href="https://github.com/lab-midas/med_segmentation/commits?author=KaijieMo1" title="Code">💻</a> <a href="https://github.com/lab-midas/med_segmentation/commits?author=KaijieMo1" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/sergiosgatidis"><img src="https://avatars1.githubusercontent.com/u/52936169?v=4?s=100" width="100px;" alt=""/><br /><sub><b>sergiosgatidis</b></sub></a><br /><a href="#ideas-sergiosgatidis" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-sergiosgatidis" title="Project Management">📆</a> <a href="#question-sergiosgatidis" title="Answering Questions">💬</a></td>
    <td align="center"><a href="https://github.com/SarahMue"><img src="https://avatars3.githubusercontent.com/u/30621951?v=4?s=100" width="100px;" alt=""/><br /><sub><b>SarahMue</b></sub></a><br /><a href="https://github.com/lab-midas/med_segmentation/commits?author=SarahMue" title="Code">💻</a> <a href="#ideas-SarahMue" title="Ideas, Planning, & Feedback">🤔</a> <a href="#plugin-SarahMue" title="Plugin/utility libraries">🔌</a> <a href="#maintenance-SarahMue" title="Maintenance">🚧</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/CDStark"><img src="https://avatars.githubusercontent.com/u/64153881?v=4?s=100" width="100px;" alt=""/><br /><sub><b>CDStark</b></sub></a><br /><a href="https://github.com/lab-midas/med_segmentation/commits?author=CDStark" title="Code">💻</a> <a href="#maintenance-CDStark" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://github.com/fl3on"><img src="https://avatars.githubusercontent.com/u/57361099?v=4?s=100" width="100px;" alt=""/><br /><sub><b>fl3on</b></sub></a><br /><a href="https://github.com/lab-midas/med_segmentation/commits?author=fl3on" title="Code">💻</a> <a href="https://github.com/lab-midas/med_segmentation/commits?author=fl3on" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/hksonngan"><img src="https://avatars.githubusercontent.com/u/780912?v=4?s=100" width="100px;" alt=""/><br /><sub><b>haison</b></sub></a><br /><a href="https://github.com/lab-midas/med_segmentation/issues?q=author%3Ahksonngan" title="Bug reports">🐛</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
