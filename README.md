# Speaker Voice Separation using Neural Nets
[![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/sczhou/CodeFormer)

## Installation

```bash
git clone https://github.com/Muhammad-Ahmad-Ghani/svoice_demo.git
cd svoice_demo
conda create -n svoice python=3.7 -y
conda activate svoice
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
```

| Pretrained-Model | Dataset | Epochs | Train Loss | Valid Loss |
|:------------:|:------------:|:------------:|:------------:|:------------:
| [checkpoint.th](https://drive.google.com/drive/folders/1WzhvH1oIB9LqoTyItA6jViTRai5aURzJ?usp=sharing) | Librimix-7 (16k-mix_clean) | 31 | 0.04 | 0.64 |

This is an intermediate checkpoint just for demo purpose.

create directory ```outputs/exp_``` and save checkpoint there
```
svoice_demo
├── outputs
│   └── exp_
│       └── checkpoint.th
...
```

## Run Gradio Demo
```bash
conda activate svoice
python demo.py
```
 
## Training
Create dataset ```mix_clean``` with sample rate ```16K``` using [librimix](https://github.com/shakeddovrat/librimix) repo.

Dataset Structure
```
svoice_demo
├── Libri{NUM_OF_SPEAKERS}Mix_Dataset -> Libri7Mix_Dataset
│   └── wav{SAMPLE_RATE_VALUE}k -> wav16k
│       └── min
│       │   └── dev
│       │       └── ...
│       │   └── test
│       │       └── ...
│       │   └── train-360
│       │       └── ...
...
```

#### Create ```metadata``` files
Run predefined scripts if you want.
```
# for 7 speakers
bash create_metadata_librimix7.sh
# for 10 speakers
bash create_metadata_librimix10.sh
```

Change ```conf/config.yaml``` according to your settings. Set ```C: NUM_OF_SPEAKERS``` value at line 66 for number of speakers.

```
python train.py
```
This will automaticlly read all the configurations from the `conf/config.yaml` file.
To know more about the training you may refer to original [svoice](https://github.com/facebookresearch/svoice) repo.

#### Distributed Training

```
python train.py ddp=1
```

### Evaluating

```
python -m svoice.evaluate <path to the model> <path to folder containing mix.json and all target separated channels json files s<ID>.json>
```

### Citation

The svoice code is borrowed from original [svoice](https://github.com/facebookresearch/svoice) repository. All rights of code are reserved by [META Research](https://github.com/facebookresearch).

```
@inproceedings{nachmani2020voice,
  title={Voice Separation with an Unknown Number of Multiple Speakers},
  author={Nachmani, Eliya and Adi, Yossi and Wolf, Lior},
  booktitle={Proceedings of the 37th international conference on Machine learning},
  year={2020}
}
```
```
@misc{cosentino2020librimix,
    title={LibriMix: An Open-Source Dataset for Generalizable Speech Separation},
    author={Joris Cosentino and Manuel Pariente and Samuele Cornell and Antoine Deleforge and Emmanuel Vincent},
    year={2020},
    eprint={2005.11262},
    archivePrefix={arXiv},
    primaryClass={eess.AS}
}
```
## License
This repository is released under the CC-BY-NC-SA 4.0. license as found in the [LICENSE](LICENSE) file.

The file: `svoice/models/sisnr_loss.py` and `svoice/data/preprocess.py` were adapted from the [kaituoxu/Conv-TasNet][convtas] repository. It is an unofficial implementation of the [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation][convtas-paper] paper, released under the MIT License.
Additionally, several input manipulation functions were borrowed and modified from the [yluo42/TAC][tac] repository, released under the CC BY-NC-SA 3.0 License.

[icml]: https://arxiv.org/abs/2003.01531.pdf
[icassp]: https://arxiv.org/pdf/2011.02329.pdf
[web]: https://enk100.github.io/speaker_separation/
[pytorch]: https://pytorch.org/
[hydra]: https://github.com/facebookresearch/hydra
[hydra-web]: https://hydra.cc/
[convtas]: https://github.com/kaituoxu/Conv-TasNet 
[convtas-paper]: https://arxiv.org/pdf/1809.07454.pdf
[tac]: https://github.com/yluo42/TAC
[nprirgen]: https://github.com/ty274/rir-generator
[rir]:https://asa.scitation.org/doi/10.1121/1.382599
