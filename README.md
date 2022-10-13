# Speaker Voice Separation using Neural Nets Gradio Demo

## Installation For Separation

### For Older GPU

```bash
git clone https://github.com/Muhammad-Ahmad-Ghani/svoice_demo.git
cd svoice_demo
conda create -n svoice python=3.7 -y
conda activate svoice
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt  
```

### For RTX series GPU

In file svoice/solver.py at line 209 replace the code
```
sisnr_loss, snr, est_src, reorder_est_src = cal_loss(sources, est_src, lengths)
```
With
```
sisnr_loss, snr, est_src, reorder_est_src = cal_loss(sources, estimate_source[c_idx], lengths)
```

```bash
git clone https://github.com/Muhammad-Ahmad-Ghani/svoice_demo.git
cd svoice_demo
conda create -n svoice python=3.7 -y
conda activate svoice
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

## Installation For Speech-To-Text (Nvidia NEMO)

```bash
conda create -n nemo python=3.8 -y
conda activate nemo
conda install -c nvidia/label/cuda-11.4.4 cuda-toolkit -y
pip install nemo_toolkit[all]
pip install flask flask-cors asgiref uvicorn[standard]
```

#### Pre-trained [checkpoint](https://drive.google.com/drive/folders/1WzhvH1oIB9LqoTyItA6jViTRai5aURzJ?usp=sharing) trained on ```librimix7``` (19 epochs with 0.62 train and 1.17 val loss) just for demo purpose.
create directory outputs/exp_ and save checkpoint there.

## Running End To End project
#### Terminal 1
```bash
conda activate svoice
# for backend server
python app.py
# for interactive demo
python demo.py
```

#### Terminal 2
 ```bash
 conda activate nemo
 python app_transcribe.py
 ```
 
## Training
To know more about the training you may refer to original [svoice](https://github.com/facebookresearch/svoice) repo.
```
python train.py
```

This will automaticlly read all the configurations from the `conf/config.yaml` file.

#### Distributed Training

```
python train.py ddp=1
```

### Evaluating

```
python -m svoice.evaluate <path to the model> <path to folder containing mix.json and all target separated channels json files s<ID>.json>
```

### Citation

The Code is borrowed from Original [svoice](https://github.com/facebookresearch/svoice) repository. All rights of code are reserved by facebookresearch.

```
@inproceedings{nachmani2020voice,
  title={Voice Separation with an Unknown Number of Multiple Speakers},
  author={Nachmani, Eliya and Adi, Yossi and Wolf, Lior},
  booktitle={Proceedings of the 37th international conference on Machine learning},
  year={2020}
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
