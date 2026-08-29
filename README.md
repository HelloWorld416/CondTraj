# CondTraj

PyTorch implementation of **Fine-Grained Motion Pattern-Guided Conditional
Diffusion for Pedestrian Trajectory Prediction**. This release supports the SDD
and NBA datasets.

## Environment

- Python 3.10+
- PyTorch 1.12+
- CUDA 11.3+

```bash
pip install -r requirements.txt
```

## Data Preparation

Place the preprocessed datasets under `data/`:

```text
data/
|-- sdd/
|   |-- train_8_12.npy
|   `-- val_8_12.npy
`-- nba/
    |-- nba_train.npy
    `-- nba_test.npy
```

Dataset files, checkpoints, logs, and visualizations are excluded from the
repository.

## Configuration

| Dataset | Observation | Prediction | DCT coefficients | Reverse steps | Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| SDD | 8 | 12 | 10 | 5 | 20 |
| NBA | 10 | 20 | 20 | 5 | 20 |

Training parameters are defined in `configs/sdd.yml` and `configs/nba.yml`.
SDD training uses time-reversal augmentation; evaluation does not apply this
augmentation.

## Training and Evaluation

Train the motion-pattern classifier before training CondTraj.

SDD:

```bash
python main.py --cfg configs/sdd.yml --mode train_classifier
python main.py --cfg configs/sdd.yml --mode train
python main.py --cfg configs/sdd.yml --mode eval
```

NBA:

```bash
python main.py --cfg configs/nba.yml --mode train_classifier
python main.py --cfg configs/nba.yml --mode train
python main.py --cfg configs/nba.yml --mode eval
```

By default, checkpoints are stored as `checkpoint/<DATASET>/classifier.pt` and
`checkpoint/<DATASET>/condtraj.pt`. Use `--classifier-ckpt` and `--model-ckpt`
to provide custom paths. Evaluation reports APD, minADE, and minFDE.

## License

This project is released under the [Apache License 2.0](LICENSE).

## Acknowledgement

The code implementation borrows from
[HumanMAC](https://github.com/LinghaoChan/HumanMAC) and
[LED](https://github.com/MediaBrain-SJTU/LED). The data preprocessing pipeline
is the same as [CGD-TraP](https://github.com/HelloWorld416/CGD-TraP). We thank
the authors for sharing their work and code.
