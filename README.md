# NYCU Computer Vision 2026 HW4

* **Student ID:** 111550017
* **Name:** 藍逸薰

## Introduction

This homework focuses on single-image restoration for mixed weather degradations (rain and snow).
The selected method is a PromptIR-style unified model that uses a transformer U-Net backbone with prompt-guided decoder adaptation, so one model can handle both degradation types.
Model: https://drive.google.com/file/d/1A878HaYS4z9BjGsX5TsHkMlimaW0TlMz/view?usp=sharing

---

## Environment Setup

How to install dependencies.

```bash
pip install -r requirements.txt
```
---

## Usage

# Training

folder format:  
folder 

&emsp; train.py 
&emsp; predict.py  
&emsp; utils.py  
&emsp; dataset.py
&emsp; model_promptir.py

&emsp; train  
&emsp;&emsp; degraded
&emsp;&emsp; clean
&emsp; test  
&emsp;&emsp; degraded 




# Training(Baseline)
```bash
python train.py --data-root . --save-dir checkpoints --epochs 50 --batch-size 2 --patch-size 128 --val-ratio 0.1 --amp
```
# Training(Full dataset with resume checkpoint)
```bash
python train.py --data-root . --save-dir checkpoints_full --init-ckpt checkpoints_pt/best.pt --epochs 25 --batch-size 2 --patch-size 128 --val-ratio 0 --amp
```
# Testing

```bash
python predict.py --data-root . --ckpt checkpoints_pt/best.pt --out pred.npz --amp --tile-size 192 --tile-overlap 32
```
---

## Performance Snapshot

<img width="1182" height="52" alt="Image" src="leaderboard.png" />
