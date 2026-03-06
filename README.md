# RAC: Rectified Flow Auto Coder

[[Paper]]() [[Project Page]]() 

Here is a nano demonstration, which is the initial proof-of-concept code we developed at the beginning of this project. It does not include the significant contributions mentioned in the paper (i.e., Informal implementation).

More details will be released after the code is cleaned.

![RAC Cover](./static/images/cover.png)

**Fig. 1:** The trajectory demonstration of RAC: Make the reconstruction task a condition generation task; Make the decoder the encoder; Make the single-step decoding and encoding a multi-step decoding and encoding.

## Usage

```code
conda create -n RAC python=3.11 -y
conda activate RAC
pip install numpy matplotlib tqdm torch torchvision diffusers
python train_nano_rac.py
```

## BibTex

```code

```
