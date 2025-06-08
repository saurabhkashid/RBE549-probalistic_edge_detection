# RBE549-probalistic_edge_detection

This project implements an advanced **Probability of Boundary (Pb)** edge detection pipeline inspired by the BSDS500 benchmark. The method combines high-level perceptual cues:
- **Texture gradients (Tg)** using texton maps
- **Brightness gradients (Bg)**
- **Color gradients (Cg)**  
with classical edge detectors (Sobel + Canny) to produce accurate and perceptually meaningful boundary maps.

---

## 🚀 Project Goals

- Implement multi-scale, multi-orientation **filter banks**
- Generate **texton, brightness, and color maps** using KMeans clustering
- Apply **half-disc masks** and **chi-square distance** to compute local gradients
- Fuse feature gradients with **baseline edge detectors** to compute final **Pb edge maps**
- Evaluate Pb edge maps vs. **ground truth boundaries** (BSDS500)

---

## 🗺️ Pipeline

1️⃣ Filter Banks:
- Derivative of Gaussian (DoG) filters
- Leung-Malik (LM) filters
- Gabor filters

2️⃣ Feature Maps:
- Texton map (via LM + KMeans)
- Brightness map (grayscale + KMeans)
- Color map (Lab color space + KMeans)

3️⃣ Half-Disc Masks:
- Multi-scale, multi-orientation masks
- Compare left/right region distributions via **chi-square distance**

4️⃣ Pb Edge Map:
- Combine Tg, Bg, Cg gradients
- Fuse with Sobel and Canny baselines
- Output: **Pb edge probability map**

---
## Example Results

### Original Image
![Input](BSDS500\Images\1.jpg)

### Pb Edge
![Pb Edge](result\pb_img_0.png)
