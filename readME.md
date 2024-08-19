# Multimodal Drivers of Attention Interruption to Baby Product Video Ads

This repository contains the code and data for the paper titled "Multimodal Drivers of Attention Interruption to Baby Product Video Ads," published at ICPR 2024.

<img src="figs/framework.png" alt="Overview" width="256"/>

## Table of Contents

- [Dataset](#dataset)
- [Model](#model)
- [Feature Extraction and Analysis](#feature-extraction-and-analysis)
- [How to Run](#how-to-run)
- [Results](#results)
- [Contact](#contact)
- [Citations](#citations)

## Dataset

The dataset used in this study consists of video ads for baby products, annotated with viewers' points of interest during their viewing of the ads. Figure below is an illustration, where the red dots indicate the location of points of interest.

<img src="figs/attention.png" alt="Points of Interest" width="256"/>

We extracted visual, audio, and linguistic features along with an attention interruption measure. The feature extraction code can be found in the `feature extraction` folder.

- **Visual Features:** Extracted using image processing techniques, including 78 features such as color, texture, and object detection.
- **Audio Features:** Extracted from the audio tracks of the videos, including 63 features such as RMS, pitch, and spectral features.
- **Linguistic Features:** Derived from the textual content of the ads, encompassing 156 features such as sentiment, complexity, and thematic elements.

We have also split the dataset into training and testing datasets for future research. All datasets can be found in the `dataset` folder.

## Model

We built a multimodality feature-infused model for predicting attention interruption. The model is visualized below:

<img src="figs/model.png" alt="Model Architecture" width="32"/>

Our model outperformed benchmark models in predicting attention interruption, as shown in the table below:

<img src="figs/result.png" alt="Results Comparison" width="32"/>

## Feature Extraction and Analysis

We employed a linear regression model to analyze the relationship between multimodal features and attention interruption. The code for feature reduction and regression estimation can be found in the `feature importance` folder.

## How to Run

1. Clone the repository
2. Run each notebook for feature extraction, PCA, or model training and testing.
3. Run the R code for feature importance analysis

## Contact

For any questions or inquiries, please contact the authors at we.xie@northeastern.edu.

## Citations

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wen2024multimodal,
  title={Multimodal Drivers of Attention Interruption to Baby Product Video Ads},
  author={Xie, Wen and Luan, Lingfei and Zhu, Yanjun and Bart, Yakov and Ostadabbas, Sarah},
  booktitle={International Conference on Pattern Recognition (ICPR)},
  month={12},
  year={2024}
}
```
