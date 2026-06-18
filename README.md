# GutCoder


GutCoder is an attention-based transformer encoder that learns microbial community structure in the gut during healthy and chronic inflammatory disease states from high dimensional, noisy relative abundance data.

## Overview
- GutCoder is extensively pretrained on unlabelled species relative abundance data using a Masked Modelling approach
- It is finetuned on labelled Chronic Inflammatory cohorts - IBD, T2D and CRC to distinguish from healthy conditions
  - Implements Nested Leave-One-Study-Out Cross validation (LOSO-CV) to rigorously evaluate supervised classification tasks with a baseline Multi-Layer Perceptron (MLP)
  - Implements unsupervised k-means clustering to visualize and interpret separation of diseases in the latent embedding space with a traditional Autoencoder
- Mean attention scores and species co-attention is calculated to correlate high attended microbial species with existing literature

## Dataset
Species-relative abundance data from the `curatedMetagenomicData` along with its metadata is used for training GutCoder. 
To do this, first, the `curatedMetagenomicData` package (Version 3.18.0) is installed via Bioconductor:

```r
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("curatedMetagenomicData")
```

## Methodology

## Key Results

### Finetuning Performance (Supervised Classification)

| Model | IBD vs Healthy | | T2D vs Healthy | | CRC vs Healthy | |
|-------|---------------|---|---------------|---|---------------|---|
| | CV | Test | CV | Test | CV | Test |
| GutCoder | 0.72 | 0.69 | 0.56 | 0.62 | 0.62| 0.66 |
| MLP | 0.68 | 0.69 | 0.55 | 0.49 | 0.67 | 0.83 |

*AUC scores from nested LOSO-CV. CV = OOF cross-validation AUC; Test = held-out test AUC.*

- Achieves similar performance to standard MLP in the IBD task
- Outperforms standard MLP in the T2D task
- Struggles with the CRC task compared to standard MLP

### Cluster Composition analysis 

  

## Concluding remarks

## Citation
Pasolli E, Schiffer L, Manghi P, Renson A, Obenchain V, Truong D, Beghini F, Malik F, Ramos M, Dowd J, Huttenhower C, Morgan M, Segata N, Waldron L (2017). “Accessible, curated metagenomic data through ExperimentHub.” Nat. Methods, 14(11), 1023–1024. ISSN 1548-7091, 1548-7105. doi:10.1038/nmeth.4468.
