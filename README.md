# GutCoder

GutCoder is an attention-based transformer encoder that learns community interactions within the gut microbiome during healthy and chronic inflammatory disease states by recovering biological signals from high dimensional, noisy relative abundance data.

## Objectives
- Develop a microbiome foundation model extensively pretrained on unlabelled species relative abundance data using a Masked Modelling approach
- Finetune on labelled Chronic Inflammatory cohorts - IBD, T2D and CRC to distinguish from healthy conditions
  - Implement Nested Leave-One-Study-Out Cross validation (LOSO-CV) to rigourously evaluate supervised classification tasks with a baseline Multi-Layer Perceptron (MLP)
  - Implement unsupervised k-means clustering to visualize and interpret separation of diseases in the latent embedding space with a traditional Autoencoder
- Calculate mean attention scores and species co-attention to correlate top 15 microbial species with existing literature

## Dataset
Species-relative abundance data from the `curatedMetagenomicData` along with its metadata is used for training GutCoder. 
To do this, first, The `curatedMetagenomicData` package (Version 3.18.0) is installed via Bioconductor:

```r
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("curatedMetagenomicData")
```

## Methodology

## Key Results

## Concluding remarks

## Key references
