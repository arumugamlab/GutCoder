import numpy as np
import pandas as pd
# QC
# Conversion of percentage to fractions

def fraction_convert(percentage_train):

    Xtrain_percent = percentage_train.astype(float)
    
    #Convert to range 0-1
    if Xtrain_percent.max() > 1:
        Xtrain_percent = Xtrain_percent/ 100.0
    
    return Xtrain_percent


# # Feature engineering - Discretization by quantile binning

"""
    Convert a raw abundance matrix into discrete bins.

    For each sample:
    1. Non-zero species are sorted by descending abundance.
    2. They are divided evenly into bins. Zero abundance bins go to bin 0. Highest goes to the highest bin.
    When a sample has fewer detected species than the bins, each species is mapped individually using an interpolated rank.
    """

def abundance_binning(abundance,bins):
    
    # Load relative abundance numpy file
    abundance=np.asarray(abundance)
    rows,cols=abundance.shape
    bin_mat=np.zeros((rows, cols), dtype=np.int16)


    # Find non-zero abundance and sort in descending order
    # Iterate sample by sample
    # Extract abundance of each sample
    # Check if they are non-zero and count how many 
    
    for i in range (rows):
        sample = abundance[i]
        # Only detected species to be binned, others automatically assigned zero
        non_zero = np.flatnonzero(sample>0)
        non_zero_size = non_zero.size
        if non_zero_size == 0:
            continue
        value=sample[non_zero]
        # Descending order
        order=np.argsort(-value,kind="quicksort")
        sorted_indices=non_zero[order]
        # Divide the sorted values evenly into bins
        if non_zero_size >= bins:
            quotient=non_zero_size//bins
            remainder=non_zero_size%bins
            start=0 
            
            for j in range (bins):
                current_bin_size=quotient + (1 if j< remainder else 0)
                # Bin assignment
                end = start + current_bin_size
                bin_mat [i,sorted_indices [start:end]] = (bins-j)
                start=end
                # High abundant species goes to high bin
                
        else:
            # What happens if non_zero_size<bins
            
            ranks = np.arange(non_zero_size, dtype=np.float32)
            mapped = np.floor((bins - 1) * (1 - ranks / max(non_zero_size - 1, 1)) + 1).astype(np.int16)
            bin_mat[i, sorted_indices] = mapped

    return bin_mat
