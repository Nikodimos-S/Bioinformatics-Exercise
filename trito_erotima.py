from collections import defaultdict
import numpy as np

# function to build the HMM profile
def build_hmm_profile(aligned_strings):
    profile = defaultdict(lambda: defaultdict(int))
    
    for column in zip(*aligned_strings):
        for char in column:
            profile[column][char] += 1
    
    return profile

# function to calculate alignment scores and paths for sequences in datasetB
def alignment_score_path(hmm_profile, sequence):
    n = len(sequence)
    m = len(hmm_profile)
    dp = np.zeros((n+1, m+1))
    
    # calculate the alignment score matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            match = dp[i-1][j-1] + hmm_profile[j-1].get(sequence[i-1], -1)
            delete = dp[i-1][j] - 1
            insert = dp[i][j-1] - 1
            dp[i][j] = max(match, delete, insert)
    
    # reconstruct the alignment path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        if dp[i][j] == dp[i-1][j-1] + hmm_profile[j-1].get(sequence[i-1], -1):
            path.append((i-1, j-1))
            i, j = i-1, j-1
        elif dp[i][j] == dp[i-1][j] - 1:
            i = i-1
        else:
            j = j-1
    
    path.reverse()
    return dp[n][m], path