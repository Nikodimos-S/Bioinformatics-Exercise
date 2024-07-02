import random
from proto_erotima import *
from deutero_erotima import *
from trito_erotima import *

# Generation of the 50 strings
all_strings = [synthesize_string() for _ in range(50)]

# Splitting strings into dataset A and B
random.shuffle(all_strings)
datasetA = all_strings[:15]
datasetB = all_strings[15:]

print("Dataset A:", datasetA)
print("Dataset B:", datasetB)

# Perform multiple sequence alignment for datasetA
aligned_datasetA = multiple_alignment(datasetA, a)

print("Aligned Dataset A:")
for aligned_string in aligned_datasetA:
    print(aligned_string)

# Construct the HMM profile
hmm_profile = build_hmm_profile(aligned_datasetA)

print("HMM Profile:")
for column, counts in hmm_profile.items():
    print(column, dict(counts))

# Calculate alignment scores and paths for datasetB
alignment_results = [alignment_score_path(hmm_profile, seq) for seq in datasetB]

print("Alignment Scores and Paths for Dataset B:")
for score, path in alignment_results:
    print("Score:", score, "Path:", path)