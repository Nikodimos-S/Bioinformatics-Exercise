import random
import numpy as np
from collections import defaultdict

# Given patterns
patterns = ["AATTGA", "CGCTTAT", "GGACTCAT", "TTATTCGTA"]
abet = ["A", "C", "G", "T"]

# i) Generate sequences
def random_mutation(sequence):
    seq_list = list(sequence)
    num_mutations = random.randint(0, 2)
    for _ in range(num_mutations):
        index = random.randint(0, len(seq_list) - 1)
        if random.random() > 0.5:
            seq_list[index] = random.choice(abet)
        else:
            del seq_list[index]  # Corrected: removing the base properly
    return ''.join(seq_list)

def generate_sequence():
    initial_symbols = ''.join(random.choices(abet, k=random.randint(1, 3)))
    sequence = initial_symbols
    for pattern in patterns:
        mutated_pattern = random_mutation(pattern)
        sequence += mutated_pattern
    final_symbols = ''.join(random.choices(abet, k=random.randint(1, 2)))
    sequence += final_symbols
    return sequence

# Generate 50 sequences
sequences = [generate_sequence() for _ in range(50)]
# Split into datasetA and datasetB
random.shuffle(sequences)
datasetA = sequences[:15]
datasetB = sequences[15:]
# ii) Multiple Sequence Alignment (MSA)
a = 2 # AM is 21150 which is an even number
def needleman_wunsch(seq1, seq2, a):
    len1, len2 = len(seq1), len(seq2)
    score_matrix = np.zeros((len1 + 1, len2 + 1))
    traceback_matrix = np.zeros((len1 + 1, len2 + 1), dtype='str')
    # Initialize the score matrix
    for i in range(1, len1 + 1):
        score_matrix[i][0] = -a * i
        traceback_matrix[i][0] = '↑'
    for j in range(1, len2 + 1):
        score_matrix[0][j] = -a * j
        traceback_matrix[0][j] = '←'
    # Fill the score matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            match = score_matrix[i-1][j-1] + (1 if seq1[i-1] == seq2[j-1] else -a/2)
            delete = score_matrix[i-1][j] - a
            insert = score_matrix[i][j-1] - a
            max_score = max(match, delete, insert)
            score_matrix[i][j] = max_score
            if max_score == match:
                traceback_matrix[i][j] = '↖'
            elif max_score == delete:
                traceback_matrix[i][j] = '↑'
            else:
                traceback_matrix[i][j] = '←'
    # Traceback
    aligned_seq1 = ""
    aligned_seq2 = ""
    i, j = len1, len2
    while i > 0 or j > 0:
        if traceback_matrix[i][j] == '↖':
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            i -= 1
            j -= 1
        elif traceback_matrix[i][j] == '↑':
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = '-' + aligned_seq2
            i -= 1
        else:
            aligned_seq1 = '-' + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            j -= 1
    return aligned_seq1, aligned_seq2, score_matrix[len1][len2]

def progressive_alignment(sequences, a):
    aligned_seq = sequences[0]
    for i in range(1, len(sequences)):
        aligned_seq, aligned_next_seq, _ = needleman_wunsch(aligned_seq, sequences[i], a)
        aligned_seq = aligned_seq  # Corrected: updating aligned_seq to use the first sequence
    return aligned_seq
# Align the sequences in datasetA
aligned_datasetA = [progressive_alignment([datasetA[0], seq], a) for seq in datasetA[1:]]
# iii) Hidden Markov Model (HMM) Profile
def build_hmm_profile(aligned_sequences):
    n_sequences = len(aligned_sequences)
    seq_length = len(aligned_sequences[0])
    # Initialize probabilities
    transition_probs = defaultdict(lambda: defaultdict(float))
    emission_probs = defaultdict(lambda: defaultdict(float))
    state_count = defaultdict(int)
    # Count occurrences of each state and transitions
    for seq in aligned_sequences:
        prev_state = None
        for i, symbol in enumerate(seq):
            state = f"M{i+1}"
            state_count[state] += 1
            emission_probs[state][symbol] += 1
            if prev_state:
                transition_probs[prev_state][state] += 1
            prev_state = state
    # Normalize probabilities
    for state in state_count:
        total_emissions = sum(emission_probs[state].values())
        for symbol in emission_probs[state]:
            emission_probs[state][symbol] /= total_emissions
    for state in transition_probs:
        total_transitions = sum(transition_probs[state].values())
        for next_state in transition_probs[state]:
            transition_probs[state][next_state] /= total_transitions
    return transition_probs, emission_probs
def viterbi(sequence, transition_probs, emission_probs, start_prob):
    seq_length = len(sequence)
    states = list(transition_probs.keys())
    viterbi_matrix = np.zeros((len(states), seq_length))
    backpointer_matrix = np.zeros((len(states), seq_length), dtype='int')
    for s, state in enumerate(states):
        viterbi_matrix[s][0] = start_prob[state] * emission_probs[state].get(sequence[0], 1e-6)  # Corrected: Avoid zero probability
    for t in range(1, seq_length):
        for s, state in enumerate(states):
            max_prob = max(
                viterbi_matrix[prev_s][t-1] * transition_probs[states[prev_s]].get(state, 0) * emission_probs[state].get(sequence[t], 1e-6)
                for prev_s in range(len(states))
            )
            viterbi_matrix[s][t] = max_prob
            backpointer_matrix[s][t] = np.argmax([
                viterbi_matrix[prev_s][t-1] * transition_probs[states[prev_s]].get(state, 0)
                for prev_s in range(len(states))
            ])
    best_path = []
    best_last_state = np.argmax(viterbi_matrix[:, seq_length - 1])
    best_path.append(states[best_last_state])
    for t in range(seq_length - 1, 0, -1):
        best_last_state = backpointer_matrix[best_last_state][t]
        best_path.insert(0, states[best_last_state])
    return best_path, np.max(viterbi_matrix[:, seq_length - 1])

# Build HMM profile from aligned datasetA
transition_probs, emission_probs = build_hmm_profile(aligned_datasetA)
# Compute alignment scores for datasetB using the HMM
start_prob = {state: 1.0 / len(transition_probs) for state in transition_probs}
alignment_scores = []
alignment_paths = []
for sequence in datasetB:
    path, score = viterbi(sequence, transition_probs, emission_probs, start_prob)
    alignment_scores.append(score)
    alignment_paths.append(path)
print("Alignment Scores for datasetB:", alignment_scores)
print("Alignment Paths for datasetB:", alignment_paths)
