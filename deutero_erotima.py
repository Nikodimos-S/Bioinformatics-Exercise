import numpy as np

def global_alignment(s1, s2, a):
    m, n = len(s1), len(s2)
    dp = np.zeros((m+1, n+1))
    
    # Initialize the dp table
    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] - a
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] - a

    # Fill the dp table
    for i in range(1, m+1):
        for j in range(1, n+1):
            match = dp[i-1][j-1] + (1 if s1[i-1] == s2[j-1] else -a/2)
            delete = dp[i-1][j] - a
            insert = dp[i][j-1] - a
            dp[i][j] = max(match, delete, insert)
    
    return dp

def align_sequences(s1, s2, dp, a):
    aligned_s1, aligned_s2 = "", ""
    i, j = len(s1), len(s2)
    while i > 0 and j > 0:
        if dp[i][j] == dp[i-1][j-1] + (1 if s1[i-1] == s2[j-1] else -a/2):
            aligned_s1 = s1[i-1] + aligned_s1
            aligned_s2 = s2[j-1] + aligned_s2
            i, j = i-1, j-1
        elif dp[i][j] == dp[i-1][j] - a:
            aligned_s1 = s1[i-1] + aligned_s1
            aligned_s2 = "-" + aligned_s2
            i = i-1
        else:
            aligned_s1 = "-" + aligned_s1
            aligned_s2 = s2[j-1] + aligned_s2
            j = j-1

    while i > 0:
        aligned_s1 = s1[i-1] + aligned_s1
        aligned_s2 = "-" + aligned_s2
        i = i-1
    while j > 0:
        aligned_s1 = "-" + aligned_s1
        aligned_s2 = s2[j-1] + aligned_s2
        j = j-1

    return aligned_s1, aligned_s2

def multiple_alignment(strings, a):
    aligned_strings = [strings[0]]

    for i in range(1, len(strings)):
        s1 = aligned_strings[-1]
        s2 = strings[i]
        dp = global_alignment(s1, s2, a)
        aligned_s1, aligned_s2 = align_sequences(s1, s2, dp, a)

        aligned_strings[-1] = aligned_s1
        aligned_strings.append(aligned_s2)

    # Ensure all aligned strings are of the same length
    max_len = max(len(s) for s in aligned_strings)
    for i in range(len(aligned_strings)):
        aligned_strings[i] = aligned_strings[i].ljust(max_len, '-')
    
    return aligned_strings

