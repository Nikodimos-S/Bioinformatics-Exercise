import random

# Patterns
patterns = ["AATTGA", "CGCTTAT", "GGACTCAT", "TTATTCGTA"]
alphabet = "ACGT"

#Function for creating a pattern using the alphabet
def modify_pattern(pattern):
    pattern_list = list(pattern)
    indices = random.sample(range(len(pattern)), 2)
    for index in indices:
        if random.random() < 0.5:
            pattern_list[index] = random.choice(alphabet)
        else:
            pattern_list[index] = ""
    return ''.join(pattern_list)

#Fucntion for making the string
def synthesize_string():
    prefix = ''.join(random.choices(alphabet, k=random.randint(1, 3)))
    synthesized_string = prefix
    for pattern in patterns:
        modified_pattern = modify_pattern(pattern)
        synthesized_string += modified_pattern
    suffix = ''.join(random.choices(alphabet, k=random.randint(1, 2)))
    synthesized_string += suffix
    return synthesized_string
