# Candidate Elimination Algorithm

import csv

def candidate_elimination(data):
    # Initialize S and G
    S = ['0'] * (len(data[0]) - 1)
    G = [['?'] * (len(data[0]) - 1)]

    for row in data:
        attributes = row[:-1]
        target = row[-1]

        if target == 'Yes':
            # Generalize S
            for i in range(len(S)):
                if S[i] == '0':
                    S[i] = attributes[i]
                elif S[i] != attributes[i]:
                    S[i] = '?'

            # Remove inconsistent hypotheses from G
            G = [g for g in G if all(
                g[i] == '?' or g[i] == attributes[i]
                for i in range(len(attributes))
            )]

        else:
            # Specialize G
            new_G = []

            for g in G:
                for i in range(len(attributes)):
                    if g[i] == '?':
                        if S[i] != attributes[i]:
                            new_g = g.copy()
                            new_g[i] = S[i]
                            new_G.append(new_g)

            G = new_G

    return S, G


# Training dataset
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

S, G = candidate_elimination(data)

print("Final Specific Boundary (S):")
print(S)

print("\nFinal General Boundary (G):")
for g in G:
    print(g)