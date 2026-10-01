import math
from collections import Counter

# Calculate Entropy
def entropy(data):
    labels = [row[-1] for row in data]
    count = Counter(labels)

    total = len(labels)
    ent = 0

    for value in count.values():
        probability = value / total
        ent -= probability * math.log2(probability)

    return ent


# Calculate Information Gain
def information_gain(data, attribute_index):
    total_entropy = entropy(data)

    values = set(row[attribute_index] for row in data)
    weighted_entropy = 0

    for value in values:
        subset = [
            row for row in data
            if row[attribute_index] == value
        ]

        weighted_entropy += (
            len(subset) / len(data)
        ) * entropy(subset)

    return total_entropy - weighted_entropy


# ID3 Algorithm
def id3(data, attributes):
    labels = [row[-1] for row in data]

    # If all examples have the same class
    if len(set(labels)) == 1:
        return labels[0]

    # If no attributes are left
    if not attributes:
        return Counter(labels).most_common(1)[0][0]

    # Find attribute with maximum information gain
    gains = [
        information_gain(data, i)
        for i in attributes
    ]

    best_attribute = attributes[gains.index(max(gains))]

    tree = {best_attribute: {}}

    values = set(row[best_attribute] for row in data)

    remaining_attributes = [
        i for i in attributes
        if i != best_attribute
    ]

    for value in values:
        subset = [
            row for row in data
            if row[best_attribute] == value
        ]

        tree[best_attribute][value] = id3(
            subset,
            remaining_attributes
        )

    return tree


# Dataset
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]


# Build decision tree
attributes = [0, 1, 2, 3]

tree = id3(data, attributes)

print("Decision Tree using ID3:")
print(tree)

print("\nInformation Gain:")

for i in attributes:
    print("Attribute", i, ":", information_gain(data, i))