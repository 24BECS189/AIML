# Naive Bayes Classification

from collections import Counter

# Training dataset
data = [
    ['Sunny', 'Hot', 'High', 'No'],
    ['Sunny', 'Hot', 'High', 'No'],
    ['Overcast', 'Hot', 'High', 'Yes'],
    ['Rain', 'Mild', 'High', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Yes'],
    ['Sunny', 'Mild', 'High', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Yes'],
    ['Rain', 'Mild', 'High', 'No']
]


# Naive Bayes classifier
def naive_bayes(data, test):
    total = len(data)

    # Find classes
    classes = set(row[-1] for row in data)

    probabilities = {}

    for c in classes:

        # Rows belonging to this class
        class_rows = [
            row for row in data
            if row[-1] == c
        ]

        # Prior probability
        probability = len(class_rows) / total

        # Conditional probabilities
        for i in range(len(test)):
            count = sum(
                row[i] == test[i]
                for row in class_rows
            )

            probability *= count / len(class_rows)

        probabilities[c] = probability

    # Select class with highest probability
    prediction = max(
        probabilities,
        key=probabilities.get
    )

    return prediction, probabilities


# Test data
test = ['Sunny', 'Cool', 'High']

prediction, probabilities = naive_bayes(
    data, test
)


# Display output
print("Naive Bayes Classification")
print("---------------------------")

print("Test Data:", test)

print("\nClass Probabilities:")

for c, probability in probabilities.items():
    print(c, ":", probability)

print("\nPredicted Class:", prediction)