# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils

def main():
    accuracy = 0.0
    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    predictions = []
    for line in open("birth_dev.tsv", encoding='utf-8'):
        predictions.append('London')
    total, correct = utils.evaluate_places("birth_dev.tsv", predictions)
    
    if total > 0:
        print(f'Correct: {correct} out of {total}: {correct/total*100}%')
    else:
        print(f'Predictions written to london_baseline_accuracy.txt; no targets provided')
    accuracy = correct/total
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
