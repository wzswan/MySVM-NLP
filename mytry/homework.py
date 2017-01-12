# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 17:12:14 2017

@author: wzswan
"""

#!/usr/bin/env python
import numpy as np

from nlpsvm import parse_nlp, bag_set
import mysvm


def main():
    # Load list of NLP Examples
    train_set = parse_nlp('train')
    test_set = parse_nlp('test')

    # Group examples into bags
    bagtrain = bag_set(train_set)
    bagtest = bag_set(test_set)

    # Convert bags to NumPy arrays
    # (The ...[1:] removes first  column,
    #  which is the bag/instance ids )
    bags1 = [np.array(b.to_float())[1:] for b in bagtrain]
    labels1 = np.array([b.label for b in bagtrain], dtype=float)
    bags2 = [np.array(b.to_float())[1:] for b in bagtest]
    labels2 = np.array([b.label for b in bagtest], dtype=float)
    
    
    # Convert 0/1 labels to -1/1 labels
    labels1 = 2 * labels1 - 1
    labels2 = 2 * labels2 - 1

    # Spilt dataset arbitrarily to train/test sets
    train_bags = bags1[]
    train_labels = labels1[]
    test_bags = bags2[]
    test_labels = labels2[]
   

    # Construct classifiers
    classifiers = {}
    classifiers['SIL'] = mysvm.SIL(kernel='linear', C=1.0)

    # Train/Evaluate classifiers
    accuracies = {}
    for algorithm, classifier in classifiers.items():
        classifier.fit(train_bags, train_labels)
        predictions = classifier.predict(test_bags)
        accuracies[algorithm] = np.average(test_labels == np.sign(predictions))

    for algorithm, accuracy in accuracies.items():
        print '\n%s Accuracy: %.1f%%' % (algorithm, 100 * accuracy)


if __name__ == '__main__':
    main()