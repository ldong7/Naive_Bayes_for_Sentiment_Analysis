# Litong "Leighton" Dong
# Naive Bayes Algorithm

import argparse
import re
import os
import collections
import numpy as np
import math
import random


# Stop word list
stopWords = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also',
             'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be',
             'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear',
             'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for',
             'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers',
             'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is',
             'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may',
             'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor',
             'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our',
             'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since',
             'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then',
             'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us',
             've', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
             'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet',
             'you', 'your']


def parseArgument():                                                # function that extract info from command line
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('-d', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def getFileContent(filename):
    input_file = open(filename, 'r')        # open file
    text = input_file.read()                # read file to a string
    input_file.close()                      # end reading
    return text                             # return string


def parseFile(cnt, text):
    text = re.findall(r'[^\W\d][\w\'-_]*(?<=\w)', text)  # strip all the symbols and collect all the words
    for word in text:                                    # use counter to count each word
        if word not in stopWords:                        # check if it is a stop word, if not
            cnt[word] += 1                               # add count
    return cnt                                           # return counter


def subFileProcess(directory, i):
    cnt_train = collections.Counter()                           # initialize counter
    a = random.sample(range(1, len(os.listdir(directory + '/' + i))), (len(os.listdir(directory+'/'+i))-1)/3)
                                                                # generate some random number to be used as index
    test_file = []                                              # initialize lists
    train_file = []

    for j in range(1, len(os.listdir(directory + '/' + i))):    # iterator
        k = os.listdir(directory + '/' + i)[j]                  # each file in subdirectory
        t = getFileContent(directory + '/' + i + '/' + k)       # get file content
        if j in a:                                              # if the index is one of the random
            test_file.append(t)                                 # store file into test files
        else:                                                   # if not
            train_file.append(t)                                # store into training files
            parseFile(cnt_train, t)                             # update counter

    return cnt_train, test_file, train_file                     # return counter, and file lists


def fileProcess(directory):
    for i in os.listdir(directory):                             # going through each subdirectory
        if i == 'neg':                                          # going into neg
            cnt_train_neg, test_file_neg, train_file_neg = subFileProcess(directory, i)     # call function
        elif i == 'pos':                                                                    # going into pos
            cnt_train_pos, test_file_pos, train_file_pos = subFileProcess(directory, i)
    return cnt_train_neg, test_file_neg, train_file_neg, cnt_train_pos, test_file_pos, train_file_pos


def testMatch(i, dict_prob_train_neg, dict_prob_train_pos, c_pos, c_neg,
              total_unique_words, prob_neg, prob_pos):
    cnt_test = collections.Counter()                                                # initialize counter
    parseFile(cnt_test, i)                                                          # update counter
    prob_neg_d = 0                                                                  # initialize probability
    prob_pos_d = 0
    for k in cnt_test:                                                              # for each word
        n = k in dict_prob_train_neg                                                # check if k in training set neg
        p = k in dict_prob_train_pos                                                # check if k in training set pos
        num = cnt_test[k]                                                           # number of the word k in the file

        if n & p:                                                                   # if it is known to both training
            prob_neg_d += math.log(dict_prob_train_neg[k]) * num                    # add probability accordingly
            prob_pos_d += math.log(dict_prob_train_pos[k]) * num
        elif n:                                                                     # if it is know to only neg
            prob_neg_d += math.log(dict_prob_train_neg[k]) * num                    # add probability accordingly
            prob_pos_d += math.log(1 / float(c_pos + total_unique_words + 1)) * num
        elif p:
            prob_pos_d += math.log(dict_prob_train_pos[k]) * num
            prob_neg_d += math.log(1 / float(c_neg + total_unique_words + 1)) * num
        else:
            prob_neg_d += math.log(1 / float(c_neg + total_unique_words + 1)) * num
            prob_pos_d += math.log(1 / float(c_pos + total_unique_words + 1)) * num

    prob_neg_d += math.log(prob_neg)                        # add probability with probability of the class
    prob_pos_d += math.log(prob_pos)
    return prob_pos_d, prob_neg_d                           # return probability for the file


def trainProb(cnt_train, c, total):
    dict_prob_train = dict(cnt_train)                       # transform counter into dictionary
    for i in dict_prob_train:                               # for each key
        dict_prob_train[i] = float(dict_prob_train[i] + 1) / (c + total + 1)  # change value from count to probability
    return dict_prob_train                                  # output dictionary with probability


def main():
    args = parseArgument()
    directory = args['d'][0]
    cnt_train_neg, test_file_neg, train_file_neg, cnt_train_pos, test_file_pos, train_file_pos = fileProcess(directory)
                                                             # call function to process file
    c_neg = sum(cnt_train_neg.values())                                     # counts of words in the training set in neg
    c_pos = sum(cnt_train_pos.values())                                     # counts of words in the training set in pos
    total_unique_words = len(cnt_train_neg + cnt_train_pos)                 # total number of unique words
    prob_pos = float(len(train_file_pos)) / (len(train_file_pos) + len(train_file_neg)) # probability of pos class
    prob_neg = float(len(train_file_neg)) / (len(train_file_pos) + len(train_file_neg)) # probability of neg class
    dict_prob_train_pos = trainProb(cnt_train_pos, c_pos, total_unique_words)   # call function to obtain probability
    dict_prob_train_neg = trainProb(cnt_train_neg, c_neg, total_unique_words)
    accuracy_count_neg = 0                              # initialize count
    for i in test_file_neg:                             # each file in test set in neg
        prob_pos_d, prob_neg_d = testMatch(i, dict_prob_train_neg, dict_prob_train_pos, c_pos, c_neg,
                                           total_unique_words, prob_neg, prob_pos)
                                                        # compute probability of the class given the file
        if prob_neg_d > prob_pos_d:                     # if probability of neg is larger
            accuracy_count_neg += 1                     # we have one correct case

    accuracy_count_pos = 0                              # initialize count
    for i in test_file_pos:                             # each file in test set in pos
        prob_pos_d, prob_neg_d = testMatch(i, dict_prob_train_neg, dict_prob_train_pos, c_pos, c_neg,
                                           total_unique_words, prob_neg, prob_pos)
                                                        # compute probability of the class given the file
        if prob_neg_d < prob_pos_d:                     # if probability of pos is larger
            accuracy_count_pos += 1                     # we have one correct case

    aa = len(test_file_pos)                             # outputs
    bb = len(train_file_pos)
    cc = accuracy_count_pos
    dd = len(test_file_neg)
    ee = len(train_file_neg)
    ff = accuracy_count_neg
    gg = ((accuracy_count_neg + accuracy_count_pos) * 100 / (len(test_file_pos) + len(test_file_neg)))  # accuracy
    return aa, bb, cc, dd, ee, ff, gg


sum_acc = 0                                              # initialize sum
N = 3                                                    # number of iteration

for i in range(1, N+1):
    print 'iteration %d:' % i                            # print title
    a, b, c, d, e, f, g = main()                         # call main function
    print 'num_pos_test_docs:%d' % a
    print 'num_pos_training_docs:%d' % b
    print 'num_pos_correct_docs:%d' % c
    print 'num_neg_test_docs:%d' % d
    print 'num_neg_training_docs:%d' % e
    print 'num_neg_correct_docs:%d' % f
    print 'accuracy:%d%%' % g
    sum_acc += g                                            # add sum of accuracy

print 'ave_accuracy:%.1f%%' % (float(sum_acc) / N)          # print avg accuracy