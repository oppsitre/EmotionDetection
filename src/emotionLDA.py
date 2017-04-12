"""
Implementation of the collapsed Gibbs sampler for Sentiment-LDA, described in
Sentiment Analysis with Global Topics and Local Dependency (Li, Huang and Zhu)
"""

import numpy as np
import re
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk import word_tokenize,sent_tokenize, pos_tag
from preprocess import *


MAX_VOCAB_SIZE = 50000


def sampleFromDirichlet(alpha):
    """
    Sample from a Dirichlet distribution
    alpha: Dirichlet distribution parameter (of length d)
    Returns:
    x: Vector (of length d) sampled from dirichlet distribution

    """
    return np.random.dirichlet(alpha)


def sampleFromCategorical(theta):
    """
    Samples from a categorical/multinoulli distribution
    theta: parameter (of length d)
    Returns:
    x: index ind (0 <= ind < d) based on probabilities in theta
    """
    theta = theta/np.sum(theta)
    return np.random.multinomial(1, theta).argmax()


def word_indices(wordOccuranceVec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in wordOccuranceVec.nonzero()[0]:
        for i in range(int(wordOccuranceVec[idx])):
            yield idx


class SentimentLDAGibbsSampler:

    def __init__(self, numTopics, alpha, beta, gamma, numSentiments=22):
        """
        numTopics: Number of topics in the model
        numSentiments: Number of sentiments (default 2)
        alpha: Hyperparameter for Dirichlet prior on topic distribution
        per document
        beta: Hyperparameter for Dirichlet prior on vocabulary distribution
        per (topic, sentiment) pair
        gamma:Hyperparameter for Dirichlet prior on sentiment distribution
        per (document, topic) pair
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.numTopics = numTopics
        self.numSentiments = numSentiments
        self.word2id = None
        self.id2word = None
        self.vocabSize = 0
        self.numDocs = 0

    def processSingleReview(self, review, d=None):
        """
        Convert a raw review to a string of words, including Tokensization and Filtering stopwords
        """
        expressions = expressions_read()
        stops = stopwords_read()
        words = sentence_process(expression_segment(review, expressions), stops)
        # print(words)
        return(words)

    def word_record(self, docs):
        word2id = {}
        id2word = {}
        i = 0
        for doc in docs:
            for w in doc:
                if w not in word2id.keys():
                    word2id[w] = i
                    id2word[i] = w
                    i += 1
        self.vocabSize = i
        return word2id, id2word

    def doc2mat(self, docs):
        '''
        change the words of documets into numbers.
        '''
        self.word2id, self.id2word = self.word_record(docs)
        matrix = []
        for i, doc in enumerate(docs):
            tmp = []
            for w in doc:
                tmp.append(self.word2id[w])
            matrix.append(tmp)
        return matrix
    def processReviews(self, reviews, saveAs=None, saveOverride=False):
        processed_reviews = []
        i = 0
        self.numDocs = len(reviews)
        # self.numDocs = 10
        for review in reviews:
            if((i + 1) % 1000 == 0):
                print("Review %d of %d" % (i + 1, len(reviews)))
            processed_reviews.append(self.processSingleReview(review, i))
            i += 1
        # print(processed_reviews)
        wordOccurenceMatrix = self.doc2mat(processed_reviews)
        return wordOccurenceMatrix

    def _initialize_(self, reviews, saveAs=None, saveOverride=False):
        """
        wordOccuranceMatrix: numDocs x vocabSize matrix encoding the
        bag of words representation of each document
        """
        self.wordmatrix = self.processReviews(reviews, saveAs, saveOverride)
        # numDocs, vocabSize = self.wordOccuranceMatrix.shape

        # Pseudocounts
        self.n_dt = np.zeros((self.numDocs, self.numTopics))
        self.n_dts = np.zeros((self.numDocs, self.numTopics, self.numSentiments))
        self.n_d = np.zeros((self.numDocs))
        self.n_vts = np.zeros((self.vocabSize, self.numTopics, self.numSentiments))
        self.n_ts = np.zeros((self.numTopics, self.numSentiments))
        self.topics = {}
        self.sentiments = {}
        self.priorSentiment = {}

        alphaVec = self.alpha * np.ones(self.numTopics)
        gammaVec = self.gamma * np.ones(self.numSentiments)
        emotion_dict = emotion_dict_read()
        for i in range(self.vocabSize):
            w = self.id2word[i]
            if w in emotion_dict.keys():
                self.priorSentiment[i] = emotion_dict[self.id2word[i]][0]
            else:
                self.priorSentiment[i] = 21

        for d in range(self.numDocs):
            topicDistribution = sampleFromDirichlet(alphaVec)
            sentimentDistribution = np.zeros(
                (self.numTopics, self.numSentiments))
            for t in range(self.numTopics):
                sentimentDistribution[t, :] = sampleFromDirichlet(gammaVec)
            for i, w in enumerate(self.wordmatrix[d]):
                t = sampleFromCategorical(topicDistribution)
                s = sampleFromCategorical(sentimentDistribution[t, :])
                self.topics[(d, i)] = t
                self.sentiments[(d, i)] = s
                self.n_dt[d, t] += 1
                self.n_dts[d, t, s] += 1
                self.n_d[d] += 1
                self.n_vts[w, t, s] += 1
                self.n_ts[t, s] += 1

    def conditionalDistribution(self, d, v):
        """
        Calculates the (topic, sentiment) probability for word v in document d
        Returns:    a matrix (numTopics x numSentiments) storing the probabilities
        """
        probabilities_ts = np.ones((self.numTopics, self.numSentiments))
        firstFactor = (self.n_dt[d] + self.alpha) / \
            (self.n_d[d] + self.numTopics * self.alpha)
        secondFactor = (self.n_dts[d, :, :] + self.gamma) / \
            (self.n_dt[d, :] + self.numSentiments * self.gamma)[:, np.newaxis]
        thirdFactor = (self.n_vts[v, :, :] + self.beta) / \
            (self.n_ts + self.n_vts.shape[0] * self.beta)
        probabilities_ts *= firstFactor[:, np.newaxis]
        probabilities_ts *= secondFactor * thirdFactor
        probabilities_ts /= np.sum(probabilities_ts)
        return probabilities_ts

    def getTopKWordsByLikelihood(self, K):
        """
        Returns top K discriminative words for topic t and sentiment s
        ie words v for which p(t, s | v) is maximum
        """
        pseudocounts = np.copy(self.n_vts)
        normalizer = np.sum(pseudocounts, (1, 2))
        pseudocounts /= normalizer[:, np.newaxis, np.newaxis]
        for t in range(self.numTopics):
            for s in range(self.numSentiments):
                topWordIndices = pseudocounts[:, t, s].argsort()[-1:-(K + 1):-1]
                vocab = self.vectorizer.get_feature_names()
                print(t, s, [vocab[i] for i in topWordIndices])

    def getTopKWords(self, K):
        """
        Returns top K discriminative words for topic t and sentiment s
        ie words v for which p(v | t, s) is maximum
        """
        pseudocounts = np.copy(self.n_vts)
        normalizer = np.sum(pseudocounts, (0))
        pseudocounts /= normalizer[np.newaxis, :, :]
        for t in range(self.numTopics):
            for s in range(self.numSentiments):
                topWordIndices = pseudocounts[:, t, s].argsort()[-1:-(K + 1):-1]
                # vocab = self.vectorizer.get_feature_names()
                print(t, s, [self.id2word[i] for i in topWordIndices])


    def run(self, reviews, maxIters=30, saveAs=None, saveOverride=False):
        """
        Runs Gibbs sampler for sentiment-LDA
        """
        self._initialize_(reviews, saveAs, saveOverride)
        # numDocs, vocabSize = self.wordOccuranceMatrix.shape
        for iteration in range(maxIters):
            print("Starting iteration %d of %d" % (iteration + 1, maxIters))
            for d in range(self.numDocs):
                for i, v in enumerate(self.wordmatrix[d]):
                    t = self.topics[(d, i)]
                    s = self.sentiments[(d, i)]
                    self.n_dt[d, t] -= 1
                    self.n_d[d] -= 1
                    self.n_dts[d, t, s] -= 1
                    self.n_vts[v, t, s] -= 1
                    self.n_ts[t, s] -= 1

                    probabilities_ts = self.conditionalDistribution(d, v)
                    if v in self.priorSentiment:
                        s = self.priorSentiment[v]
                        t = sampleFromCategorical(probabilities_ts[:, s])
                    else:
                        ind = sampleFromCategorical(probabilities_ts.flatten())
                        t, s = np.unravel_index(ind, probabilities_ts.shape)

                    self.topics[(d, i)] = t
                    self.sentiments[(d, i)] = s
                    self.n_dt[d, t] += 1
                    self.n_d[d] += 1
                    self.n_dts[d, t, s] += 1
                    self.n_vts[v, t, s] += 1
                    self.n_ts[t, s] += 1

    def get_wordmap(self):
        with open('wordmap.txt', 'w') as f:
            f.write(str(self.vocabSize) + '\n')
            for w in self.word2id.keys():
                f.write(w + ' ' + str(self.word2id[w]) + '\n')


    def get_tassign(self):
        with open('topic_emotion_assign.txt', 'w') as f:
            for d in range(self.numDocs):
                doc_str = ''
                for i, v in enumerate(self.wordmatrix[d]):
                    t = self.topics[(d, i)]
                    s = self.sentiments[(d, i)]
                    c = t * (self.numSentiments + 1) + s
                    doc_str += str(v) + ':' + str(c) + ' '
                f.write(doc_str[:-1] + '\n')
