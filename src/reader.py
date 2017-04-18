from __future__ import division
import sys
# append the upper dir into the system path
sys.path.append('../')
import numpy as np
import random, math
from preprocess import Comment, Danmuku


class Reader:

    def __init__(self, args = None):
        '''
        self.wid_tid2wtid: (word_id, topic_id) -> id of (word,topic)
        self.wtid2vec: the id of (word,topic) -> vector
        '''
        self.vector_size = args.vector_size
        self.n_part_danmuku = args.n_part_danmuku
        self.ratio_train = args.ratio_split[0]
        self.ratio_test = args.ratio_split[1]
        self.ratio_validate = 1 - self.ratio_test - self.ratio_train
        self.batch_size = args.batch_size
        self.modality = args.modality
        self.n_classes = args.n_classes
        self.videos = [] # [...,(video_i_id, label_i),...]
        with open('video_labeled.txt', 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                self.videos.append([int(line[0]), int(line[1])])
        random.shuffle(self.videos)
        print(self.videos)

        self.n_data = len(self.videos)
        # self.n_train = int(self.n_data * self.ratio_train)
        # self.n_test = int(self.n_data * self.ratio_test)
        self.n_train = 2
        self.n_test = 1
        self.n_validate = int(self.n_data - self.n_train - self.n_test)
        self.data_train = self.videos[:self.n_train]
        self.data_validate = self.videos[self.n_train:self.n_train + self.n_validate]
        self.data_test = self.videos[self.n_test*(-1):]

        self.x_train = [x[0] for x in self.data_train]
        self.y_train = [x[1] for x in self.data_train]
        self.x_test = [x[0] for x in self.data_test]
        self.y_test = [x[1] for x in self.data_test]
        self.x_validate = [x[0] for x in self.data_validate]
        self.y_validate = [x[1] for x in self.data_validate]


        self.wid_tid2wtid, self.wtid2vec = self.word2vec_read()
        self.id2doc = self.id2doc_read() # the id of doc -> [...,word_id:topic_id,...]




    def word2vec_read(self):
        'read the log.txt from topic_word_embeding'
        with open('word2vec/log.txt', 'r') as f:
            wid_tid2wtid = {}
            for line in f:
                line = line.strip().split('\t')
                wid_tid2wtid[(line[1], line[2])] = line[3]

        'read the vectors.txt from topic_word_embeding'
        with open('word2vec/vectors.txt') as f:
            wtid2vec = {}
            for line in f:
                line = line.strip().split(' ')
                wtid2vec[line[0]] = [float(i) for i in line[1:]]
                if self.vector_size == -1:
                    self.vector_size = len(line[1:])

        return wid_tid2wtid, wtid2vec

    def id2doc_read(self):
        id2doc = {}
        with open('doc2id.txt', 'r') as fa, open('topic_emotion_assign.txt', 'r') as fb:
            for la, lb in zip(fa, fb):
                la = la.strip().split('\t')
                lb = lb.strip().split(' ')
                id2doc[(str(la[0]), str(la[1]))] = lb
        return id2doc

    def vid2docs(self, vid):
        docs = []
        for i in range(self.n_part_danmuku + 1):
            docs.append(self.id2doc[(str(vid), str(i))])
        return docs

    def tf_idf_calc(self, w, did, vids):
        '''
        wid: the id of words
        tid: the id of topic
        did: the id of doc (video_id, style_id)
        vids: all video_id in this step(train, test, validata)
        '''
        doc = self.id2doc[did]
        # print('Doc', did)
        # print(doc)
        tf = 0
        for w_ in doc:
            if w == w_:
                tf += 1
        tf /= len(doc)
        idf = 1
        for vid in vids:
            docs = self.vid2docs(vid)
            for doc in docs:
                if str(w) in doc:
                    idf += 1
        idf = math.log(int(len(docs)) / idf)
        return tf * idf

    def get_comment_embedding(self, index, data):
        docs = []
        # print('wid_tid2wtid', len(self.wid_tid2wtid.keys()) )
        # print(self.wid_tid2wtid.keys())
        # print(('0', '113') in self.wid_tid2wtid)
        for idx in index:
            vid = data[idx]
            # print('DID', vid)
            cmt = self.id2doc[(str(vid), str(0))]
            doc_embedding = [0.0 for i in range(self.vector_size)]
            for w in cmt:
                wid, tid = w.split(':')
                wtid = self.wid_tid2wtid[(wid, tid)]
                if wtid in self.wtid2vec:
                    tf_idf = self.tf_idf_calc(w, (str(vid), str(0)), data)
                    for i in range(len(doc_embedding)):
                        doc_embedding[i] += tf_idf * self.wtid2vec[wtid][i]
            docs.append(doc_embedding)
        return docs


    def get_danmuku_embedding(self, index, data):
        docs = []
        for idx in index:
            vid = data[idx]
            dans = []
            for i in range(self.n_part_danmuku):
                dan = self.id2doc[(str(vid), str(i+1))]
                doc_embedding = [0.0 for i in range(self.vector_size)]
                for w in dan:
                    wid, tid = w.split(':')
                    wtid = self.wid_tid2wtid[(wid, tid)]
                    if wtid in self.wtid2vec:
                        for i in range(len(doc_embedding)):
                            tf_idf = self.tf_idf_calc(w, (str(vid), str(i)), data)
                            doc_embedding[i] += tf_idf * self.wtid2vec[wtid][i]
                dans.append(doc_embedding)
            docs.append(dans)
        return docs

    def next_batch(self):
        """
        @brief return a batch of train and target data
        @return comment_batch_data: [batch_size, n_input]
        @return danmuku_batch_data:  [batch_size, n_part_danmuku, n_input]
        @return target_data_batch: [batch_size, 1]
        """
        index = np.random.choice(np.arange(self.n_train), self.batch_size, replace=False)
        batch = []
        if self.modality[0] == 1:
            comment_batch_data = self.get_comment_embedding(index, self.x_train)
            batch.append(np.array(comment_batch_data))
        if self.modality[1] == 1:
            danmuku_batch_data = self.get_danmuku_embedding(index, self.x_train)
            batch.append(np.array(danmuku_batch_data))
        y_train = np.zeros((self.batch_size, self.n_classes), dtype = float)
        for i, v in enumerate(index):
            y_train[i, self.y_train[v]] = 1
        # y_train = np.reshape(np.array(self.y_train)[index], (-1, 1))
        batch.append(y_train)

        return batch

    def get_test_data(self):
        index = np.arange(self.n_test)
        # print('Index', index, self.n_test)
        batch = []
        if self.modality[0] == 1:
            comment_batch_data = self.get_comment_embedding(index, self.x_test)
            batch.append(np.array(comment_batch_data))
        if self.modality[1] == 1:
            danmuku_batch_data = self.get_danmuku_embedding(index, self.x_test)
            batch.append(np.array(danmuku_batch_data))

        y_test = np.zeros((self.n_test, self.n_classes), dtype = float)
        for i, v in enumerate(index):
            y_test[i, self.y_train[v]] = 1
        # y_test = np.reshape(np.array(self.y_train)[index], (-1, 1))
        batch.append(y_test)
        return batch
    # def get_train_set(self, start=None, length=None):
    #     """
    #     @brief return the total dataset
    #     """
    #     if start is None:
    #         start = 0
    #     if length is None or start + length > self.train_num:
    #         length = self.train_num - start
    #     if length <= 0:
    #         return []
    #
    #     train_set = []
    #     if self.modality[0] == 1:
    #         train_set.append((self.ir_train_data[start:start+length] - self.ir_mean) / self.ir_std)
    #     if self.modality[1] == 1:
    #         train_set.append((self.mete_train_data[start:start+length] - self.mete_mean) / self.mete_std)
    #     if self.modality[2] == 1:
    #         train_set.append(self.path2image(self.sky_cam_train_data[start:start+length]))
    #     train_set.append(self.train_hour_index[start:start+length])
    #     train_set.append(self.target_train_data[start:start+length])
    #
    #     return train_set
    #
    # #The returned validataion and test set:
    # #ir_data and mete_data: [batch_size, n_step, n_input], batch_size = validation_num/test_num
    # #target_data: [batch_size, n_model], each of the target_data contains all model target in a tesor
    # def get_validation_set(self, start=None, length=None):
    #     """
    #     @brief return the total validation dataset
    #     """
    #     if start is None:
    #         start = 0
    #     if length is None or start + length > self.validation_num:
    #         length = self.validation_num - start
    #     if length <= 0:
    #         return []
    #
    #     validation_set = []
    #     if self.modality[0] == 1:
    #         validation_set.append((self.ir_validation_data[start:start+length] - self.ir_mean) / self.ir_std)
    #     if self.modality[1] == 1:
    #         validation_set.append((self.mete_validation_data[start:start+length] - self.mete_mean) / self.mete_std)
    #     if self.modality[2] == 1:
    #         validation_set.append(self.path2image(self.sky_cam_validation_data[start:start+length]))
    #     validation_set.append(self.validation_hour_index[start:start+length])
    #     validation_set.append(self.target_validation_data[start:start+length])
    #
    #     return validation_set
    #
    # def get_test_set(self, start=None, length=None):
    #     """
    #     @brief return a test set in the specific test num
    #     """
    #     if start is None:
    #         start = 0
    #     if length is None or start + length > self.test_num:
    #         length = self.test_num - start
    #     if length <=0:
    #         return []
    #
    #     test_set = []
    #     if self.modality[0] == 1:
    #         test_set.append((self.ir_test_data[start:start+length] - self.ir_mean) / self.ir_std)
    #     if self.modality[1] == 1:
    #         test_set.append((self.mete_test_data[start:start+length] - self.mete_mean) / self.mete_std)
    #     if self.modality[2] == 1:
    #         test_set.append(self.path2image(self.sky_cam_test_data[start:start+length]))
    #     test_set.append(self.test_hour_index[start:start+length])
    #     test_set.append(self.target_test_data[start:start+length])
    #
    #     return test_set

if __name__ == '__main__':
    reader = Reader()
    data = reader.next_batch()
    print(data[0].shape, data[1].shape, data[2].shape)
