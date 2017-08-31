#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
import tarfile
import random
import json
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import numpy as np
from PIL import Image

'''
    下载数据
'''
class Download:
    URL = 'https://commondatastorage.googleapis.com/books1000/'
    Last_Percent_Reported = None
    DATA_ROOT = 'data'
    NUM_CLASSES = 10

    LABEL_DICT = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 8,
        'J': 9,
    }


    def __init__(self):
        pass


    @staticmethod
    def __downloadProgressHook(count, blockSize, totalSize):
        """
            A hook to report the progress of a download. This is mostly intended for users with
            slow internet connections. Reports every 5% change in download progress.
        """
        percent = int(count * blockSize * 100 / totalSize)

        if Download.Last_Percent_Reported != percent:
            if percent % 5 == 0:
                sys.stdout.write("%s%%" % percent)
                sys.stdout.flush()
            else:
                sys.stdout.write(".")
                sys.stdout.flush()

            Download.Last_Percent_Reported = percent


    @staticmethod
    def __maybeDownload(filename, expected_bytes, force=False):
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.isdir(Download.DATA_ROOT):
            os.mkdir(Download.DATA_ROOT)
        dest_filename = os.path.join(Download.DATA_ROOT, filename)
        if force or not os.path.exists(dest_filename):
            print('Attempting to download:', filename)
            filename, _ = urlretrieve(Download.URL + filename, dest_filename, reporthook=Download.__downloadProgressHook)
            print('\nDownload Complete!')
        statinfo = os.stat(dest_filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', dest_filename)
        else:
            raise Exception(
                'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
        return dest_filename


    @staticmethod
    def __maybeExtract(filename, force=False):
        root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
        if os.path.isdir(root) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping extraction of %s.' % (root, filename))
        else:
            print('Extracting data for %s. This may take a while. Please wait.' % root)
            tar = tarfile.open(filename)
            sys.stdout.flush()
            tar.extractall(Download.DATA_ROOT)
            tar.close()
        data_folders = [
            os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))]
        if len(data_folders) != Download.NUM_CLASSES:
            raise Exception(
                'Expected %d folders, one per class. Found %d instead.' % (
                    Download.NUM_CLASSES, len(data_folders)))
        print(data_folders)
        return data_folders


    @staticmethod
    def __genFilesList(folders, file_list_name):
        file_list_path = os.path.join(Download.DATA_ROOT, file_list_name)

        if not os.path.exists(file_list_path):
            print 'generating data_list_file ' + file_list_path + ' ... '

            data_set = []
            for dir_path in folders:
                data_set += [os.path.abspath(os.path.join(dir_path, file_name))
                             for file_name in os.listdir(dir_path)
                             if file_name != '.' and file_name != '..']

            random.shuffle(data_set)
            with open(file_list_path, 'wb') as f:
                f.write(json.dumps(data_set))

            print 'finishing generating'

        else:
            print 'already has data_list_file ' + file_list_path

            with open(file_list_path, 'rb') as f:
                data_set = json.loads(f.read())

        return data_set


    @staticmethod
    def __loadAndSaveData(data_set):
        size = len(data_set)
        divide_part = 4
        size_per_part = size // divide_part

        cur_index = 0
        for i in range(divide_part):
            file_path = os.path.join(Download.DATA_ROOT, 'data%d.pickle' % i)
            if os.path.exists(file_path):
                print '%s exist' % file_path
                continue

            if i == divide_part - 1:
                tmp_data_set = data_set[cur_index:]
            else:
                tmp_data_set = data_set[cur_index: cur_index + size_per_part]
                cur_index += size_per_part

            print 'loading %d part of data ...' % i
            X, y = Download.__getData(tmp_data_set)
            print 'finish loading'

            print 'saving %s ... ' % file_path
            with open(file_path, 'wb') as f:
                pickle.dump([X, y], f, pickle.HIGHEST_PROTOCOL)
            print 'finish saving'


    @staticmethod
    def __getData(data_list):
        X = []
        y = []

        for i, path in enumerate(data_list):
            try:
                label = Download.__convert(path.split('/')[-2])
                image = ndimage.imread(path).astype(np.float32)
                image = ((image - 255.0 / 2) / 255.0).reshape(image.shape[0] * image.shape[1])

                X.append(image)
                y.append(label)

            except Exception, ex:
                continue

        return np.array(X), np.array(y)


    @staticmethod
    def __convert(label):
        if label not in Download.LABEL_DICT:
            return
        label = Download.LABEL_DICT[label]

        zeros = np.zeros(10, dtype=np.float32)
        zeros[label] = 1
        return zeros


    @staticmethod
    def run():
        exist_pickle = True
        for i in range(4):
            data_path = os.path.join(Download.DATA_ROOT, 'data%d.pickle' % i)
            if not os.path.isfile(data_path):
                exist_pickle = False
                break

        if exist_pickle:
            print 'data pickles exist'
            return

        print '****************************'
        train_filename = Download.__maybeDownload('notMNIST_large.tar.gz', 247336696)
        print ''
        test_filename = Download.__maybeDownload('notMNIST_small.tar.gz', 8458043)
        print ''

        print '****************************'
        train_folders = Download.__maybeExtract(train_filename)
        test_folders = Download.__maybeExtract(test_filename)
        print ''

        print train_folders
        print test_folders
        print ''

        print '**********************'
        folders = train_folders + test_folders
        data_set = Download.__genFilesList(folders, 'data_list.txt')
        print ''

        print '**********************'
        Download.__loadAndSaveData(data_set)


class Data:
    PICKLE_SIZE = {
        '0': 136957,
        '1': 136961,
        '2': 136958,
        '3': 136962,
    }

    TOTAL_SIZE = 547838


    def __init__(self, start_ratio = 0.0, end_ratio = 1.0):
        start_ratio = min(max(0.0, start_ratio), 1.0)
        end_ratio = min(max(0.0, end_ratio), 1.0)

        start_index = int(self.TOTAL_SIZE * start_ratio)
        end_index = int(self.TOTAL_SIZE * end_ratio)
        init_index = 0

        self.__size = end_index - start_index
        self.__curFileNo = 0
        self.__minFileNo = 0
        self.__maxFileNo = 3

        self.__minIndexOfFirst = 0
        self.__maxIndexOfLast = -1

        for i in range(4):
            pickle_size = self.PICKLE_SIZE[str(i)]

            if init_index <= start_index < init_index + pickle_size:
                self.__minFileNo = i
                self.__curFileNo = i
                self.__minIndexOfFirst = start_index - init_index

            if init_index <= end_index < init_index + pickle_size:
                self.__maxFileNo = i
                self.__maxIndexOfLast = end_index - init_index
                break

            init_index += pickle_size

        self.__readFile()


    def __readFile(self):
        file_path = os.path.join(Download.DATA_ROOT, 'data%d.pickle' % self.__curFileNo)
        with open(file_path, 'rb') as f:
            self.__X, self.__y = pickle.load(f)

        if self.__curFileNo == self.__minFileNo:
            self.__X = self.__X[self.__minIndexOfFirst:, ]
            self.__y = self.__y[self.__minIndexOfFirst:, ]
        elif self.__curFileNo == self.__maxFileNo:
            self.__X = self.__X[:self.__maxIndexOfLast, ]
            self.__y = self.__y[:self.__maxIndexOfLast, ]

        self.__curIndex = 0
        self.__maxIndex = self.__X.shape[0]


    def getSize(self):
        return self.__size


    def show(self):
        print '***************************'
        print self.__size
        print self.__curIndex
        print self.__maxIndex
        print self.__curFileNo
        print self.__minFileNo
        print self.__maxFileNo
        print self.__minIndexOfFirst
        print self.__maxIndexOfLast
        print ''


    def nextBatch(self, batch_size, loop = True):
        if not loop and self.__curIndex >= self.__maxIndex and self.__curFileNo >= self.__maxFileNo:
            return None, None

        start_index = self.__curIndex
        end_index = self.__curIndex + batch_size
        left_num = 0

        if end_index > self.__maxIndex:
            left_num = end_index - self.__maxIndex
            end_index = self.__maxIndex

        X = self.__X[start_index: end_index, :]
        y = self.__y[start_index: end_index, :]

        if not left_num:
            self.__curIndex = end_index
            if self.__curFileNo == self.__maxFileNo:
                self.__curFileNo = self.__minFileNo
                self.__readFile()
            return X, y

        self.__curFileNo += 1
        if self.__curFileNo > self.__maxFileNo:
            if not loop:
                self.__curIndex = self.__maxIndex
            else:
                self.__curFileNo = self.__minFileNo
                self.__readFile()
            return X, y

        else:
            self.__readFile()

            while left_num:
                end_index = left_num
                if end_index > self.__maxIndex:
                    left_num = end_index - self.__maxIndex
                    end_index = self.__maxIndex
                else:
                    left_num = 0

                X = np.array(list(X) + list(self.__X[0: end_index, :]))
                y = np.array(list(y) + list(self.__y[0: end_index, :]))

                if not left_num:
                    self.__curIndex = end_index
                    if self.__curFileNo == self.__maxFileNo:
                        self.__curFileNo = self.__minFileNo
                        self.__readFile()
                    return X, y

                self.__curFileNo += 1
                if self.__curFileNo > self.__maxFileNo:
                    if not loop:
                        self.__curIndex = self.__maxIndex
                    else:
                        self.__curFileNo = self.__minFileNo
                        self.__readFile()
                    return X, y

                else:
                    self.__readFile()

'''
    数据对象
'''
class DataOld:
    LABEL_DICT = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 8,
        'J': 9,
    }


    def __init__(self, start_ratio = 0.0, end_ratio = 1.0):
        self.__getFileList()

        self.__startIndex = int(start_ratio * self.__len)
        self.__endIndex = int(end_ratio * self.__len)
        self.__curIndex = self.__startIndex
        self.__size = self.__endIndex - self.__startIndex

        self.__reachEnd = False


    def __getFileList(self):
        file_list_name = 'data_list.txt'
        file_list_path = os.path.join(os.path.join(os.path.split(__file__)[0], 'data'), file_list_name)

        with open(file_list_path, 'rb') as f:
            self.__fileList = json.loads(f.read())
            self.__len = len(self.__fileList)


    def nextBatch(self, batch_size):
        if self.__curIndex >= self.__endIndex:
            return None, None

        start_index = self.__curIndex
        end_index = self.__curIndex + batch_size
        if end_index > self.__endIndex:
            end_index = self.__endIndex
            self.__curIndex = self.__startIndex
            self.__reachEnd = True
        else:
            self.__curIndex = end_index
            self.__reachEnd = False

        data_list = self.__fileList[start_index: end_index]
        return self.__getData(data_list)


    def getSize(self):
        return self.__size


    def hasReachEnd(self):
        return self.__reachEnd


    @staticmethod
    def __getData(data_list):
        X = []
        y = []

        for i, path in enumerate(data_list):
            try:
                label = DataOld.__convert(path.split('/')[-2])
                image = ndimage.imread(path).astype(np.float32)
                image = ((image - 255.0 / 2) / 255.0).reshape(image.shape[0] * image.shape[1])

                X.append(image)
                y.append(label)

            except Exception, ex:
                continue

        return np.array(X), np.array(y)


    @staticmethod
    def __convert(label):
        if label not in DataOld.LABEL_DICT:
            return
        label = DataOld.LABEL_DICT[label]

        zeros = np.zeros(10, dtype=np.float32)
        zeros[label] = 1
        return zeros


    @staticmethod
    def showImage(matrix):
        im = Image.fromarray(np.array(matrix).astype(np.int8))
        im.show()


# Download.run()

# train_data = Data(0.6, 0.8)
# batch_x , batch_y = train_data.nextBatch(300)
