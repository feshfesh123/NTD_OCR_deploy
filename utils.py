from tensorflow.keras import backend
import tensorflow as tf
from net_config import ArchitectureConfig, FilePaths
import os
import numpy as np
import cv2
import time
import datetime
from data_preprocess import remove_noise_and_smooth
def labels_to_text(letters, labels):
    return ''.join(list(map(lambda x: letters[x] if x < len(letters) else "", labels)))  # noqa


def text_to_labels(letters, text):
    return list(map(lambda x: letters.index(x), text))

def decode_predict_ctc(out, chars = ArchitectureConfig.CHARS, top_paths=1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(top_paths):
        lables = backend.get_value(
            backend.ctc_decode(
                out, input_length=np.ones(out.shape[0]) * out.shape[1],
                greedy=False, beam_width=beam_width, top_paths=top_paths
            )[0][i]
        )[0]
        text = labels_to_text(chars, lables)
        results.append(text)
    return results

def decode_word_beamsearch(out):
    word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')
    f = open(FilePaths.fnCorpus, "r")
    ArchitectureConfig.CORPUS = f.read()
    labels = word_beam_search_module.word_beam_search(out, 5, 'Words', 0.1, ArchitectureConfig.CORPUS, ArchitectureConfig.CHARS, ArchitectureConfig.WORD_CHARS)
    result = labels_to_text(ArchitectureConfig.chars, labels)

    return result

class Sample:
	"sample from the dataset"
	def __init__(self, gtText, filePath):
		self.gtText = gtText
		self.filePath = filePath

def load_data(start = 0, end = 1000):
    print("Loading data...")
    start_time = time.time()
    root = FilePaths.fnDataPreProcessed
    file_list = os.listdir(root)
    max_file = len(file_list)
    if end > max_file:
        end = max_file

    corpus = ''
    samples = []
    for i in range(start, end):
        file_name = file_list[i]
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            label_path = file_name.replace(".jpg", ".txt")
            label_path = label_path.replace(".png", ".txt")
            label_name = os.path.join(root, label_path)
            image_path = os.path.join(root, file_name)
            with open(label_name, encoding="utf-8-sig" ) as f:
                lines = f.readlines()
                word = lines[0]
            if word != None and len(word) > 0 and len(word) <= ArchitectureConfig.MAX_TEXT_LENGTH - 2:
                samples.append(Sample(word, image_path))
                #corpus = corpus + word + ' '

    #f = open(FilePaths.fnCorpus, "w")
    #f.write(corpus)
    #f.close()

    print("Load data successfull - ", len(samples)," images")
    print("Time costs: ", str(datetime.timedelta(seconds=(time.time() - start_time))))
    
    return samples

class TextSequenceGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, samples, batch_size=ArchitectureConfig.BATCH_SIZE,
                 img_size=ArchitectureConfig.IMG_SIZE, max_text_len=ArchitectureConfig.MAX_TEXT_LENGTH,
                 downsample_factor=4, shuffle=True):

        self.imgs = [sample.filePath for sample in samples]
        self.gt_texts = [sample.gtText for sample in samples]
        self.max_text_len = max_text_len
        self.chars = ArchitectureConfig.CHARS
        self.blank_label = len(self.chars)
        self.ids = range(len(self.imgs))
        self.img_size = img_size
        self.img_w, self.img_h = self.img_size
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        X, y = self.__data_generation(indexes )
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids):
        """Generates data containing batch_size samples"""
        size = len(ids)

        if backend.image_data_format() == 'channels_first':
            X = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X = np.ones([size, self.img_w, self.img_h, 1])
        Y = np.ones([size, self.max_text_len])
        #         input_length = np.ones((size, 1), dtype=np.float32) * \
        #             (self.img_w // self.downsample_factor - 2)
        input_length = np.ones((size, 1), dtype=np.float32) * 158
        label_length = np.zeros((size, 1), dtype=np.float32)

        # Generate data
        for i, id_ in enumerate(ids):
            img = cv2.imread(self.imgs[id_], cv2.IMREAD_GRAYSCALE)  # (h, w)
            if img is None:
                continue

            if backend.image_data_format() == 'channels_first':
                img = np.expand_dims(img, 0)  # (1, h, w)
                img = np.expand_dims((0, 2, 1))  # (1, w, h)
            else:
                img = np.expand_dims(img, -1)  # (h, w, 1)
                img = img.transpose((1, 0, 2))  # (w, h, 1)

            text2label = text_to_labels(self.chars, self.gt_texts[id_])
            if (len(text2label) > self.max_text_len):
                continue
            X[i] = img
            Y[i] = text2label + \
                   [self.blank_label for _ in range(
                       self.max_text_len - len(text2label))]
            label_length[i] = len(self.gt_texts[id_])

        inputs = {
            'the_input': X,
            'the_labels': Y,
            'input_length': input_length,
            'label_length': label_length,
        }
        outputs = {'ctc': np.zeros([size])}

        return (inputs, outputs)