import embeddings
import os
import random
import shutil
import zipfile
import urllib.request
import numpy as np
from glob import glob


def encode_sentences(sentences, max_length, emb_lookup, unk_emb, unks):
    encoded_sents = []
    for sentence in sentences:
        sentence_emb = [[0] * emb_lookup.d_emb for _ in range(max_length)]
        for i, w in enumerate(sentence):
            if w in emb_lookup:
                sentence_emb[i][:] = emb_lookup.emb(w)
            else:
                unks.add(w)
                sentence_emb[i][:] = unk_emb
        encoded_sents.append(sentence_emb)
    return encoded_sents


class UCISentimentDataset:
    """
    dataset_dir
    ├── uci_sentiment
        ├── yelp_labelled.txt           (reviews from yelp, labelled with positive/negative sentiment)
        ├── amazon_cells_labelled.txt   (reviews from amazon, labelled with positive/negative sentiment)
        ├── imdb_labelled.txt           (reviews from imdb, labelled with positive/negative sentiment)
    """
    
    @staticmethod
    def _download_and_extract(root):
        dataset_path = os.path.join(root, "uci_sentiment")
        os.makedirs(dataset_path, exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip"
        filename = url.split("/")[-1]
        z_path = os.path.join(dataset_path, filename)
        uncompressed_path = os.path.join(dataset_path, filename[:-4])
        
        if not os.path.exists(uncompressed_path):
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, z_path)

            print(f"Extracting {z_path}")
            with zipfile.ZipFile(z_path, 'r') as f_in:
                f_in.extractall(uncompressed_path)
            os.remove(z_path)
            
            txt_paths = glob(uncompressed_path + f'/**/*.txt', recursive=True)
            for path in txt_paths:
                shutil.move(path, os.path.join(dataset_path, os.path.basename(path)))
            shutil.rmtree(uncompressed_path)
    
    def __init__(self, root, emb_lookup, download=True):
        if download and not os.path.exists(os.path.join(root, "uci_sentiment")):
            self._download_and_extract(root)
        
        yelp_path = os.path.join(root, "uci_sentiment", 'yelp_labelled.txt')
        amazon_path = os.path.join(root, "uci_sentiment", 'amazon_cells_labelled.txt')
        imdb_path = os.path.join(root, "uci_sentiment", 'imdb_labelled.txt')

        self.sentences = []
        self.labels = []

        for path in [yelp_path, amazon_path, imdb_path]:
            with open(path, 'r') as f:
                for line in f:
                    sentence, label = line.strip().split('\t')
                    sentence = sentence.strip().split()

                    self.labels.append(int(label))
                    self.sentences.append(sentence)
                    
        self.emb_lookup = emb_lookup
        self.encode()
                    
    def encode(self):
        max_length = 0
        for sent in self.sentences:
            max_length = max(max_length, len(sent))
        
        unks = set()
        unk_emb = [0.1 * (random.random() - 0.5) for i in range(max_length)]
        
        self.samples = encode_sentences(self.sentences, max_length, self.emb_lookup, unk_emb, unks)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sample = self.samples[item]
        label = np.array([self.labels[item]])
        return sample, label
