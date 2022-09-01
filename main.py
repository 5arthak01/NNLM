import re
import pickle
from clean import clean_text
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import gensim.downloader as gensim_downloader
from gensim.models import KeyedVectors

# from gensim.models import Word2Vec
# import gensim.models.KeyedVectors.load_word2vec_format as load_word2vec_format
# from gensim.scripts.glove2word2vec import glove2word2vec


BATCH_SIZE = 128
MAX_UNK_FREQ = 2
EMBEDDING_TYPE = "w2v"
EMBEDDING_DIM = 300 if EMBEDDING_TYPE == "w2v" else 50

# Given in question
CONTEXT_SIZE = 4
NGRAM_SIZE = CONTEXT_SIZE + 1
HIDDEN_LAYER_1_SIZE = 300
HIDDEN_LAYER_2_SIZE = 300

# can be anything
RIGHT_PAD_SYMBOL = "<EOS>"
LEFT_PAD_SYMBOL = "<SOS>"
UNK_TOKEN = "<UNK>"

TO_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if TO_GPU else "cpu")


def is_padding(word):
    return word == LEFT_PAD_SYMBOL or word == RIGHT_PAD_SYMBOL


def add_padding(text, n):
    n = max(n, 0)
    return [LEFT_PAD_SYMBOL] * n + text + [RIGHT_PAD_SYMBOL] * n


def read_data(filename):
    """
    Read data from file
    """
    with open(filename, "r") as f:
        data = f.read()
    return data


def tokenise(data):
    """
    Tokenise data, return list of sentences,
    which are lists of words
    """
    data = re.sub(r"\s+", " ", data)
    data = clean_text(data)
    data = re.sub(r"\s+", " ", data)
    data = sent_tokenize(data)
    data = [word_tokenize(x) for x in data]
    return data


def make_vocab(data):
    """
    Make vocabulary dict from data
    """
    # data = read_data(filename)
    # data = tokenise(data)
    vocab = [word for line in data for word in line]
    vocab = Counter(vocab)
    # vocab = sorted(vocab, key=vocab.get, reverse=True)
    return vocab


def tokenise_and_pad_text(data, context_size=CONTEXT_SIZE):
    """
    Tokenise data and pad with SOS/EOS
    """
    data = tokenise(data)
    data = [add_padding(line, context_size) for line in data]
    return data


class WordEmbedddings:
    # https://github.com/RaRe-Technologies/gensim-data
    def __init__(self, download=False, emedding_type="w2v"):
        self.emedding_type = emedding_type
        self.model_path = f"{self.emedding_type}_model"
        self.model_name = (
            "word2vec-google-news-300"
            if self.emedding_type == "w2v"
            else "glove-wiki-gigaword-50"
        )

        if download:
            self.embeddings = self.download_pretrained()
        else:
            if emedding_type == "w2v":
                self.embeddings = KeyedVectors.load_word2vec_format(self.model_path)
            # else:
            #     self.embeddings = gensim.models.KeyedVectors.load(
            #         f"{self.emedding_type}_model.pth"
            #     )

        self.embedding_size = EMBEDDING_DIM

        custom_tokens = [LEFT_PAD_SYMBOL, RIGHT_PAD_SYMBOL, UNK_TOKEN]
        self.custom_embeddings = {
            token: np.random.rand(self.embedding_size) for token in custom_tokens
        }

    def download_pretrained(self):
        model = gensim_downloader.load(self.model_name)
        # if self.emedding_type == "glove":
        #     glove2word2vec("pretrained_model", "pretrained_model")

        # model = KeyedVectors(model)
        model.save_word2vec_format(self.model_path)
        return model

    def get_word_embedding(self, word):
        """
        Get embedding for word
        """
        try:
            return self.embeddings[word]
        except KeyError:
            if is_padding(word):
                return self.custom_embeddings[word]

            return self.custom_embeddings[UNK_TOKEN]

    def get_embeddings(self, words):
        """
        Get embeddings for list of words
        """
        return [self.get_word_embedding(word) for word in words]


class Corpus(Dataset):
    def __init__(self, context_size=CONTEXT_SIZE, batch_size=BATCH_SIZE):
        self.data_folder = "./"
        self.context_size = context_size
        self.batch_size = batch_size
        (
            self.train_words,
            self.validation_words,
            self.test_words,
        ) = self.load_all_datasets()
        self.vocab = make_vocab(self.train_words)
        self.uniq_words = sorted(self.vocab, key=self.vocab.get, reverse=True)
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        # self.train_words_as_inds = [self.word_to_index[w] for w in self.train_words]
        # self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_vectors = WordEmbedddings().get_embeddings(self.uniq_words)

    def load_dataset(self, dataset_type="train"):
        """
        Load data from file
        """
        if dataset_type not in ["train", "validation", "test"]:
            raise ValueError(
                "dataset_type must be one of ['train', 'validation', 'test']"
            )

        data = read_data(f"{self.data_folder}{dataset_type}.txt")
        data = tokenise_and_pad_text(data, self.context_size)

        if dataset_type == "train":
            data = self.replace_with_unk(data)

        return data

    def load_all_datasets(self):
        return (
            self.load_dataset("train"),
            self.load_dataset("validation"),
            self.load_dataset("test"),
        )

    def replace_with_unk(self, data):
        vocab = make_vocab(data)

        data = [
            [x if vocab.get(x, 0) > MAX_UNK_FREQ else UNK_TOKEN for x in line]
            for line in data
        ]
        return data

    def get_word_onehot(self, word):
        """
        Get onehot representation of word
        """
        index = self.word_to_index[word]
        onehot = np.zeros(len(self.uniq_words))
        onehot[index] = 1
        return onehot

    def get_word_index(self, word):
        if word not in self.vocab:
            word = UNK_TOKEN
        return self.word_to_index[word]

    def get_word_vectors(self, words):
        return [self.word_vectors[self.word_to_index[w]] for w in words]

    def __len__(self):
        return len(self.train_words) - self.context_size + 1

    def __getitem__(self, index):
        # ret = (context, word)

        ret = (
            torch.tensor(
                self.get_word_vectors(
                    self.train_words[index : index + self.context_size]
                )
            ),
            torch.tensor(
                self.get_word_onehot(self.train_words[index + self.context_size])
            ),
        )

        # if TO_GPU:
        #     ret.to(DEVICE)

        return ret


class NNLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=EMBEDDING_DIM,
        context_size=CONTEXT_SIZE,
        batch_size=BATCH_SIZE,
    ):
        super(NNLM, self).__init__()

        # self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.embeddings.weight.data.uniform_(0, 1)
        self.batch_size = batch_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim

        self.layer1 = nn.Sequential(
            nn.Linear((embedding_dim * context_size), HIDDEN_LAYER_1_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_2_SIZE, vocab_size),
            nn.Softmax(dim=1),
        )

    def forward(
        self,
        x,
    ):
        # x = self.embeddings(x).view((self.batch_size, -1))
        x = x.view((self.batch_size, -1))
        return self.layer(x)


def get_sentence_perplexity(sent, model, get_word_index):
    if len(sent) < 1:
        return -1

    # sent_prob = 1
    log_prob = 0

    sent = add_padding(sent, model.context_size)

    for i in range(len(sent) - model.context_size):
        pred = model(sent[i : i + model.context_size]).detach().numpy()
        prediction_prob = pred[get_word_index(sent[i + model.context_size])]
        log_prob += np.log(prediction_prob)
        # sent_prob *= prediction_prob

    return np.exp(-log_prob / len(sent))
    # return (1 / sent_prob) ** (1 / len(sent))


def get_text_perplexity(text, model, get_word_index, filepath=None):
    # text is list of sentences

    model.eval()
    if len(text) < 1 or len(text[0]) < 1:
        return [-1], -1

    text_pp = [get_sentence_perplexity(sent, model, get_word_index) for sent in text]
    avg_pp = sum(text_pp) / len(text_pp)

    if filepath is not None:
        with open(filepath, "w") as f:
            for i in range(len(text)):
                f.write(f"{text[i]}\t{text_pp[i]:.3f}\n")
            f.write(f"{avg_pp:.3f}\n")

    return text_pp, avg_pp


def train(dataset, model, num_epochs=1):
    """
    Return trained model and avg losses
    """
    model.train()
    min_pp = np.inf
    best_model = 0

    dataloader = DataLoader(dataset, batch_size=model.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = []

    for epoch in range(num_epochs):
        print("EPOCH: ", epoch)
        model.train()
        losses = []

        for _, (X, y) in enumerate(dataloader):
            if TO_GPU:
                X = X.to(DEVICE)
                y = y.to(DEVICE)

            # Prediction
            pred = model(X)
            loss = criterion(pred, y)

            # Back propagation
            optimizer.zero_grad()
            loss.backward()

            # GD step
            optimizer.step()
            losses.append(loss.item())

        torch.save(model, f"./model_{epoch}.pth")

        pp = get_text_perplexity(
            text=dataset.validation_words,
            model=model,
            get_word_index=dataset.get_word_index,
        )

        epoch_losses.append(np.mean(losses))

        if pp < min_pp:
            min_pp = pp
            best_model = epoch

    print("Best model: ", best_model)
    print("Min perplexity: ", min_pp)

    model = torch.load(f"./model_{best_model}.pth")
    return model, epoch_losses


def make_pp_files(model, dataset):
    get_text_perplexity(
        text=dataset.train_words,
        model=model,
        get_word_index=dataset.get_word_index,
        filepath="./2019115003-LM1-train-perplexity.txt",
    )
    get_text_perplexity(
        text=dataset.validation_words,
        model=model,
        get_word_index=dataset.get_word_index,
        filepath="./2019115003-LM1-validation-perplexity.txt",
    )
    get_text_perplexity(
        text=dataset.test_words,
        model=model,
        get_word_index=dataset.get_word_index,
        filepath="./2019115003-LM1-test-perplexity.txt",
    )


def load_stored_files(model_path, dataset_path):
    model = torch.load(model_path)
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    return model, dataset


if __name__ == "__main__":
    x = input("To get perplexity of a sentence, enter 1\n").strip()
    if x == "1":
        print("Loading....")
        model, dataset = load_stored_files("model_0.pth", "brown_corpus.pkl")
        x = input("Enter a sentence: ").strip()

        print(
            get_sentence_perplexity(
                tokenise_and_pad_text(x), model, dataset.get_word_index
            )
        )
    # elif x == "train":
    else:
        corpus = Corpus()
        model = NNLM()
        rets = train(model, corpus)
        for x in rets:
            print(x)
