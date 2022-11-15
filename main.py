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
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%I:%M:%S %p"
)

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
CPU = torch.device("cpu")


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
    # data = [word for line in data for word in line]
    vocab = Counter(data)
    return vocab


def tokenise_and_pad_text(data, context_size=CONTEXT_SIZE):
    """
    Tokenise data and pad with SOS/EOS
    """
    data = tokenise(data)
    data = [add_padding(line, context_size) for line in data]
    return data


def unravel_data(data):
    """
    Unravel data into list of words
    """
    data = [word for line in data for word in line]
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

        self.embeddings = gensim_downloader.load(self.model_name)

        # if download:
        #     self.embeddings = self.download_pretrained()
        # else:
        #     if emedding_type == "w2v":
        #         self.embeddings = KeyedVectors.load_word2vec_format(self.model_path)
        #     # else:
        #     #     self.embeddings = gensim.models.KeyedVectors.load(
        #     #         f"{self.emedding_type}_model.pth"
        #     #     )

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
    def __init__(self, context_size=CONTEXT_SIZE, batch_size=BATCH_SIZE, dummy=False):
        self.data_folder = "./" if not dummy else "./dummy/"
        self.context_size = context_size

        self.batch_size = batch_size

        (self.train_words, self.validation_words, self.test_words,) = (
            self.load_dummy() if dummy else self.load_all_datasets()
        )

        self.vocab = make_vocab(self.train_words)
        self.uniq_words = list(self.vocab)
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        self.word_vectors = WordEmbedddings().get_embeddings(self.uniq_words)

    def load_dataset(self, dataset_type="train"):
        """
        Load data from file
        """

        data = read_data(f"{self.data_folder}{dataset_type}.txt")
        data = tokenise_and_pad_text(data, self.context_size)

        if dataset_type == "train":
            data = unravel_data(data)
            data = self.replace_with_unk(data)

        return data

    def load_all_datasets(self):
        return (
            self.load_dataset("train"),
            self.load_dataset("validation"),
            self.load_dataset("test"),
        )

    def load_dummy(self):
        return (
            self.load_dataset("train"),
            self.load_dataset("validation"),
            self.load_dataset("test"),
        )

    def replace_with_unk(self, words):
        # words is a list of words
        vocab = make_vocab(words)

        words = [x if vocab.get(x, 0) > MAX_UNK_FREQ else UNK_TOKEN for x in words]
        return words

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
        return np.mean(
            np.array([self.word_vectors[self.get_word_index(w)] for w in words]), axis=0
        )

    def __len__(self):
        return len(self.train_words) - self.context_size

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
        self.batch_size = batch_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim

        self.layer = nn.Sequential(
            nn.Linear(embedding_dim, HIDDEN_LAYER_1_SIZE),
            # nn.Linear((embedding_dim * context_size), HIDDEN_LAYER_1_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYER_2_SIZE, vocab_size),
            nn.Softmax(),
        )

    def forward(
        self,
        x,
    ):
        return self.layer(x)


class RNNLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=EMBEDDING_DIM,
        context_size=CONTEXT_SIZE,
        batch_size=BATCH_SIZE,
    ):
        super(RNNLM, self).__init__()
        self.batch_size = batch_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim

        self.make_hidden = nn.Sequential(
            nn.Linear(EMBEDDING_DIM + HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_1_SIZE),
            nn.Tanh(),
        )

        self.make_op = nn.Sequential(
            nn.Linear(EMBEDDING_DIM + HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYER_2_SIZE, vocab_size),
            nn.Softmax(),
        )

    def forward(self, x, hidden_state):
        try:
            combined = torch.cat((x, hidden_state), 1)
        except:
            x = torch.reshape(x, (-1, EMBEDDING_DIM))
            hidden_state = torch.reshape(hidden_state, (-1, HIDDEN_LAYER_1_SIZE))
            combined = torch.cat((x, hidden_state), 1)

        return self.make_op(combined), self.make_hidden(combined)

    def init_hidden(self, dimension=None):
        if dimension is None:
            dimension = self.batch_size
        return torch.zeros(dimension, HIDDEN_LAYER_1_SIZE)


def get_sentence_perplexity(sent, model, dataset, rnn=False):
    if len(sent) < 1:
        return -1
    with torch.no_grad():
        model.eval()
        model.to(CPU)

        if rnn:
            hidden_state = model.init_hidden(dimension=1)

        # sent_prob = 1
        log_prob = 0

        sent = add_padding(sent, model.context_size)
        # sent = torch.tensor(dataset.get_word_vectors(sent)).float()
        for i in range(len(sent) - model.context_size):
            if rnn:
                pred, hidden_state = model(
                    torch.tensor(
                        dataset.get_word_vectors(sent[i : i + model.context_size])
                    ).float(),
                    hidden_state,
                )
                hidden_state = hidden_state.detach()
            else:
                pred = model(
                    torch.tensor(
                        dataset.get_word_vectors(sent[i : i + model.context_size])
                    ).float()
                ).numpy()
            pred = pred.T
            prediction_prob = pred[dataset.get_word_index(sent[i + model.context_size])]
            log_prob += np.log(prediction_prob)
            # sent_prob *= prediction_prob

        # return (1 / sent_prob) ** (1 / len(sent))
        return np.exp(-log_prob / len(sent))


def get_text_perplexity(text, model, dataset, filepath=None, rnn=False):
    # text is list of sentences

    if len(text) < 1 or len(text[0]) < 1:
        return [-1], -1

    text_pp = [get_sentence_perplexity(sent, model, dataset, rnn) for sent in text]
    avg_pp = sum(text_pp) / len(text_pp)
    # avg_pp = np.mean(avg_pp)
    if filepath is not None:
        with open(filepath, "w") as f:
            try:
                for i in range(len(text)):
                    f.write(f"{text[i]}\t{text_pp[i]:.3f}\n")
                f.write(f"{avg_pp:.3f}\n")
            except:
                print(type(text[i]), text[i])
                print(type(text_pp[i]), text_pp[i])

    return text_pp, avg_pp


def train(model, dataset, num_epochs=1):
    """
    Return trained model and avg losses
    """
    logging.info("Training....")

    min_pp = np.inf
    best_model = 0

    dataloader = DataLoader(dataset, batch_size=model.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = []

    dims = set()

    for epoch in range(num_epochs):
        logging.info(f"EPOCH: {epoch}")
        model.to(DEVICE)
        model.train()
        losses = []

        for _, (X, y) in enumerate(dataloader):
            if TO_GPU:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
            X = X.float()
            # y = y.long()

            # Prediction
            pred = model(X)
            loss = criterion(pred, y)

            dims.add((X.shape, y.shape, pred.shape))

            # Back propagation
            optimizer.zero_grad()
            loss.backward()

            # GD step
            optimizer.step()
            losses.append(loss.item())

        torch.save(model, f"./model_{epoch}.pth")

        _, pp = get_text_perplexity(
            text=dataset.validation_words,
            model=model,
            dataset=dataset,
        )

        epoch_losses.append(np.mean(losses))

        print(pp)

        if pp < min_pp:
            min_pp = pp
            best_model = epoch

    logging.info(f"Best model: {best_model}")
    logging.info(f"Min perplexity: {min_pp}")

    print(f"Dimensions: ", dims)

    model = torch.load(f"./model_{best_model}.pth")
    return model, epoch_losses


def train_rnn(model: RNNLM, dataset, num_epochs=1):
    """
    Return trained model and avg losses
    """
    logging.info("Training....")

    min_pp = np.inf
    best_model = 0

    dataloader = DataLoader(dataset, batch_size=model.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = []

    for epoch in range(num_epochs):
        logging.info(f"EPOCH: {epoch}")
        model.to(DEVICE)
        model.train()
        losses = []
        hidden_state = model.init_hidden().to(DEVICE)

        for _, (X, y) in enumerate(dataloader):
            if TO_GPU:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
            X = X.float()
            # y = y.long()

            # clear gradients
            optimizer.zero_grad()

            # Prediction
            pred, hidden_state = model(X, hidden_state)
            hidden_state = hidden_state.detach()

            loss = criterion(pred, y)

            # Back propagation
            loss.backward()

            # GD step
            optimizer.step()
            losses.append(loss.item())

        torch.save(model, f"./rnnmodel_{epoch}.pth")

        _, pp = get_text_perplexity(
            text=dataset.validation_words,
            model=model,
            dataset=dataset,
            rnn=True,
        )

        epoch_losses.append(np.mean(losses))

        print(pp)

        if pp < min_pp:
            min_pp = pp
            best_model = epoch

    logging.info(f"Best model: {best_model}")
    logging.info(f"Min perplexity: {min_pp}")

    model = torch.load(f"./rnnmodel_{best_model}.pth")
    return model, epoch_losses


def make_pp_files(model, dataset, model_number=1, rnn=False):
    names = ["test", "validation", "train"]
    for name in names:
        data = read_data(f"{name}.txt")
        data = tokenise_and_pad_text(data, model.context_size)

        get_text_perplexity(
            text=data,
            model=model,
            dataset=dataset,
            rnn=rnn,
            filepath=f"./2019115003-LM{model_number}-{name}-perplexity.txt",
        )


def load_stored_files(model_path, dataset_path):
    model = torch.load(model_path)
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    return model, dataset


if __name__ == "__main__":

    prompt = """
    Enter:
    1 - to get perplexity of an input sentence
    2 - to train NNLM and make pp files
    3 - to train RNNLM and make pp files
    4 - to load stored model and dataset and make pp files
    """

    x = input(prompt).strip()
    # x = "train"
    if x == "1":
        logging.info("Loading Corpus....")
        corpus = Corpus()

        x = input("Enter 1 for NNLM or 2 if you want RNN model").strip()
        logging.info("Loading Model....")
        model = torch.load("model_0.pth") if x == "1" else torch.load("rnnmodel_0.pth")

        x = input("Enter a sentence: ").strip()

        print(get_sentence_perplexity(tokenise(x), model, corpus, rnn=x == "2"))
    elif x == "2":
        logging.info("Loading Corpus....")
        corpus = Corpus()
        logging.info("Loading NNLM Model....")
        model = NNLM(vocab_size=len(corpus.vocab))
        model, losses = train(model, corpus)
        print(losses)
        # if x == "make":
        logging.info("Making pp files.....")
        make_pp_files(model, corpus, 1)
    elif x == "3":
        logging.info("Loading Corpus....")
        corpus = Corpus()
        logging.info("Loading RNNLM Model....")
        model = RNNLM(vocab_size=len(corpus.vocab))
        model, losses = train_rnn(model, corpus)
        print(losses)
        logging.info("Making pp files.....")
        make_pp_files(model, corpus, 2, rnn=True)
    else:
        logging.info("Loading Corpus....")
        corpus = Corpus()

        x = input("Enter 1 for NNLM or 2 if you want RNN model").strip()
        logging.info("Loading Model....")
        model = torch.load("model_0.pth") if x == "1" else torch.load("rnnmodel_0.pth")

        logging.info("Making pp files.....")
        make_pp_files(model, corpus, x, rnn=x == "2")
