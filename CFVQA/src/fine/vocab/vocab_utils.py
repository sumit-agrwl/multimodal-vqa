import pickle
import json
import torch
import numpy as np
from torchtext.data import get_tokenizer
from collections import Counter, OrderedDict, defaultdict
from torchtext.vocab import pretrained_aliases, Vectors
from fine.constants import constants

ROOT_DIR = constants.ROOT_DIR
config_file = constants.CONFIG_FILE
sg_vocab_path = constants.SG_VOCAB_PATH
qa_vocab_path = constants.QA_VOCAB_PATH
programmed_question_path = constants.SPLIT_TO_PROGRAMMED_QUESTION_PATH_TABLE['train']

class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    # TODO (@mttk): Populate classs with default values of special symbols
    UNK = '<unk>'

    def __init__(self, counter, max_size=None, min_freq=1, specials=('<unk>', '<pad>'),
                 vectors=None, unk_init=None, vectors_cache=None, specials_first=True):
        """Create a Vocab object from a collections.Counter.
        Args:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary. Default: ['<unk'>, '<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: 'torch.zeros'
            vectors_cache: directory for cached vectors. Default: '.vector_cache'
            specials_first: Whether to add special tokens into the vocabulary at first.
                If it is False, they are added into the vocabulary at last.
                Default: True.
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list()
        self.unk_index = None
        if specials_first:
            self.itos = list(specials)
            # only extend max size if specials are prepended
            max_size = None if max_size is None else max_size + len(specials)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        if Vocab.UNK in specials:  # hard-coded for now
            unk_index = specials.index(Vocab.UNK)  # position in list
            # account for ordering of specials, set variable
            self.unk_index = unk_index if specials_first else len(self.itos) + unk_index
            self.stoi = defaultdict(self._default_unk_index)
        else:
            self.stoi = defaultdict()

        if not specials_first:
            self.itos.extend(list(specials))

        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def _default_unk_index(self):
        return self.unk_index

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __getstate__(self):
        # avoid picking defaultdict
        attrs = dict(self.__dict__)
        # cast to regular dict
        attrs['stoi'] = dict(self.stoi)
        return attrs

    def __setstate__(self, state):
        if state.get("unk_index", None) is None:
            stoi = defaultdict()
        else:
            stoi = defaultdict(self._default_unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def lookup_indices(self, tokens):
        indices = [self.__getitem__(token) for token in tokens]
        return indices

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

    def load_vectors(self, vectors, **kwargs):
        """
        Args:
            vectors: one of or a list containing instantiations of the
                GloVe, CharNGram, or Vectors classes. Alternatively, one
                of or a list of available pretrained vectors:
                charngram.100d
                fasttext.en.300d
                fasttext.simple.300d
                glove.42B.300d
                glove.840B.300d
                glove.twitter.27B.25d
                glove.twitter.27B.50d
                glove.twitter.27B.100d
                glove.twitter.27B.200d
                glove.6B.50d
                glove.6B.100d
                glove.6B.200d
                glove.6B.300d
            Remaining keyword arguments: Passed to the constructor of Vectors classes.
        """
        if not isinstance(vectors, list):
            vectors = [vectors]
        for idx, vector in enumerate(vectors):
            if isinstance(vector, str):
                # Convert the string pretrained vector identifier
                # to a Vectors object
                if vector not in pretrained_aliases:
                    raise ValueError(
                        "Got string input vector {}, but allowed pretrained "
                        "vectors are {}".format(
                            vector, list(pretrained_aliases.keys())))
                vectors[idx] = pretrained_aliases[vector](**kwargs)
            elif not isinstance(vector, Vectors):
                raise ValueError(
                    "Got input vectors of type {}, expected str or "
                    "Vectors object".format(type(vector)))

        tot_dim = sum(v.dim for v in vectors)
        self.vectors = torch.Tensor(len(self), tot_dim)
        for i, token in enumerate(self.itos):
            start_dim = 0
            for v in vectors:
                end_dim = start_dim + v.dim
                self.vectors[i][start_dim:end_dim] = v[token.strip()]
                start_dim = end_dim
            assert(start_dim == tot_dim)

    def set_vectors(self, stoi, vectors, dim, unk_init=torch.Tensor.zero_):
        """
        Set the vectors for the Vocab instance from a collection of Tensors.
        Args:
            stoi: A dictionary of string to the index of the associated vector
                in the `vectors` input argument.
            vectors: An indexed iterable (or other structure supporting __getitem__) that
                given an input index, returns a FloatTensor representing the vector
                for the token associated with the index. For example,
                vector[stoi["string"]] should return the vector for "string".
            dim: The dimensionality of the vectors.
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: 'torch.zeros'
        """
        self.vectors = torch.Tensor(len(self), dim)
        for i, token in enumerate(self.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                self.vectors[i] = vectors[wv_index]
            else:
                self.vectors[i] = unk_init(self.vectors[i])

class TextProcessor:
    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 tokenize=None, tokenizer_language='en', include_lengths=False,
                 batch_first=False, pad_token="<pad>", unk_token="<unk>", lower=False,
                 pad_first=False, truncate_first=False, stop_words=None,
                 is_target=False):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.lower = lower
        # store params to construct tokenizer for serialization
        # in case the tokenizer isn't picklable (e.g. spacy)
        self.tokenizer_args = (tokenize, tokenizer_language)
        self.tokenize = get_tokenizer(tokenize, tokenizer_language)
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token if self.sequential else None
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        try:
            self.stop_words = set(stop_words) if stop_words is not None else None
        except TypeError:
            raise ValueError("Stop words must be convertible to a set")
        self.is_target = is_target
        self.vocab = None
        
    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.
        If `sequential=True`, the input will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if self.sequential and isinstance(x, str):
            x = self.tokenize(x.rstrip('\n'))
        if self.lower:
            x = [x.lower() for w in x]
        if self.sequential and self.use_vocab and self.stop_words is not None:
            x = [w for w in x if w not in self.stop_words]
        return x
    
    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.
        Pad, numericalize, and postprocess a batch and create a tensor.
        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
        """
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x))
                    + ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token])
                    + [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        return padded

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.
        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token] + kwargs.pop('specials', [])
            if tok is not None))
        self.vocab = Vocab(counter, specials=specials, **kwargs)

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])): List of tokenized
                and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explicitly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, str)
                       else x for x in arr]

        var = torch.tensor(arr, dtype=self.dtype, device=device)

        if self.sequential and not self.batch_first:
            var.t_()
        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var


class QAVocab():
    qa_encoding_text = TextProcessor(sequential=True,
                        tokenize="spacy",
                        init_token="<start>",
                        eos_token="<end>",
                        tokenizer_language='en_core_web_sm',
                        include_lengths=False,
                        batch_first=False)

    def __init__(self, build=False):
        if build:
            self.build_vocab()
            self.update_config()
        else:
            self.load_vocab()
    
    def build_vocab(self):
        text_list = []
        with open(programmed_question_path) as f:
            questions = json.load(f)
        print (len(questions))
        for datum in questions:
            question_text = datum[1]
            program_text_tokenized = datum[6]
            full_answer_text = datum[5]
            # Question
            # must do preprocess (tokenization) before doing the buld vocab
            question_text_tokenized = QAVocab.qa_encoding_text.preprocess(question_text)
            text_list.append(question_text_tokenized)
            # Program
            text_list.append(program_text_tokenized)
            # Full Answer
            # must do preprocess (tokenization) before doing the buld vocab
            full_answer_text_tokenized = QAVocab.qa_encoding_text.preprocess(full_answer_text)
            text_list.append(full_answer_text_tokenized)

        QAVocab.qa_encoding_text.build_vocab(text_list, vectors="glove.6B.300d")  

        with open(qa_vocab_path, 'wb') as f:
            pickle.dump(QAVocab.qa_encoding_text, f)

        del text_list
    
    def load_vocab(self):
        with open(qa_vocab_path, 'rb') as f:
            text_reloaded = pickle.load(f)
            QAVocab.qa_encoding_text = text_reloaded

    def update_config(self):
        with open(config_file) as f:
            config = json.load(f)
        
        config['vocab_size']    = len(QAVocab.qa_encoding_text.vocab)
        config['init_token']    = QAVocab.qa_encoding_text.vocab.stoi[QAVocab.qa_encoding_text.init_token]
        config['eos_token']     = QAVocab.qa_encoding_text.vocab.stoi[QAVocab.qa_encoding_text.eos_token]
        config['pad_token']     = QAVocab.qa_encoding_text.vocab.stoi[QAVocab.qa_encoding_text.pad_token]
        config['unk_token']     = QAVocab.qa_encoding_text.vocab.stoi[QAVocab.qa_encoding_text.unk_token]

        with open(config_file, 'w') as f:
            json.dump(config, f)

class SGVocab():
    sg_encoding_text = TextProcessor(sequential=True,
                        tokenize="spacy",
                        init_token="<start>",
                        eos_token="<end>",
                        tokenizer_language='en_core_web_sm',
                        include_lengths=False,
                        batch_first=False)

    def __init__(self, build=False):
        if build:
            self.build_vocab()
            self.update_config()
        else:
            self.load_vocab()
    
    def build_vocab(self):
        def load_str_list(fname):
            with open(fname) as f:
                lines = f.read().splitlines()
            return lines

        text_list = []
        text_list += load_str_list(ROOT_DIR / 'meta_info/name_gqa.txt')
        text_list += load_str_list(ROOT_DIR / 'meta_info/attr_gqa.txt')
        text_list += load_str_list(ROOT_DIR / 'meta_info/rel_gqa.txt')

        text_list += constants.OBJECTS_INV + constants.RELATIONS_INV + constants.ATTRIBUTES_INV
        text_list.append("<self>") # add special token for self-connection
        text_list = [text_list]

        SGVocab.sg_encoding_text.build_vocab(text_list, vectors="glove.6B.300d")
        del text_list

        with open(sg_vocab_path, 'wb') as f:
            pickle.dump(SGVocab.sg_encoding_text, f)
    
    def load_vocab(self):
        with open(sg_vocab_path, 'rb') as f:
            text_reloaded = pickle.load(f)
            SGVocab.sg_encoding_text = text_reloaded

    def update_config(self):
        with open(config_file) as f:
            config = json.load(f)

        config['sg_vocab_size'] = len(SGVocab.sg_encoding_text.vocab)
        config['sg_init_token'] = SGVocab.sg_encoding_text.vocab.stoi[SGVocab.sg_encoding_text.init_token]
        config['sg_eos_token']  = SGVocab.sg_encoding_text.vocab.stoi[SGVocab.sg_encoding_text.eos_token]
        config['sg_pad_token']  = SGVocab.sg_encoding_text.vocab.stoi[SGVocab.sg_encoding_text.pad_token]
        config['sg_unk_token']  = SGVocab.sg_encoding_text.vocab.stoi[SGVocab.sg_encoding_text.unk_token]
        with open(config_file, 'w') as f:
            json.dump(config, f)

if __name__ == '__main__':
    sgVocab = SGVocab(build = True)
    qaVocab = QAVocab(build = True)
