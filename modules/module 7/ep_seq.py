import torch

GO_token, EOS_token, PAD_token = 0, 1, 2

class Lang:
    """Class that represents the vocabulary in one language"""
    def __init__(self, name):
        self.name, self.word2ix, self.word2cnt = name, {}, {}
        self.ix2word = {GO_token:'GO', EOS_token:'EOS', PAD_token:'PAD'}
        self.n_words = 3  # Count GO, EOS, PAD
    def add_sentence(self, sentence):
        for w in sentence.split(' '):
            if w not in self.word2ix:
                self.word2ix[w] = self.n_words
                self.word2cnt[w] = 1
                self.ix2word[self.n_words] = w
                self.n_words += 1
            else:
                self.word2cnt[w] += 1

class NMTDataset(torch.utils.data.Dataset):
    """Dataset for NMT. The output is a tuple of two tensors, which represent the same sequence
    in the source and target languages. Each sentence tensor contains the indices of the words in the vocabulary
    """
    def __init__(self, txt_file, dataset_size:int, mxlen:int, seed:int=0, vectorized=False):
        from io import open
        import random
        super().__init__()
        random.seed(seed)
        self.mxlen, self.vectorized = mxlen, vectorized
        lines = open(txt_file, encoding='utf-8').read().strip().split('\n')
        # split every line into pairs and normalize
        pairs = []
        for _ in range(dataset_size):
            pairs += [[self.normalize_string(s) for s in lines[_].split('\t')]]
        pairs = [list(reversed(p)) for p in pairs]
        self.input_lang = Lang('fra')
        self.output_lang = Lang('eng')
        # filter the pairs to reduce the size of the dataset
        filtered = list()
        eng_prefixes = (
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s ",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
        )
        for p in pairs:
            if len(p[0].split(' '))<self.mxlen-1 and len(p[1].split(' '))<self.mxlen-1 and p[1].startswith(eng_prefixes):
                filtered.append(p)
        pairs = filtered
        pairs = [random.choice(pairs) for _ in range(dataset_size)]
        # create vocabularies
        for pair in pairs:
            self.input_lang.add_sentence(pair[0])
            self.output_lang.add_sentence(pair[1])
        self.pairs = pairs
        self.dataset = list()
        # convert all sentences to tensors
        for pair in self.pairs:
            source_sentence_by_ix = [self.input_lang.word2ix[w] for w in pair[0].split(' ')]+[EOS_token]
            output_sentence_by_ix = [self.output_lang.word2ix[w] for w in pair[1].split(' ')]+[EOS_token]
            if self.vectorized:  # set fixed size length
                source_sentence_by_ix += [PAD_token]*(self.mxlen-len(source_sentence_by_ix))
                output_sentence_by_ix += [PAD_token]*(self.mxlen-len(output_sentence_by_ix))
                source_sentence_tensor = torch.tensor(source_sentence_by_ix, dtype=torch.long)
                output_sentence_tensor = torch.tensor(output_sentence_by_ix, dtype=torch.long)
            else:
                source_sentence_tensor = torch.tensor(source_sentence_by_ix, dtype=torch.long).view(-1, 1)
                output_sentence_tensor = torch.tensor(output_sentence_by_ix, dtype=torch.long).view(-1, 1)
            self.dataset.append((source_sentence_tensor, output_sentence_tensor))
    @staticmethod
    def normalize_string(s):
        import re
        import unicodedata
        s = ''.join(
            c for c in unicodedata.normalize('NFD', s.lower().strip())
            if unicodedata.category(c) != 'Mn'
        )
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s
    def sentence_to_sequence(self, sentence:str):
        """Convert the string sentence to a tensor"""
        sequence = [self.input_lang.word2ix[w] for w in self.normalize_string(sentence).split(' ')]+[EOS_token]
        if self.vectorized:  # fixed vector length
            sequence = sequence + [PAD_token] * (self.mxlen-len(sequence))
        sequence = torch.tensor(sequence, dtype=torch.long).view(-1,1)
        return sequence
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx:int):
        return self.dataset[idx]