import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import spacy
from collections import Counter

class Vocab:
    def __init__(self, itos, stoi):
        self.itos = itos
        self.stoi = stoi

    def __len__(self):
        return len(self.itos)

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx] if 0 <= idx < len(self.itos) else "<unk>"

    def lookup_indices(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]


class Multi30kDataset(Dataset):
    def __init__(self, split='train', vocab_path='vocab.pt'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        self.dataset = load_dataset("bentrevett/multi30k", split=split)
        
        # Load spacy tokenizers for de and en
        self.de_nlp = spacy.load("de_core_news_sm")
        self.en_nlp = spacy.load("en_core_web_sm")
        
        # Check if vocab path exists in current directory or parent directory
        resolved_vocab_path = vocab_path
        if not os.path.exists(resolved_vocab_path):
            parent_path = os.path.join("..", vocab_path)
            if os.path.exists(parent_path):
                resolved_vocab_path = parent_path

        # If vocab file exists, load it. Otherwise, build and save it.
        if os.path.exists(resolved_vocab_path):
            checkpoint = torch.load(resolved_vocab_path, weights_only=False)
            self.src_vocab = checkpoint['src_vocab']
            self.tgt_vocab = checkpoint['tgt_vocab']
        else:
            if split != 'train':
                # Build from train split to ensure consistent vocabulary mapping
                train_dataset = load_dataset("bentrevett/multi30k", split='train')
                self.src_vocab, self.tgt_vocab = self.build_vocab_from_dataset(train_dataset)
            else:
                self.src_vocab, self.tgt_vocab = self.build_vocab_from_dataset(self.dataset)
            torch.save({'src_vocab': self.src_vocab, 'tgt_vocab': self.tgt_vocab}, resolved_vocab_path)
        
        # Process the data
        self.data = self.process_data()

    def build_vocab_from_dataset(self, dataset, min_freq=2):
        print("Building vocabs from training set...")
        de_counter = Counter()
        en_counter = Counter()
        for item in dataset:
            de_tokens = [tok.text.lower() for tok in self.de_nlp.tokenizer(item['de'])]
            en_tokens = [tok.text.lower() for tok in self.en_nlp.tokenizer(item['en'])]
            de_counter.update(de_tokens)
            en_counter.update(en_tokens)
        
        # Define special tokens
        special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
        
        # Build German vocab
        de_itos = special_tokens + [tok for tok, freq in de_counter.items() if freq >= min_freq]
        de_stoi = {tok: idx for idx, tok in enumerate(de_itos)}
        src_vocab = Vocab(de_itos, de_stoi)
        
        # Build English vocab
        en_itos = special_tokens + [tok for tok, freq in en_counter.items() if freq >= min_freq]
        en_stoi = {tok: idx for idx, tok in enumerate(en_itos)}
        tgt_vocab = Vocab(en_itos, en_stoi)
        
        print(f"German vocab size: {len(src_vocab)}, English vocab size: {len(tgt_vocab)}")
        return src_vocab, tgt_vocab

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        processed = []
        for item in self.dataset:
            # German (source): tokenized, lowercase, convert to indices, wrap with <sos> and <eos>
            de_tokens = [tok.text.lower() for tok in self.de_nlp.tokenizer(item['de'])]
            de_indices = [self.src_vocab.stoi.get('<sos>')] + \
                         [self.src_vocab.stoi.get(tok, self.src_vocab.stoi['<unk>']) for tok in de_tokens] + \
                         [self.src_vocab.stoi.get('<eos>')]
            
            # English (target): tokenized, lowercase, convert to indices, wrap with <sos> and <eos>
            en_tokens = [tok.text.lower() for tok in self.en_nlp.tokenizer(item['en'])]
            en_indices = [self.tgt_vocab.stoi.get('<sos>')] + \
                         [self.tgt_vocab.stoi.get(tok, self.tgt_vocab.stoi['<unk>']) for tok in en_tokens] + \
                         [self.tgt_vocab.stoi.get('<eos>')]
            
            processed.append((de_indices, en_indices))
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
