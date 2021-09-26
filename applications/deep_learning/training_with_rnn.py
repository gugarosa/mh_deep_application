import torch
import torchtext
from models import LSTM
from torch.utils.data import DataLoader


def yield_tokens(iterator):
    # Iterates through iterator
    for _, text in iterator:
        # Yields tokenized text
        yield tokenizer(text)

def collate_batch(batch):
    # Empty lists to hold labels and texts
    labels, texts = [], []
    
    # Iterates through every sample in batch
    for (label, text) in batch:
        # Appends pre-processed labels and texts
        labels.append(label_transform(label))
        texts.append(torch.tensor(text_transform(text)))

    return torch.nn.utils.rnn.pad_sequence(texts).to(device), torch.tensor(labels).to(device)

# Gathers the available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loads data, tokenizer and creates the vocabulary
train = torchtext.datasets.IMDB(split='train')
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Instantiates the transforms
text_transform = lambda x: vocab(tokenizer(x))
label_transform = lambda x: 1 if x == 'pos' else 0

# Loads training and testing data
train, test = torchtext.datasets.IMDB()

# Creates training and testing loaders
train_loader = DataLoader(train, batch_size=100, num_workers=0, collate_fn=collate_batch)
test_loader = DataLoader(test, batch_size=100, num_workers=0, collate_fn=collate_batch)

# Instantiates the model
model = LSTM(n_input=len(vocab), n_embedding=256, n_hidden=128, n_output=2, lr=0.001, device=device)

# Fits the model
model.fit(train_loader, test_loader, epochs=10)
