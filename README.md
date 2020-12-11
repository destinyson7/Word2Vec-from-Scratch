# Advanced NLP
## Assignment 1

Training Vector Representation of Words

## CBOW

```bash
jupyter-notebook
```

Open cbow.ipynb
and run each cell one by one

## Skip-Gram

```bash
jupyter-notebook
```

Open skip-gram.ipynb
and run each cell one by one

## How to restore embeddings

Go to this Google Drive Folder: https://drive.google.com/drive/folders/18QNuxYVmu3z28bOUFIvbjkSUTIhcucW4?usp=sharing
Download both the embedding files

### How to load and run

They are json files.
There are 3 important keys in the json dictionary of both the files (total 4 keys, 4th one is not important)
1. word2Ind: Mapping from word to its index
2. Ind2word: Mapping from index to the word present at that index
3. W1 (for skip-gram) or W2 (for cbow): these contain the embeddings. At every index, a list of floats is there.

To get any embedding for cbow,
```python
W2[word2Ind[word]]
```

To get any embedding for skip-gram,
```python
W1[word2Ind[word]]
```

#### Tanish Lad (2018114005)