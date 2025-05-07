# Word2Vec Skip-gram with Negative Sampling

## Project Overview

This repository contains an implementation of the Skip-gram model with Negative Sampling for learning word embeddings, trained on the [text8 dataset](http://mattmahoney.net/dc/text8.zip). Word embeddings capture semantic and syntactic properties of words in dense vector representations.

## Features

* **Data Download & Preprocessing**: Tokenization, lowercasing, punctuation removal, vocabulary construction.
* **Model Implementation**: Skip-gram architecture with negative-sampling loss implemented from scratch using NumPy.
* **Training**: Configurable hyperparameters (embedding dimension, context window size, negative samples, learning rate).
* **Evaluation**:

  * 2D visualization of embeddings via SVD.
  * Analogy tests (e.g., “king” - “man” + “woman” ≈ “queen”).
* **Hyperparameter Analysis**: Impact of embedding size, window size, and negative sample count on embedding quality.

## Repository Structure

```
├── data
│   └── text8.zip            # Downloaded text8 dataset
├── notebooks
│   └── Word2Vec.ipynb       # Exploratory work and results
├── src
│   ├── preprocess.py        # Text cleaning and vocabulary creation
│   ├── dataset.py           # Generating training pairs and negative samples
│   ├── model.py             # Skip-gram model implementation
│   ├── train.py             # Training loop and logging
│   └── evaluate.py          # Visualization and analogy evaluation
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/<your-username>/word2vec-skipgram-ns.git
   cd word2vec-skipgram-ns
   ```
2. **Create a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

1. **Download** the text8 dataset into the `data/` directory:

   ```bash
   wget http://mattmahoney.net/dc/text8.zip -P data/
   unzip data/text8.zip -d data/
   ```
2. **Run preprocessing** to generate tokens and vocabulary:

   ```bash
   python src/preprocess.py --input data/text8 --output data/processed.pkl --min-count 5
   ```

## Training the Model

Train the Skip-gram model with negative sampling:

```bash
python src/train.py \
  --data-path data/processed.pkl \
  --embed-dim 100 \
  --window-size 5 \
  --neg-samples 5 \
  --batch-size 512 \
  --epochs 5 \
  --learning-rate 0.025 \
  --save-path models/sg_ns_embed.npy
```

## Evaluation

* **2D Visualization**:

  ```bash
  python src/evaluate.py --mode visualize --embed-path models/sg_ns_embed.npy --output figures/embeddings_2d.png
  ```
* **Analogy Tests**:

  ```bash
  python src/evaluate.py --mode analogy --embed-path models/sg_ns_embed.npy --questions data/questions-words.txt
  ```

## Hyperparameter Impact Discussion

* **Embedding Dimension**: Larger dimensions capture finer nuances but may overfit on small corpora and increase computation.
* **Context Window Size**: Larger windows capture broader context (semantic similarity), smaller windows focus on syntactic relationships.
* **Negative Samples**: More negative samples can improve embedding quality but slow down training.

## Results Summary

* Achieved semantic analogies with high accuracy on the `questions-words.txt` benchmark.
* Visual clusters observed for related word groups (e.g., countries, royalty terms).

## Requirements

* Python 3.7+
* NumPy
* matplotlib
* scikit-learn
* tqdm

## License

This project is licensed under the MIT License. Feel free to use and modify!
