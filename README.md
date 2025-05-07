# Word2Vec Skip-gram with Negative Sampling

## Project Overview

This repository contains an implementation of the Skip-gram model with Negative Sampling for learning word embeddings, trained on the [text8 dataset](http://mattmahoney.net/dc/text8.zip). Word embeddings capture semantic and syntactic properties of words in dense vector representations.

## Repository Contents

* `Word Representation.ipynb`: Jupyter notebook containing all code for data preprocessing, model implementation, training, and evaluation.
* `text8.zip`: Compressed text8 dataset.
* `README.md`: Project documentation (this file).

## Notebook Structure

The `Word Representation.ipynb` notebook includes:

1. **Data Preprocessing**: Downloading and extracting `text8.zip`, cleaning, tokenization, and vocabulary creation.
2. **Dataset Generation**: Constructing positive skip-gram pairs and sampling negative examples.
3. **Model Definition**: Implementing the Skip-gram architecture with Negative Sampling loss using NumPy.
4. **Training**: Training loop with configurable hyperparameters such as embedding dimension, context window size, number of negative samples, batch size, epochs, and learning rate.
5. **Evaluation**:

   * **2D Visualization**: Projecting embeddings via SVD and plotting in 2D.
   * **Analogy Tests**: Performing word analogies (e.g., “king” - “man” + “woman” ≈ “queen”).
6. **Hyperparameter Discussion**: Analysis of how embedding dimension, window size, and negative sample count affect embedding quality.

## Hyperparameter Configuration

Within the notebook, you can adjust:

* **Embedding Dimension**: Typically 50–300.
* **Context Window Size**: Typically 2–5.
* **Negative Samples**: Typically 5–20.
* **Learning Rate**: Starting around 0.025, decaying over time.
* **Batch Size** and **Epochs**.

## Requirements

* Python 3.7+
* NumPy
* matplotlib
* scikit-learn
* tqdm

## Results Summary

* Semantic analogies achieved high accuracy on the `questions-words.txt` benchmark.
* Clear clustering in 2D visualizations for related word groups (e.g., countries, royalty terms).

## License

This project is licensed under the MIT License. Feel free to use and modify!
