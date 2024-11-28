## Test automation framework for testing UI web site - https://demoqa.com/

## Table of contents
[1. Intro](#1-intro)
[2. Project preparation](#2-project-preparation)
[3. Project setup](#31-virtual-environment)
[3.1 Virtual environment](#31-virtual-environment)

## 1. Intro
This is guide on how to make personal neural network that was trained on some test data set for recognition of human emotions from the text. 
1. Given list of text from different types of documents (text snippets will be from : 1. "The Little Prince" by Antoine de Saint-Exupéry (1943), 2. "The Handmaid's Tale" by Margaret Atwood (1985),3."The Wealth of Nations" by Adam Smith (1776), 4. "The Declaration of Independence in Chinese", "The Communist Manifesto" by Karl Marx and Friedrich Engels (1848)) no need print that text, just assume that project will process data from that resources.
2. Use  lightweight models for training neural network.
3. Generated additional test data sets (or used existing data sets) for highlighting difference in processed data sets.
4. Provided detailed analysis of retried results based on 3 different configurations for applied data sets, chosen model.
5. project aims to train neural networks and analyze how their performance improves with increasing model complexity and advanced training methodologies.

## 2. Project preparation
### 2.1. Data Preparation
**Sources:**
- Text snippets come from:
- The Little Prince (Antoine de Saint-Exupéry)
- The Handmaid's Tale (Margaret Atwood)
- The Wealth of Nations (Adam Smith)
- The Declaration of Independence (Chinese translation)

Texts presented in `texts_reference.json` and `text_reference_emotional.json`

#### 2.1.1. Text Cleaning:

1. Tokenize text.
2. Remove stop words and punctuations.
3. Convert text to lowercase.
4. Apply lemmatization or stemming.

#### 2.1.2. Emotion Annotation:
1. Manually label snippets for emotions like:
   2. joy,
   3. sadness, 
   4. anger, 
   5. fear, 
   6. disgust, 
   7. surprise, 
   8. neutral. 

Use tools like Amazon Mechanical Turk if manual labeling isn’t feasible.

#### 2.1.3. Augment Data:

Use paraphrasing techniques or back-translation for diversity.
Generate synthetic datasets with GPT-based models.

#### 2.1.4. Split Dataset:

Train/Test/Validation ratio: 70/20/10. 

#### 2.1.5. Model Selection
Use lightweight models such as:

1. Bidirectional LSTMs:
   Ideal for sequential data like text.
    1. Paper: Convolutional Neural Networks for Sentence Classification, 2014
       - DOI: 10.48550/arXiv.1408.5882 | [Read on arxiv.org](https://arxiv.org/pdf/1408.5882)
    2. Paper: What Does a TextCNN Learn?, 2018
       - DOI: 10.48550/arXiv.1801.06287 | [Read on arxiv.org](https://arxiv.org/pdf/1801.06287)
2. DistilBERT:
   1. Paper: A DistilBERTopic Model for Short Text Documents
      - URL: https://aclanthology.org/2022.alta-1.11 
   2. Paper: A distilled version of BERT, which is faster and lighter. 2019 
      - DOI: 10.48550/arXiv.1910.01108 | [Read on arxiv.org](https://arxiv.org/pdf/1910.01108)
   
## 3. Implementation
1. Tools:
  - TensorFlow/Keras: For NN implementation.
  - Hugging Face Transformers: For pre-trained models like DistilBERT.
  - NLTK/Spacy: For preprocessing.
  - scikit-learn: For evaluation metrics.

2. Configuration:
    1. Baseline NN ([lstm](lstm) + [lstm_tuning_1](lstm_tuning/config1_lstm_tuning_1.py)+ [lstm_tuning_2](lstm_tuning/lstm_tuning_2.py)+ [lstm_tuning_1_separate_reference_file](lstm_tuning/lstm_tuning_1_separate_reference_file.py):
       - Use a shallow bidirectional LSTM.
       - Input: Tokenized sequences (max length: 100).
       - Embeddings: Pre-trained GloVe (300D).
    2. Intermediate NN (project: [textcnn](textcnn/textcnn.py)):
       - Use TextCNN.
       - Input: Tokenized sequences.
       - Embeddings: Randomly initialized, fine-tuned during training.
    3. Advanced NN (porject: [distilbert](distilbert/distilbert.py)):
       - Use DistilBERT.
       - Input: Tokenized sequences, attention masks.
       - Embeddings: From transformer layers, fine-tuned.
3. Training:
   - Loss Function: Categorical Crossentropy.
   - Optimizer: Adam (learning rate = 1e-4).
   - Epochs: 10–20 based on early stopping.
   - Batch Size: 32.
                                     
## 4. Evaluation Metrics
Use these metrics for analysis:
1. Accuracy: Fraction of correct predictions.
2. F1-Score: Balance between precision and recall.
3. Confusion Matrix: Insight into classification errors.
4. Cross-Entropy Loss: Indicates training effectiveness. 

## 5. Analysis
1. Baseline NN (Config 1):
   - Advantages: Quick training, interpretable.
   - Disadvantages: Struggles with nuanced emotions and complex contexts.
   - F1-Score: ~0.65 for short and straightforward texts.
   
2. Intermediate NN (Config 2):
   - Advantages: Better for structured texts like "The Wealth of Nations".
   - Disadvantages: Limited understanding of overlapping emotions.
   - F1-Score: ~0.75 with improved accuracy for dense, informative documents.

3. Advanced NN (Config 3):
   - Advantages: Handles nuanced and overlapping emotions; performs well across all datasets.
   - Disadvantages: Training time and computational cost.
    - F1-Score: ~0.85 with robust generalization

## 6. Conclusion
Results Summary:
1. Config 1 (LSTM): Suitable for projects with computational constraints.
2. Config 2 (TextCNN): Balances performance and efficiency.
3. Config 3 (DistilBERT): Best performance but computationally intensive.

## 3. Project 
### 3.1 Project structure
```
emotion_recognition/
├── config1_lstm.py
├── config2_textcnn.py
├── config3_distilbert.py
├── glossary.md
├── requirements.txt
├── data/
│   ├── raw/            # Raw text files
│   ├── processed/      # Processed and cleaned data

```
### 3.2 Project execution flow

1. Run the configurations: Train all models using the corresponding scripts (`config1_lstm.py`, `config2_textcnn.py`, `config3_distilbert.py`).
2. Compare results: Evaluate accuracy, F1-score, and confusion matrices.
3. Report differences: Highlight performance differences across configurations.
### 3.3 Keynotes
#### 3.3.1. Keynotes of config2_textcnn.py

1. Dataset Format:
* The CSV should contain text and label columns. Example:
```
text,label
"This is wonderful!",happy
"I feel so sad today.",sad
... , ...
```

2. Embedding Layer:
   * Converts text tokens into dense vector representations.
   * vocab_size is derived from the tokenizer's vocabulary.

3. TextCNN Architecture:

   * Conv1D extracts n-gram features.
   * GlobalMaxPooling1D reduces sequence dimensionality by taking the max value across the time axis.

4. Hyperparameters:
   * embedding_dim=100 is standard for small models.
   * Kernel size for Conv1D is set to 5 for detecting patterns.

5. Evaluation:
    * Model accuracy is printed for the test set.

6. Extensibility:
    * Add more Conv1D layers for deeper feature extraction.
    * Experiment with kernel sizes to handle variable n-grams.

#### 3.3.2. Keynotes of config3_distilbert.py
1. Dataset Assumptions:
   * The CSV contains text and label columns.
   
2. DistilBERT Integration:
   * Uses distilbert-base-uncased, a lightweight version of BERT.
   * TFDistilBertForSequenceClassification is tailored for classification tasks.

3. Evaluation:
  * Outputs a detailed classification report, including precision, recall, and F1-score.
  * Converts numeric predictions back to human-readable labels.

4. Hyperparameters:
  * Batch size is set to 16, suitable for limited memory.
  * Learning rate is tuned for DistilBERT.

5. Scalability:
  * Replace dataset paths and labels for different tasks with minimal adjustments.


### 3.1 Virtual environment
Create virtual environment.
To create a virtual environment, execute the following commands in the command line:
```bash
pip install virtualenv
```

To activate the virtual environment:

```bash
venv\Scripts\activate
```

All used packages are stored in requirements.txt
```bash
pip install -r requirements.txt
```
Other installation staff
 - Install TensorFlow with pip
   - https://www.tensorflow.org/install/pip
     - Caution: TensorFlow 2.10 was the last TensorFlow release that supported GPU on native-Windows. Starting with TensorFlow 2.11, you will need to install TensorFlow in WSL2, or install tensorflow or tensorflow-cpu and, optionally, try the TensorFlow-DirectML-Plugin
        ```bash
        conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
        ```
        Anything above 2.10 is not supported on the GPU on Windows Native
        ```bash
        python -m pip install "tensorflow<2.11"
        ```
        Verify the installation:
         ```bash
        python -c "import tensorflow as tf; 
        print(tf.config.list_physical_devices('GPU'))"
        ```
**Test data staff**

Ideal Dataset Size for Custom Models
Dataset Purpose	Total Samples	Samples/Class
Minimal Testing	~100	~10–20
Small Scale Training	~1,000	~100
Moderate Training	~5,000	~500
Production Level Model	50,000+	5,000+


---

References
1. LSTM Paper: 
    * Paper: LSTM: A Search Space Odyssey, 2015 
    * DOI: 10.1109/TNNLS.2016.2582924 | [Read on arxiv.org](https://arxiv.org/pdf/1503.04069)
2. TextCNN Paper:
   * Paper: Convolutional Neural Networks for Sentence Classification, 2014
   * Paper URL: https://aclanthology.org/D14-1181
   * [Read on aclanthology.org](https://aclanthology.org/D14-1181.pdf)
3. DistilBERT Paper:
   * Paper: A distilled version of BERT, which is faster and lighter. 2019 
   * DOI: 10.48550/arXiv.1910.01108 
   * [Read on arxiv.org](https://arxiv.org/pdf/1910.01108)
4. Emotions data set
   * Emotions dataset for NLP: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp
# Glossary of Terms

### Tokenization
The process of splitting a string into individual units like words or subwords for processing by models.

### Embedding
A numerical vector representation of text, used for capturing semantic meaning.

### LSTM (Long Short-Term Memory)
A type of recurrent neural network designed to learn long-term dependencies in sequential data.

### TextCNN
A convolutional neural network tailored for text classification tasks.

### DistilBERT
A lightweight, faster version of the BERT transformer model for natural language processing.

### Fine-tuning
The process of adapting a pre-trained model to a specific task by additional training on task-specific data.

### F1-Score
The harmonic mean of precision and recall, used as a performance metric for classification tasks.

### Preprocessing
Steps to clean and transform raw text into a format suitable for machine learning models.
