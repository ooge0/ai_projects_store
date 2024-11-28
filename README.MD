# Project overview

Project contains implementations some of AI models, some basic structures of neural networks.

**Projects list**

1. Model name: LSTM
   1. project source: [lstm](scr/emotion_recognition/lstm/lstm.py)
   2. implementation of: Long short-term memory network
   3. dependencies: pandasб tensorflow, sklearn
2. Model name: LSTM     
   1. project source: [lstm_tuning](scr/emotion_recognition/lstm_tuning),
   2. implementation of: convolution neural network
   3. dependencies:
3. Model name: 
   1. project source: [distil_bert](scr/emotion_recognition/distilbert/distilbert.py)
   2. implementation of: convolution neural network
   3. dependencies:
4. 
   1. project source: [text_cnn](scr/emotion_recognition/textcnn)
   2. implementation of: convolution neural network
   3. dependencies: 
5. README.MD : [text_cnn README](scr/emotion_recognition/README.md)

6. model: perceptron
   1. project name: [perceptron_by_numpy](scr/perceptron_by_numpy)
   2. implementation of: Perceptron
   3. dependencies: numpy
   2. README.MD : [LSTM project README](scr/perceptron_by_numpy/README.md)

8. model: LLM
   1. project name: [prompt_evaluation_guide](scr/prompt_evaluation_guide)
   2. implementation of: Perceptron
   3. dependencies: transformers datasets torch matplotlib seaborn
   4README.MD : [LSTM project README](scr/emotion_recognition/README.md) 
      

## Virtual environment configuration

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

## Other installation/configuration staff
### Check installed version of python package
For checking version of installed python package use 
```
pip freeze | findstr {desiered_package_name}
```  
e.g. for `transformers` use
```
pip freeze | findstr transform
```
or for `tensorflow`
```
pip freeze | findstr tensorflow
```


