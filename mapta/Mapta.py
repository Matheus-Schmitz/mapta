# Custom Pytorch NN
import Pytorch_NN
import mapta

# Py Data Stack
import numpy as np
import pandas as pd

# Machine Learning
import torch
from sklearn.preprocessing import MinMaxScaler

# File Manipulation
from glob import glob
import joblib
import os
import gdown

# NLP
import sent2vec
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re


class MAPTA():

	def __init__(self):

		# Define package path
		self.package_path = os.path.dirname(mapta.__file__)

		# Install Sent2Vec model if needed
        if "wiki_unigrams.bin" not in os.listdir(self.package_path):
            print("Downloading sent2vec model...")
            url = 'https://drive.google.com/u/0/uc?id=0B6VhzidiLvjSa19uYWlLUEkzX3c'
            output = self.package_path + '/wiki_unigrams.bin'
            gdown.download(url, output, quiet=False)

        # Load Sent2Vec model
        self.sent2vec_model = sent2vec.Sent2vecModel()
        self.sent2vec_model.load_model('/' + self.package_path + '/' + 'wiki_unigrams.bin')


        # Set stopwords
        self.stop = stopwords.words('english')
        self.stop += ['Nan', 'NaN', 'nan', 'removed', 'deleted', "'m", 've', 're', 'wa']

        # Lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Setting the device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # LGBT model
        self.model_lgbt = Pytorch_NN.PytorchNN(num_features=timeseries_features.shape[1], 
                                               learning_rate=1e-4, 
                                               optimizer=torch.optim.AdamW, 
                                               loss_fn=nn.BCELoss(), 
                                               device=device)
        self.model_lgbt.load_state_dict(torch.load(self.package_path + '/' + 'model_lgbt.pt', map_location=torch.device(device)))
        self.model_lgbt.eval()
        self.model_lgbt = model_lgbt.to(device)

        # Drug model
        self.model_drug = Pytorch_NN.PytorchNN(num_features=timeseries_features.shape[1], 
                                               learning_rate=1e-4, 
                                               optimizer=torch.optim.AdamW, 
                                               loss_fn=nn.BCELoss(), 
                                               device=device)
        self.model_drug.load_state_dict(torch.load(self.package_path + '/' + 'model_drug.pt', map_location=torch.device(device)))
        self.model_drug.eval()
        self.model_drug = model_drug.to(device)

        # Scalers
        self.scaler_lgbt = joblib.load(self.package_path + '/' + 'scaler_lgbt.joblib') 
        self.scaler_drug = joblib.load(self.package_path + '/' + 'scaler_drug.joblib') 


    # Function to clean stopwords
    def clean_sentence(self, sentence):
        clean = ' '.join([re.sub(r'[^\w\s]','',word.strip()) for word in sentence.split() if re.sub(r'[^\w\s]','',word.strip()) not in self.stop]).lower()
        clean = ' '.join([self.lemmatizer.lemmatize(word) for word in clean.split() if len(word)>1 and 'http' not in word])
        if clean != 'nan':
            return clean

    def predict(self, text):
        text = clean_sentence(text)
        text.replace('', np.nan, inplace=True)
        text.dropna(inplace=True)
        embeddings = sent2vec_model.embed_sentences(text.values)

        lgbt_score = self.model_lgbt.predict_proba(embeddings)
        drug_score = self.model_drug.predict_proba(embeddings)

        return [lgbt_score[1], drug_score[1]]