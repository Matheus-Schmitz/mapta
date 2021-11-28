# MAPTA
from mapta import Mapta

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Natural Langauge Processing
from nrclex import NRCLex
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Data Manipulation
import pandas as pd
import numpy as np

# IPython
from ipywidgets import FileUpload
from IPython.display import display

# Set stopwords
stop = stopwords.words('english')

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to clean stopwords
def clean_sentence(sentence):
    clean = ' '.join([re.sub(r'[^\w\s]','',word.strip()) for word in sentence.split() if re.sub(r'[^\w\s]','',word.strip()) not in stop]).lower()
    clean = ' '.join([lemmatizer.lemmatize(word) for word in clean.split() if len(word)>1 and 'http' not in word])
    if clean != 'nan':
        return clean
    
# Function to extract affects in parallel
def get_affect(text):
    affect_dict = NRCLex(text).affect_frequencies
    if 'anticipation' not in affect_dict:
        affect_dict['anticipation'] = np.NaN
    return affect_dict
    
def getSingleOuput(text): 
    # Predict
    output = model.predict(text)
    output_df = pd.DataFrame([["LGBT", output[0]*100], ["Drug", output[1]*100]], columns = ["Group", "Probability"])
    sns.barplot(x = 'Group', y = 'Probability', data = output_df, hue = "Group")
    plt.ylim(0, 100)
    plt.title("Predicted Probability Distributions")
    # Show the plot
    plt.show()
    print('''Prediction Results:\n
    LGBT group with proability {:.6f}%\n
    Drug group with probability {:.6f}%'''.format(output[0]*100, output[1]*100))
    # Preprocess the text
    
    print("\n\nSentiment and Emotional Analysis")
    clean_text = clean_sentence(text)

    # Extract emotions and sentiments from text
    affects = get_affect(clean_text)

    # Convert affects to DataFrame for easier plotting
    df_affects = pd.DataFrame.from_dict(affects, orient='index', columns=['scores'])
    # Split emotions and sentiments
    df_emotions = df_affects.filter(['positive', 'negative'], axis=0)
    df_sentiments = df_affects.filter(['fear', 'anger', 'trust', 'surprise', 'sadness', 'disgust', 'joy', 'anticipation'], axis=0)

    # Calculate relative percentages
    df_emotions['percentage'] = df_emotions['scores']/df_emotions['scores'].sum()
    df_sentiments['percentage'] = df_sentiments['scores']/df_sentiments['scores'].sum()
    fig, ax = plt.subplots(ncols=2, figsize=(16,9), gridspec_kw={'width_ratios': [1, 4]})

    fig.sca(ax[0])
    sns.barplot(data=df_emotions, x=df_emotions.index, y='percentage', color='gold', alpha=0.8)
    ax[0].yaxis.set_major_locator(mtick.MultipleLocator(0.2))
    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    plt.xticks(fontsize=25, rotation=30)
    plt.yticks(fontsize=20)
    plt.xlabel(None)
    plt.ylabel('', fontsize=30, weight='normal', labelpad=20)
    plt.title('Emotions', size=30, weight='normal')

    fig.sca(ax[1])
    sns.barplot(data=df_sentiments, x=df_sentiments.index, y='percentage', color='deepskyblue', alpha=0.8)
    ax[1].yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    plt.xticks(fontsize=25, rotation=30)
    plt.yticks(fontsize=20)
    plt.xlabel(None)
    plt.ylabel('', fontsize=30, weight='normal', labelpad=20)
    plt.title('Sentiments', size=30, weight='normal')

    fig.align_xlabels(ax)
    fig.tight_layout()
    plt.show()

def parseContent(content): 
    content = content.decode("utf-8")
    sentences = content.split("\n")
    return sentences

def getAnalysis(sentences): 
    """
    Input: a list of sentences 
    Output: anlysis result in a dataframe
    """
    ml_predictions = []
    sentiment_predictions = []
    for sent in sentences: 
        pred = model.predict(sent)
        ml_predictions.append(pred)
        clean_text = clean_sentence(sent)
        affects = get_affect(clean_text)
        sentiment_predictions.append(affects)
    
    ml_predictions = pd.DataFrame(ml_predictions, columns = ["LGBT", "drug"])
    sentiment_predictions = pd.DataFrame(sentiment_predictions)
    sentences = pd.DataFrame(sentences, columns = ["text"])
    output = pd.concat([sentences, ml_predictions, sentiment_predictions], axis = 1)
                        
    return output.copy()

def batchUpload(): 

    for uploaded_filename in upload.value:
        print("Upload file {}".format(uploaded_filename))
        content = upload.value[uploaded_filename]['content'] 
        break
    print("Generating output")
    sentences = parseContent(content)
    output = getAnalysis(sentences)
    output["anticipation"] = output["anticipation"].fillna(0)
    print("Done!")
    output.to_csv("output.csv")
    lgbt_threshold = 0.2015
    drug_threshold = 0.1041
    num_lgbt = len(output[output["LGBT"]>=lgbt_threshold])
    num_drug = len(output[output["drug"]>=drug_threshold])
    agg = pd.DataFrame([["LGBT", num_lgbt], ["Drug", num_drug]], columns=["Group", "Number"])
    sns.barplot(data=agg, x="Group", y='Number', alpha=0.8)
    plt.title("Number of Samples in LGBT and Drug Groups")
    plt.show()
    print('''There are {} samples in the uploaded file
    \n   {} samples are classified as the LGBT group with threshold {}
    \n   {} samples are classified as the Drug group with threshold {}'''
         .format(len(output), num_lgbt, lgbt_threshold, num_drug, drug_threshold))
    return output
