# MAPTA: MAP's Text Analyzer 

MAP is a USC DSCI 560 group doing research on identifying marginalized minorities, namely gender minorities and drug user minorities. We developed this model so as to allow public policy experts and researchers to easily identify both groups with a single pretrained model. This is highly relevant as those who belong to both minorities are at special risk for adverse health outcomes given exponential combination of the prejudices faces individually by each group. 

## Installation

Attention: this package relies on a pre-trained embedding model from sent2vec, with a size of 5 GB. The model will be automatically downloaded when the package is first instanced on a python script, and will then be saved on the package directory for future usage.

1. Clone this repo:
```
git clone https://github.com/Matheus-Schmitz/mapta.git
```
2. Go to the repo's root:
```
cd mapta/
```
3. Install with pip:
```
pip install .
```

## Sample usage

Note that the model outputs a list of [LGBT score, Drug score]

```python
from mapta import Mapta


if __name__ == "__main__":

	text = "This is the text for which I'm trying to generate a prediction regarding those marginalized minorities."

	# Predict
	model = Mapta.MAPTA()
	output = model.predict(text)

	# Results
	print(f'LGBT score: {output[0]}')
	print(f'Drug score: {output[1]}')
```

## Utilizing Predictions

MAPTA was trained on a dataset of 360k posts about drugs, 118k posts about LGBTQ+ and 580k posts about a variety of subreddits covering common discourse topics that serves as a negative class.  

A validation set that had been held out during training was then used to optimize the prediction threshold based on F1 Score. This is done because an analysis of the prediction probability curves releaved the models to be much more confident in their predictions of the negative class than in predicting the positive class. Further analysis indicates that this mostly likely is caused by the fact that both Drugs and LBTQ posts are still composed of 80-90% common discourse words, with only a minority of specific words releaving the true subject of the post. Thus, given the variability in this small percentage of descriminative words, we observe a wide range of assigned scores for positive class posts.

LGBT:
![LGBT Predicted Class Probabilities](/images/lgbt_predicted_class_probabilities.png?raw=true)

Drugs:
![Drugs Predicted Class Probabilities](/images/drugs_predicted_class_probabilities.png?raw=true)

After optimizing the prediction threshold so as to maximize F1 Scores, the following optimal thresholds are found, and the improvement in Confusion Matrix is displayed alongside. Confusion Matrix is measured on a test dataset which was not utilized during threshold optimization.

LGBT:
![LGBT F1 Score Optimization](/images/lgbt_F1_vs_thresholds.png?raw=true)
![LGBT Confusion Matrix](/images/lgbt_threshold_optimization_confusion_matrix.png?raw=true)

Drugs:
![Drugs F1 Score Optimization](/images/drugs_F1_vs_thresholds.png?raw=true)
![Drugs Confusion Matrix](/images/drugs_threshold_optimization_confusion_matrix.png?raw=true)

### Classification Recommendations

As a result of this analysis, we recommend that users utilize the following thresholds to classify samples:

LGBT: 0.2015

Drugs: 0.1041

## Results Analysis

To assess MAPTA's performance we utilize the model to generate predictions on the test dataset, then select only high confidence predictions for out target class, which is people who belong to BOTH the LGBTQ+ community and the Drug User community. To select those users we multiply both scores and then classify only users whose metric is above 0.9^2 = 0.82. We constrast those users with a set of control users whose combined metric is below 0.1^2 = 0.01. Below we present the results of our analysis that confirms the models proper functioning.

![Word Frequencies](/images/Word_Frequencies.png?raw=true "Word Frequencies")  
  
![Affect](/images/Affect.png?raw=true "Affect")  
  
![TF-IDF](/images/TF_IDF.png?raw=true "TF-IDF")  