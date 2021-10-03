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
