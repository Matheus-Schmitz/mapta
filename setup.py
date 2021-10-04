from setuptools import setup, find_packages
#import subprocess

classifiers = [
	'Operating System :: OS Independent',
	'License :: OSI Approved :: MIT License',
	'Programming Language :: Python :: 3'
	]

setup(
	name='mapta',
	version='0.0.08',
	description='Classify a text as belonging to LGBTQ+ commuty and drug user community.',
	py_modules=["Mapta",  "Pytorch_NN"],
	author = 'Matheus Schmitz',
	author_email = 'mschmitz@usc.edu',
	packages = find_packages(),
	include_package_data = True,
	classifiers = classifiers,
	install_requires=[
          'numpy >= 1.18.0, < 1.23.0',
          'pandas >= 1.3.0, < 1.4.0',
          'torch >= 1.8.0, < 1.9.0',
          'scikit-learn >= 0.24.0, < 1.0.0',
          'nltk == 3.5',
          'tqdm > 4.50.0',
          'gdown == 3.10.1',
          #'sent2vec @ git+https://github.com/epfml/sent2vec.git@v1#egg=sent2vec',
          'sent2vec @ git+https://github.com/epfml/sent2vec.git',
          #'sent2vec @ git+ssh://git@github.com/example_org/sent2vec.git'
          #'sent2vec==0.0.0'
      ],
     #dependency_links = ['http://github.com/epfml/sent2vec/tarball/master#egg=sent2vec']
	)

#if name == "__main__":cd 
#subprocess.run(["pip", "install", "git+https://github.com/epfml/sent2vec.git#egg=sent2vec"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)