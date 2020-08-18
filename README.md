# AES

[NOTE] : This is the course project for cs7650. If you are looking for detailed deep learning implementation for this project please refer to this repo:https://github.com/Gaurav-Pande/AES_DL

Course project for CS 7650 NLP (Georiga Tech)

Project presentation video: https://youtu.be/ZQ4pyrMSCgg

Project presentation link: https://docs.google.com/presentation/d/1ukMXqZSapifteOXo2_41qRhCeIfxrqMVGWkcocSn79I/edit?usp=sharing

### ASAP Dataset

We used the Automated Student Assessment Prize (ASAP) datasetby The Hewlett Foundation.  (Hewlett, 2012:  ac-cessed March 12, 2020) 
This dataset consists of essays written by students from 7th - 10th grade. The essays are divided into 8 sets. Each set has a prompt associated with it. There are 2 types of prompts Type 1: Persuasive / Narrative / Expository Type 2: Source Dependent Responses. The first type of prompt asks students to state their opinion about certain topic.  The second type of prompt has a required reading associated with it and the students are expected to answer a question basedon their understanding of this reading. Different prompts have been graded by different number of graders. But each set has a domain 1 score, which is the combined total score.


### Repository Structure

The repository has been structured into 3 main folders:<br>
1. aes_ml : Machine Learning based approaches to AES (Parth) 
2. aes_dl : Deep Learning based approaches to AES (Gaurav)
3. gender_age_models: Bias analysis in grading (Naila)


### Installing Requirements necessary:

In order to install all the libraries needed to run this project, please run requirements.txt using:

```
pip install -r requirements.txt
```

### AES using Machine Learning (aes_ml)

<br>

![image](./figs/aes_ml.png)



This folder has 2 notebooks that have the results. `aes_ml_exploration.ipynb` is the notebook that has all the exploration we did. It includes hyperparameter tuning along with the results. `aes_ml_results.ipynb` has the final results that were presented in the paper. Hence to get the final results from the paper please **run all cells in `aes_ml_results.ipynb`**. 

Please note that some results that involve word2vec based features might be slightly different due to various randomness factors. The `aes_ml_results.ipynb` notebook currently has the output values that were reported in the final paper.

Besides these notebooks the other code for data preprocessing, feature extraction, data frame generation and training has been split in to various python files.


**1. `feature_extraction.py`:** Contains code to process essays and wordlists to generate various style and content based features.
- `sentence_to_word_list(sentence, ignore_stopwords, set_no)` is used to generate list of words and calculate word level style based features. E.g.: Number of stop words, number of words with greater than 5 characters, etc.
- `essay_to_sentences(essay, set_no, ignore_stopwords=False)` is used to convert an essay to list of sentences and pass those to `sentence_to_word_list`. It is also used to generated sentence level style based features, E.g.: number of sentences. This function also sums the word level features that `sentence_to_word_list` returns. 
- `get_avg_word_vec(words, model, wv_size)` is used to return the average word vector representation of a list of words. 
- `get_prompt_word_vecs(prompt, model, wv_size)` is used to get the average word vector representation of paragraphs from the story for type 2 prompts. 
- `get_word_count_vector(words, set_no)` is used to get a sparse word count vector representation using `word_to_index` maps.
- `get_prompt_vectors(prompt, set_no)` is used to get sparse word count vectors for paragraphs from the story for type 2 prompts.


**2. `frame_maker.py`:** Contains the code to generate pandas dataframes that will be used in training the models.
- `makeDataFrame(data)` makes the features dataframe from the dataframe that contains essay data loaded from ASAP's csv.
- `split_in_sets(data, base_min_scores, base_max_scores)` is used split the features dataframe by set number of the essays and display other information about each sets like prompt type.


**3. `keys.py`:** Contains of all the keys that are used to access dataframe columns


**4. `prompts_reader.py`:** Contains the code to load paragraphs from story for type 2 prompts from the `./prompts` directory.


**5. `trainer.py`:** Contains the code to train ML models and evaluate the MQWK score accross 5 folds.
- `get_all_classifiers()` is used to get a list of classifiers that was used in the exploration phase before Linear Regression was chosen.
- `create_word_vecs(X, wv_model, wv_size, essay_wordvecs)` is used to generate the average word vector representations for word from essays in a dataframe.
- `create_sim_from_word_vecs(X, data, wv_model, wv_size, essay_wordvecs)` is used to generate similarity between the essay's average word vector representation based with average word vector based representation of paragraphs from story of type 2 prompt.
- `evaluate(X, y, data=None, model = LinearRegression(), plot=False, wordvec=False, wv_size=300, min_count=30, context=10, sample=0, lsa=False, wordvec_sim=False)` is used to train the machine learning model across 5 fold cross validation and compute the value of MQWK score.

Besides these files, all the imports have been defined in python files that have been named as `requirements_*`. 

**1. `requirements_base.py`:** Contains all the basic library imports.

**2. `requirements_feature.py`:** Contains imports for functions defined in `feature_extraction.py`.

**3. `requirements_frame.py`:** Contains  imports for functions defined in `frame_maker.py`.

**4. `requirements_key.py`:** Contains imports for dictionarys and keys defined in `keys.py`.

**5. `requirements_trainer.py:`** Contains imports for functions defined in `trainer.py`.


###  AES using Deep Learning (aes_dl)

The code in this section is exploration of different deep learning techniques on individual sets and on whole dataset.

![image](./figs/arch.png)


 
#### So what did we try:
 
The approaches tried in DL are:
1. Try 3 different architecture involving LSTM, BiLSTM, and CNNs on individual sets and on whole 
dataset separately. 
2. Use Word2vec and Bert embeddings for feature vector representation.
3. Hyperparameter tunning to optimize the loss and increase the mean QWK.
 
Currently the models were trained in keras(tensorflow as backend).

#### Prerequisites

* Python 3+
* compute as it takes around 4-5 hours to run all the models and approaches.


#### Installation

I would recommend using google collab or better if you have GPU access. If you are running this locally then
follow the instructions:

* Go the directory aes_dl and run all the following steps from that directory:

```

cd ./NLP-Project-AES/aes_dl

```

* Install virtual environment using:

```shell script
pip install virtualenv
```

* Create a virtual environment using:

```shell script

virtualenv aes

```

* Activate virtual environment

```shell script

source aes/bin/activate

```

* Install requirements from requirements.txt

```shell script

pip install -r requirements.txt

```

####  Training the models

* To train the model using BERT, first change the hyperparameters in the train_{BERT/word2vec}_{sets/all}.py file
* once you have changed the hyperparameters, run the respective file for training. For example

Using BERT and train on per set, run:

```shell script
python train_bert_sets.py
```  

*  Using BERT and train on whole data set, run:

```shell script
python train_bert_all.py
```  



*  Using WORD2VEC and train on whole dataset, run:

```shell script

python train_word2vec_all.py

```  


#### Note:

* If you would like to run the Notebook then you can directly open the AES.ipynb, and run cell by cell,
but the results there may vary.
* There is a PDF of the Notebook attached to see the results we got after training the models.


[Future Work in AES DL]:

* add commmand line parameters for passing hyperparameters.
* add pytorch support.
* add GPU support for models.
* add more extensive hyperparameters.
* add sigmoid activation.
* Topic modelling using LDA.
* Visualization for topic modelling.



### Bias analysis in grading (gender_age_models)
**1. `feature_extraction.py`:** This file contains code for extracting features like: possessive features, POS unigrams, f-measure scores, POS bigrams, n-gram character and word level features, sentiment features and text-readability features. Also contains functions to find the dataset summaries for gender and age.

**2. `preprocess.py`:**  Contains code to preprocess the data before feature extraction.

**3. `requirements.py`:** Imports all modules required for executing the models.

**4. `visualize.py`:** Contains functions which allow us to visualize features.

**5. `age_pentel_model.sav`:** Age prediction model (SVM trained with text-readability features)

**6. `ngram_model.sav and ngram_classifier_model.sav`:** Gender prediction model (Naive Bayes trained with n-grams)

**7. `ngram_char_model.sav`:** Model to extract character level n-grams (optional)

**8. `author_profiling.ipynb`:** Entire code (run cell by cell). This has been split into the next 2 notebooks. Recommended

**9. `gender_models.ipynb`:** Gender prediction models. Shows performance on generalization dataset and analysis on ASAP dataset.

**10. `age_models.ipynb`:** Age prediction models. Shows analysis on ASAP dataset.

**To use gender_models.ipynb:** Run cell by cell
Open the file in Google Colab.

Upload the following: requirements.py, preprocess.py, visualize.py, feature_extraction.py

To train the models: Upload train.json (https://drive.google.com/file/d/1rTiQQHkEAyf7of6iZ8GRXHA_IQgGsnjr/view?usp=sharing)

To check if the model generalizes well: Upload blog-gender-dataset.csv (present in the Dataset folder). Upload ngram_model.sav and ngram_classifier_model.sav

To check model's performance on HP dataset: Upload training_set_rel3.tsv (in asap-aes folder). Upload ngram_model.sav and ngram_classifier_model.sav

**To use age_models.ipynb:** Run cell by cell
Open the file in Google Colab.

Upload the following: requirements.py, preprocess.py, visualize.py, feature_extraction.py

To train the models: Upload train.json (https://drive.google.com/file/d/1rTiQQHkEAyf7of6iZ8GRXHA_IQgGsnjr/view?usp=sharing)

To check model's performance on HP dataset: Upload training_set_rel3.tsv (in asap-aes folder). Upload age_pentel_model.sav
