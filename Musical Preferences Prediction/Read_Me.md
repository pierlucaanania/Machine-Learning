✋Hello everyone!✋
We are glad to see you here in our in-class competition for Peter the Great St. Petersburg Polytechnic University 
summer school "Machine Learning: Theory and Application" course.

--Competition details--
I've collected the dataset of my music preferences from yandex.music account and suggest you to create the model, which will predict my preferences. 
It is the simple binary classification task, so you have to predict whether I like👍 the song or not👎 .

In this competition you will analyse such informations as: **artist, genre, key, year of release etc. ** 
More complete description you can find in Description.yaml file

--Data Description--
The data has been split into two groups:
- training set (train.csv)
- test set (test.csv)
The training set should be used to build your machine learning models. 
For the training set, we provide my preference (also known as the “ground truth”) for each track.

The test set should be used to see how well your model performs on unseen data. 
For the test set, we do not provide the ground truth for each passenger. Your have to predict it by yourself.

We also include sample_submition.csv, as an example of submission file.

--Evaluation--
Our score is the percentage of preferences you correctly predict. 
This is known as [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification). 
Submission File Format You should submit a csv file with exactly 300 entries plus a header row. 
Your submission will show an error if you have extra columns or rows. 
The file should have exactly 2 columns: - Id (sorted in any order) - Category (contains your binary predictions: 1 for like, 0 for dislike)
 
