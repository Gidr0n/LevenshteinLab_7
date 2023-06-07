import nltk
import Levenshtein

with open('litw-win.txt', 'r', encoding='cp1251') as file:
    words = file.read().split()

sentence='''с велечайшим усилием выбравшись из потока убегающих людей Кутузов со свитой уменьшевшейся вдвое поехал на звуки выстрелов русских орудий'''
for i, word in enumerate(sentence.split()):
    if word not in words:
        min_distance = float('inf')
        nearest_word = ''
        for word_in_dict in words:
            distance = Levenshtein.distance(word, word_in_dict)
            if distance < min_distance:
                min_distance = distance
                nearest_word = word_in_dict
        sentence = sentence.replace(word, nearest_word, 1)
print(sentence)
#2
text = "Считайте слова из файла litw-win.txt и запишите их в список words. В заданном предложении исправьте все опечатки, заменив слова с опечатками на ближайшие (в смысле расстояния Левенштейна) к ним слова из списка words. Считайте, что в слове есть опечатка, если данное слово не содержится в списке words. Разбейте этот текст из формулировки на слова; проведите стемминг и лемматизацию слов."

import re
words = re.findall(r'\w+', text)
print(words)
from nltk.stem import SnowballStemmer
nltk.download('wordnet')
russian = SnowballStemmer('russian')
stemmed_words = [russian.stem(word) for word in words]
print(stemmed_words)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for word in words:
    pos = wordnet.VERB
    if wordnet.synsets(word):
        pos = wordnet.synsets(word)[0].pos()
    lemmatized_word = lemmatizer.lemmatize(word, pos)
    lemmatized_words.append(lemmatized_word)
print(lemmatized_words)
#3
from sklearn.feature_extraction.text import CountVectorizer
sentences =words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
print(X.toarray())
print(vectorizer.get_feature_names_out())
#Лабораторная работа
#1
import pandas as pd
from nltk.tokenize import word_tokenize

df = pd.read_csv('preprocessed_descriptions.csv')
sentences = df['preprocessed_descriptions'].tolist()
words = set()
for sentence in sentences:
    try:
        words.update(word_tokenize(sentence))
    except TypeError:
        pass
print(words)
#2
import random
import editdistance

df = pd.read_csv('preprocessed_descriptions.csv')
sentences = df['preprocessed_descriptions'].tolist()
words = set()
for sentence in sentences:
    try:
        words.update(word_tokenize(sentence))
    except TypeError:
        pass

word_pairs = random.sample(list(words), k=10)
word_pairs = [(word_pairs[i], word_pairs[i+1]) for i in range(0, len(word_pairs), 2)]
for pair in word_pairs:
    distance = editdistance.eval(pair[0], pair[1])
    print(f"Расстояние между '{pair[0]}' и '{pair[1]}': {distance}")
#3
def find_closest_words(word, words, k):
    distances = [(w, Levenshtein.distance(word, w)) for w in words]
    distances.sort(key=lambda x: x[1])
    return [w[0] for w in distances[:k]]
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
for index, row in random_recipes.iterrows():
    description = row['preprocessed_descriptions']
    vector = vectorizer.fit_transform([description])
    print(f"Рецепт {index}: {vector.toarray()}")

from scipy.spatial.distance import cosine
import numpy as np
recipes = pd.read_csv('preprocessed_descriptions.csv')
random_recipes = recipes.sample(n=5)
n = len(random_recipes)
similarity_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        similarity_matrix[i][j] = 1 - cosine(random_recipes.iloc[i]['preprocessed_descriptions'], random_recipes.iloc[j]['preprocessed_descriptions'])
similarity_df = pd.DataFrame(similarity_matrix, columns=random_recipes.index, index=random_recipes.index)
print(similarity_df)
