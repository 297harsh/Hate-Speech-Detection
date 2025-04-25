import numpy as np
import pandas as pd
import re
import nltk
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load Dataset
dataset = pd.read_csv("twitter.xls")
dataset["label"] = dataset["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No hate or offensive language"})
data = dataset[["tweet", "label"]]

# Data Cleaning
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
stemmer = nltk.SnowballStemmer("english")

def clean_data(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', "", text)
    text = re.sub(r'\[.*?\]', "", text)
    text = re.sub(r'<.*?>', "", text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), "", text)
    text = re.sub(r'\w*\d\w*', ' ', text).strip()
    text = [word for word in text.split(' ') if word not in stop_words]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(" ")]
    return " ".join(text)

# Apply cleaning
data["tweet"] = data["tweet"].apply(clean_data)

# Feature Extraction
x = np.array(data["tweet"])
y = np.array(data["label"])
cv = CountVectorizer()
x = cv.fit_transform(x)

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear'),
    # "MLP": MLPClassifier(max_iter=300)
}

# Train & Evaluate
accuracies = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(5, 4))
    # sns.heatmap(cm, annot=True, fmt=".1f", cmap="YlGnBu")
    # plt.title(f"Confusion Matrix for {name}")
    # plt.show()

# Compare Accuracies
plt.figure(figsize=(8, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['skyblue', 'lightgreen', 'salmon', 'violet'])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("Algorithm Comparison")
plt.ylim(0, 1)
plt.show()

# Most Accurate Model
best_model = max(accuracies, key=accuracies.get)
print(f"Most Accurate Algorithm: {best_model} with Accuracy: {accuracies[best_model]:.4f}")

# Test on a sample tweet
sample = "Let's unite and kill all the people who are protecting against the government"
sample = clean_data(sample)
data1 = cv.transform([sample]).toarray()
result = models[best_model].predict(data1)
print("Prediction for the sample tweet:", result[0])