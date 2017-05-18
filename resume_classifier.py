import slate
import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# importing all the words from the resumes into a list
resume_list = []
for i in range(97):
    filename = "/c" + str(i+1) + ".pdf"
    f = open("CVs" + filename, "rb")
    doc = slate.PDF(f)
    each_resume = ""	
    for j in range(len(doc)):
        each_resume += doc[j]
    resume_list.append(each_resume)
print resume_list
	
# removing punctuations and other unnecessary characters
for i in range(len(resume_list)):
    resume_list[i] = resume_list[i].translate(None, string.punctuation)
    resume_list[i] = resume_list[i].translate(None, "\n")
    resume_list[i] = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', resume_list[i])
	
# extracting words as features from the resume list and making a feature matrix
vectorizer = TfidfVectorizer(analyzer="word", stop_words="english", max_features=250)
words = vectorizer.fit_transform(resume_list)
features = words.toarray()
print features

# labelling the existing resumes as being accepted(1) or being rejected(0)
# The first 36 resumes are labelled as accepted in this case and a label list is prepared
label = []
for i in range(36):
    label.append(1)
for i in range(61):
    label.append(0)
label = np.array(label)

#shuffling and splitting the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Using Decision Tree Classifier on the data
dtclf = tree.DecisionTreeClassifier()
dtclf = dtclf.fit(X_train, y_train)
print dtclf.score(X_train, y_train)
print dtclf.score(X_test, y_test)

# Using Random Forest Classifier on the data
rfclf = RandomForestClassifier()
rfclf = rfclf.fit(X_train, y_train)
print rfclf.score(X_train, y_train)
print rfclf.score(X_test, y_test)

# Using SVM Classifier on the data
model_svm = svm.SVC()
model_svm = model_svm.fit(X_train, y_train)
print model_svm.score(X_train, y_train)
print model_svm.score(X_test, y_test)