# Resume Classifier

A simple project made to apply newly learnt Machine Learning algorithms. The Resume Classifier seeks to make the procedure of initial selection of resumes for any job/university interview more robust by doing so based on the data of previously selected and rejected candidates.

##### The procedure followed was:

* Each word from all the resumes was extracted and added to a list, such that, the first element of the list had all the words from the first resume and so on.
* The punctuations, newline breaks and other characters included due to the use of the Slate(PDF) library were removed from the list.
* The selected and rejected resumes were labelled accordingly.
* The resumes in the list along with the labels were split randomly into training and testing data in the ratio 2:1
* Using the TFIDF Vectorizer, features were extracted from the training and testing data according to the frequency and relevance of the words in the resumes of selected and rejected candidates. (The number of features was set at 250)
* Learning Algorithms like Decision Tree Classifier, Random Forest Classifier, SVM and Naive Bayes Classifiers were then used on the data to obtain results.

##### The resulting accuracy which the various algorithms gave were as follows:
* Decision Tree
	* Training data : 100%
	* Testing data : 57%
* Random Forest
	* Training data : 98%
	* Testing data : 63%
* SVM
	* Training data : 65%
	* Testing data : 57%
* Bernoulli Naive Bayes
	* Training data : 93%
	* Testing data : 68%
* Gaussian Naive Bayes
	* Training data : 92%
	* Testing data : 57%