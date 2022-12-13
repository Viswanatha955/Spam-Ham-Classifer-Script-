import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer(max_features=2500)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
mul_naive_bayes = MultinomialNB()
from sklearn.metrics import confusion_matrix,accuracy_score




spam_ham_data = pd.read_csv('SMSSpamCollection',sep = '\t',
                            names=['label','messages'])
corpus = []
for i in range(0,5572):
    review = re.sub('[^a-zA-Z]',' ', spam_ham_data['messages'][i])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if word not in stopwords.words('english')] 
    review = ' '.join(review)
    corpus.append(review)
    
 X = cv.fit_transform(corpus).toarray()

y = le.fit_transform(spam_ham_data['label'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=12,stratify=y)

mul_naive_bayes.fit(X_train,y_train)

y_pred = mul_naive_bayes.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
