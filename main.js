#main.js
#Making Features and labels using count Vectorizer   
cv = CountVectorizer()
features_train= cv.fit_transform(corpus).toarray() 
labels_train= dataset.iloc[:,1].values    
    
features_test = cv.transform(testing).toarray()

#Training of model using RandomForest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(features_train,labels_train)


#Saving the model

# Save the model as a pickle in a file 
joblib.dump(classifier, 'twitter.pkl') 
  
#Load the model from the file 
loading_model= joblib.load('twitter.pkl')  
  
#Use the loaded model to make predictions 
labels_pred=loading_model.predict(features_test)

# OR

"""            
import pickle
saved_model = pickle.dumps(classifier)

loading_saved_model = pickle.loads(saved_model)

labels_pred=loading_saved_model.predict(features_test) 
"""

#visualization of predicted data

df['labels_pred']=labels_pred
df.head()
df.labels_pred.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "green"])




