"""
This Python script implements a classification model to predict whether a 
region’s economy grew by more than 5% using OECD economic and demographic data. 
It loads and preprocesses the datasets, including feature scaling and encoding,
before merging them. The data is then split into training and test sets. 
Three classification algorithms — Random Forest, Logistic Regression, and a 
Multilayer Perceptron — are trained and evaluated using precision, recall, 
F1-score, accuracy, and a confusion matrix to assess model performance. 
The script is organized within a class (OECD_Classification) to ensure 
modularity and maintainability, automating data loading, preprocessing, 
model training, and performance evaluation.

@author: Federico Durante
"""

"""Import libraries"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import sklearn.metrics

class OECD_Classification:
    """Class for classification of economic growth using OECD data"""
    
    def __init__(self):
        """
        Constructor method
        
        Loads and processes data, prepares classifiers, and splits the dataset.
        """
        
        """Prepare class level variables for data"""
        self.df = None  
        self.output_df = None  
        self.x_train = None  
        self.x_test = None  
        self.y_train = None
        self.y_test = None
        
        """Prepare classifiers"""
        self.classifiers = {
            "RandomForest": RandomForestClassifier(),
            # Increased max_iter to ensure convergence
            "LogisticRegression": LogisticRegression(max_iter=5000), 
            # MLP with two hidden layers (100, 50) and higher max_iter for better training
            "MLPClassifier": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=5000)  
        }
        
        """Load, merge and preprocess data"""
        self.load_data()
        self.split_training_and_test_set()
    
    def load_data(self):
        """
        Load and prepare the dataset.
        
        Reads the datasets and merges them on REG_ID.
        """
        
        """Load data"""
        df_growth = pd.read_csv("assignment_4_OECD_data_growth.csv")  
        df_explaining_vars = pd.read_csv("assignment_4_OECD_data_explaining_vars.csv")  
        
        """Merge data"""
        self.df = pd.merge(df_explaining_vars, df_growth, on="REG_ID", how="inner")  
        
        """Extract independent and dependent variables"""
        self.output_df = self.df[["Growth > 5%"]]  
        self.df = self.df.drop(columns=["REG_ID", "Growth > 5%"])  
        
        """Feature scaling to standardize input variables"""
        scaler = StandardScaler()
        self.df = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns)
    
    def split_training_and_test_set(self):
        """
        Splits the dataset into training and test sets.
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.df, self.output_df, test_size=0.3, random_state=42
        )  
    
    def run_classifiers(self):
        """
        Runs classification models and evaluates their performance.
        """
        
        for classifier_key in self.classifiers.keys():
            
            """Print classifier name"""
            print("\n\nClassifier:", classifier_key)  
            classifier = self.classifiers[classifier_key]  
            
            """Train the model"""
            classifier.fit(self.x_train, np.asarray(self.y_train).T[0])  
            
            """Predict test set"""
            prediction = classifier.predict(self.x_test)  
            
            """Evaluate model performance"""
            print(pd.DataFrame({"Prediction": prediction, "True value": np.asarray(self.y_test).T[0]}))  
            
            """Calculate precision"""
            precision = sklearn.metrics.precision_score(self.y_test, prediction)  
            print("Precision:", precision)
            
            """Calculate recall"""
            recall = sklearn.metrics.recall_score(self.y_test, prediction)
            print("Recall:", recall)
            
            """Calculate F1 score"""
            F1 = sklearn.metrics.f1_score(self.y_test, prediction)  
            print("F1 score:", F1)
            
            """Calculate accuracy"""
            accuracy = sklearn.metrics.accuracy_score(self.y_test, prediction)
            print("Accuracy:", accuracy)
            
            """Print confusion matrix"""
            print("Confusion matrix:\n", sklearn.metrics.confusion_matrix(self.y_test, prediction))  
            
"""Main entry point"""
if __name__ == '__main__':
    
    """Initialize and run the classification model"""
    model = OECD_Classification()
    model.run_classifiers()
