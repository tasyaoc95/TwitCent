#Script to take in preprocessed tweets and create training, testing, and validation sets.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#Stops a reoccuring future warning
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

#Read in clean tweets csv
csv = 'cleantweets.csv'
df = pd.read_csv(csv, header = 0, index_col = 0)

#Split into inputs and target
x = df.text
y = df.target
SEED = 1000

#Split into train and val_test
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.1, random_state=SEED)

#Split val_test into validation and test
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)


print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train), (len(x_train[y_train == 0]) / (len(x_train)*1.))*100, (len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation), (len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100, (len(x_validation[y_validation == 1]) / (len(x_validation)*1.))*100))
print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test), (len(x_test[y_test == 0]) / (len(x_test)*1.))*100, (len(x_test[y_test == 1]) / (len(x_test)*1.))*100))
