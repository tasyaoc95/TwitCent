from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer
from datetime import datetime
from dateutil import parser
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
#from geopy.geocoders import Nominatim, GoogleV3
#from geopy.exc import GeocoderTimedOut
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences

#load logisitic regression model and vectorizer models from disk
#filename_lr = 'saved_lr.sav'
filename_vect = 'saved_vect.sav'
filename_tok = 'tokenizer.pickle'
#Pretrained logistic regression model for sentiment analysis
# loaded_model = joblib.load(open(filename_lr, 'rb'))
#Pre trained vectorizer for transforming tweets into suitable format for the LR model.
#vect_char = joblib.load(open(filename_vect, 'rb'))

#Load neural network model
loaded_model = load_model('LSTM_best_weights.04-0.8349.hdf5')

#Load in neural network tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

tokenizer.oov_token = None
#Define some variables and transformations which will be needed for cleaning up raw tweets.
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')


#Tweet cleaning function
def tweet_cleaner(text):
    """Function which takes raw tweets and performs cleaning on them: HTML decoding, remove '@',
    remove links, remove utf BOM, remove hashtags. Returns cleaned, lowercased tweet."""
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.encode().decode("utf-8-sig").replace(u"ï¿½", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()

def clean_date(raw_time):
    """Function which reads in the dirty date format from twitter API
    and returns clean date_time format for easier usability going forwrad"""
    time_stamp = raw_time.split(" ")
    time_stamp = str(time_stamp[1]+' '+time_stamp[2]+' '+time_stamp[3]+' '+time_stamp[5])
    clean_date_time = parser.parse(time_stamp)
    return clean_date_time

def sentiment_score(text, loaded_model = loaded_model, vectorizer = tokenizer):
    """Function which takes cleaned up tweet text, represents it in feature_vector
    format e.g. tfidf and performs senitment analysis on it using a pre-trained LSTM model."""
    # tweet_tf_idf = vect_char.transform(text)
    tweet_token = tokenizer.texts_to_sequences(text)
    tweet_token = pad_sequences(tweet_token, maxlen = 40)
    sentiment = loaded_model.predict_proba(tweet_token)
    neg_prob = sentiment[0][0]
    pos_prob = sentiment[0][1]
    return neg_prob, pos_prob




# loaded_model = load_model('LSTM_best_weights.04-0.8349.hdf5')
# loaded_model.evaluate(x=x_val_seq, y=y_validation)

