#Script which reads in the raw corpus of tweets and removes tags, hashtags, urls, negations etc
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer

#Define  column names
columns = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']

#fetch data into dataframe adding column names
data = pd.read_csv('rawdataset.csv', header = None, names = columns, encoding = 'latin-1')


#Map the 4 representing positive sentiment to 1
data['sentiment'] = data['sentiment'].map({0:0, 4:1})

#Remove unnecessary columns leaving only the text and sentiment
data = data.drop(['id', 'date', 'query_string', 'user'], axis = 1)

#instantiate tokenizer
tok = WordPunctTokenizer()
#pat1 for usernames
#pat2 and www_pat for links
pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'

#Manually created dictionary of contracted words to be transformed into normal words.
contractions = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
con_pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b')

#Tweet cleaning function which takes raw tweets and performs cleaning on them: HTML decoding,
#remove '@', links,  utf BOM, hashtags. Returns cleaned, lowercased tweet.
def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.encode().decode("utf-8-sig").replace(u"ï¿½", "?")
    except:
        bom_removed = souped
    #remove links etc. and lowercase all characters
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process above, it has created unnecessay white spaces,
    # I tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()

#Clean in batches
nums = [0, 400000, 800000, 1200000, 1600000]
print("Cleaning and parsing the tweets...\n")
cleanText = []
for i in range(nums[0], nums[-1]):
    if ((i+1)%10000 == 0):
        print("Tweets %d of %d has been processed" %(i+1, nums[-1]))
    cleanText.append(tweet_cleaner(data['text'][i]))

#Make new data frame of clean tweets
cleanDf = pd.DataFrame(cleanText, columns = ['text'])

#Add the sentiment column from the original file
cleanDf['target'] = data.sentiment
print(cleanDf.head(30))

#Remove null entries
cleanDf.dropna(inplace = True)
cleanDf.reset_index(drop = True, inplace = True)

#Save as csv
cleanDf.to_csv('cleantweets.csv', encoding = 'utf-8')

#reopen, delete null rows resulting from cleaning and resave
df = pd.read_csv('cleantweets.csv', header = 0, index_col = 0)
df.dropna(inplace = True)
df.reset_index(drop = True, inplace = True)
df.to_csv('cleantweets.csv', encoding = 'utf-8')

#retrieve and print from new csv to check all is well
df = pd.read_csv('cleantweets.csv', index_col = 0, header = 0)
print(df.head(30))