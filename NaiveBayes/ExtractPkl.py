import os
import pickle
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    f.seek(0)# go back to beginning of file
    all_text = f.read()

    #split off metadata
    content = all_text.split("X-FileName:")

    if len(content) > 1:
        #remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
      
	tmp = text_string.split()	
	#stemming 
        stemmer=SnowballStemmer("english",ignore_stopwords=False)
        for i in range(0,len(tmp)):
            tmp[i]=stemmer.stem(tmp[i])

        words=" ".join(tmp)
    return words


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        path = "/home/shehab/ud120-projects/"+path[:-1]
        print path
        email = open(path, "r")
        # use parseOutText to extract the text from the opened email
        words=parseOutText(email)

        # Remove Uneccessary names"not stop words but like them"
        for i in ["sara", "shackleton", "chris", "germani"]:
            words=words.replace(i,"")

        # append the text to word_data
        word_data.append(words)
        # append 0 to from_data if email is from Sara, and 1 if email is from Chris
        if(name == "sara"):
            from_data.append(0)
        else:
            from_data.append(1)

        email.close()

print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("New_word_data.pkl", "w") )
pickle.dump( from_data, open("New_email_authors.pkl", "w") )

