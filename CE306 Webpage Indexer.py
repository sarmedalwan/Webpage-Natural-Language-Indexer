import lxml
import requests
import ssl
from lxml.html import fromstring
from lxml.html.clean import Cleaner
import urllib #Imports URLLib2, the library which allows the program to import and decode the text from a webpage
import nltk #Imports NLTK (Natural Language Toolkit), the library which will allow this program to process the text retrieved from the web
from nltk import * #Imports all sub-components of the NLTK library
from bs4 import BeautifulSoup #BeautifulSoup will allow for parsing of the HTML import to remove all of the code and leave us with only the text from the page
from nltk.corpus import stopwords #Stopwords will allow the program to remove unimportant words from the text, known as stop-words
from nltk.text import TextCollection
from nltk.stem import *
from nltk.stem.porter import *
import operator

user = " "
while user != "":
    user = input("To run the program, press enter: ")

    

#Start of part one

ssl._create_default_https_context = ssl._create_unverified_context

urlone = "http://csee.essex.ac.uk/staff/udo/index.html"
urltwo = "https://ecir2019.org/industry-day/"                    #Opening URLs


html = urllib.request.urlopen(urlone).read() #Takes the pure text from the webpage, including the HTML code
text_file = open("raw_html.txt", "w")
text_file.write(html.decode("utf-8"))
print("RAW HTML 1")
print("-----------------------------------------------------------------------------")
print(html.decode("utf-8"))
text_file.close()

#second url below

html2 = urllib.request.urlopen(urltwo).read() #Takes the pure text from the webpage, including the HTML code
text_file = open("raw_html2.txt", "w")
text_file.write(html2.decode("utf-8"))
print("RAW HTML 2")
print("-----------------------------------------------------------------------------")
print(html2.decode("utf-8"))
text_file.close()    

#End of part one








#Start of part two

soup = BeautifulSoup(html, "lxml")
for script in soup(["script", "style"]):
    script.decompose() #Removes script elements
text = soup.get_text() #Get words from the parse tree
lines = (line.strip() for line in text.splitlines()) #Splits the lines by line breaks
chunks = (phrase.strip() for line in lines for phrase in line.split("  ")) #Removes blank lines
text_without_scripts = '\n'.join(chunk for chunk in chunks if chunk) #Joins the lines into one string

pagetitlemeta = soup.find("meta",  property="og:title")
pagetitle = pagetitlemeta["content"] if pagetitlemeta else "No meta title given" #Find meta title
if pagetitle == "No meta title given": #If no meta title is found, try an alternate method
    html = requests.get(urlone)
    tree = fromstring(html.content)
    pagetitle = tree.findtext('.//title')

text_file = open("FinalPageTextWithoutScripts.txt", "w")
text_file.write(text_without_scripts)
print("TEXT WITHOUT SCRIPTS 1")
print("-----------------------------------------------------------------------------")
print(text_without_scripts)
text_file.close()

#Second url below

soup2 = BeautifulSoup(html2, "lxml")
for script in soup2(["script", "style"]):
    script.decompose() #Removes script elements
text2 = soup2.get_text() #Get words from the parse tree
lines2 = (line.strip() for line in text2.splitlines()) #Splits the lines by line breaks
chunks2 = (phrase.strip() for line in lines2 for phrase in line.split("  ")) #Removes blank lines
text_without_scripts2 = '\n'.join(chunk for chunk in chunks2 if chunk) #Joins the lines into one string

pagetitlemeta2 = soup2.find("meta",  property="og:title")
pagetitle2 = pagetitlemeta2["content"] if pagetitlemeta else "No meta title given" #Find meta title
if pagetitle2 == "No meta title given": #If no meta title is found, try an alternate method
    html2 = requests.get(urltwo)
    tree2 = fromstring(html2.content)
    pagetitle2 = tree2.findtext('.//title')


text_file = open("FinalPageTextWithoutScripts2.txt", "w")
text_file.write(text_without_scripts2)
print("TEXT WITHOUT SCRIPTS 2")
print("-----------------------------------------------------------------------------")
print(text_without_scripts2)
text_file.close()

#end of part two








#start of part three, pre-processing: sentence splitting, tokenisation and normalisation


sent_tokenize = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = sent_tokenize.tokenize(text_without_scripts)

text_file = open("SplitSentences.txt", "w")
print("SPLIT SENTENCES 1")
print("-----------------------------------------------------------------------------")
for x in sentences:
    text_file.write(x + "\n")
    print(x + "\n")
text_file.close()

word_text = nltk.word_tokenize(text_without_scripts) #Tokenises the raw text into a list of individual words for parsing
no_punctuation = [term for term in word_text if nltk.re.search("\w", term)] #Removes punctuation from the words
filtered_words = []
stopWords = set(stopwords.words('english'))

for w in no_punctuation:
    if w.lower() not in stopWords:
        filtered_words.append(w) #Adds word to list of filtered words unless it's a stop word

text_file = open("PageTextWithoutStopWords.txt", "w")
#("PAGE TEXT WITHOUT STOP WORDS 1")
#("-----------------------------------------------------------------------------")
for x in filtered_words:
    text_file.write(x + "\n")
text_file.close()

# Second url below

sent_tokenize2 = nltk.data.load('tokenizers/punkt/english.pickle')
sentences2 = sent_tokenize2.tokenize(text_without_scripts2)

text_file = open("SplitSentences2.txt", "w")
print("SPLIT SENTENCES 2")
print("-----------------------------------------------------------------------------")
for x in sentences2:
    text_file.write(x + "\n")
    print(x + "\n")
text_file.close()

word_text2 = nltk.word_tokenize(text_without_scripts2) #Tokenises the raw text into a list of individual words for parsing
no_punctuation2 = [term for term in word_text2 if nltk.re.search("\w", term)] #Removes punctuation from the words
filtered_words2 = []

for w in no_punctuation2:
    if w.lower() not in stopWords:
        filtered_words2.append(w) #Adds word to list of filtered words unless it's a stop word


text_file = open("PageTextWithoutStopWords2.txt", "w")
#("PAGE TEXT WITHOUT STOP WORDS 2")
#("-----------------------------------------------------------------------------")
for x in filtered_words2:
    text_file.write(x + "\n")
text_file.close()


#end of part three, pre-processing: sentence splitting, tokenisation and normalisation










#start of part four, POS tagging

tagged_words = nltk.pos_tag(filtered_words) #Uses NLTK's Part-of-Speech tagger to tag the words by their type

text_file = open("FinalWordswithPOSTags.txt", "w")
print("FINAL WORDS WITH POS TAGS")
print("-----------------------------------------------------------------------------")
for x in list(tagged_words):
    text_file.write(x[0] + "\n" + x[1] + "\n")     #output file of words with POS tags
    print(x[0] + "\n" + x[1] + "\n")
text_file.close()

#second url below

tagged_words2 = nltk.pos_tag(filtered_words2) #Uses NLTK's Part-of-Speech tagger to tag the words by their type

text_file = open("FinalWordswithPOSTags2.txt", "w")
print("FINAL WORDS WITH POS TAGS 2")
print("-----------------------------------------------------------------------------")
for x in list(tagged_words2):
    text_file.write(x[0] + "\n" + x[1] + "\n")     #output file of words with POS tags
    print(x[0] + "\n" + x[1] + "\n")
text_file.close()

#end of part four, POS tagging









                 #tree = lxml.etree.HTML(html)
                 #pagekeywordsmeta = tree.xpath( "//meta[@name='Keywords']" )#.get("content")







#Start of part five and six, selecting keywords and stemming


for t_k in tagged_words:
    if t_k[1] == "DT":
        tagged_words.remove(t_k)
    if t_k[1] == "EX":
        tagged_words.remove(t_k)
    if t_k[1] == "IN":
        tagged_words.remove(t_k)                       #Filtering out words by POS tags
    if t_k[1] == "WDT":
        tagged_words.remove(t_k)
    if t_k[1] == "UH":
        tagged_words.remove(t_k)
    if t_k[1] == "WP$":
        tagged_words.remove(t_k)
    if t_k[1] == "CD" and len(t_k[0])<4:
        tagged_words.remove(t_k)


        #second url below

for t_k in tagged_words2:
    if t_k[1] == "DT":
        tagged_words2.remove(t_k)
    if t_k[1] == "EX":
        tagged_words2.remove(t_k)
    if t_k[1] == "IN":
        tagged_words2.remove(t_k)                       #Filtering out words by POS tags
    if t_k[1] == "WDT":
        tagged_words2.remove(t_k)
    if t_k[1] == "UH":
        tagged_words2.remove(t_k)
    if t_k[1] == "WP$":
        tagged_words2.remove(t_k)
    if t_k[1] == "CD" and len(t_k[0])<4:
        tagged_words2.remove(t_k)







#Start of part six, stemmnig, embedded in part five

untagged_words = []
for t_k in tagged_words:                #remove POS tags, no longer needed
    untagged_words.append(t_k[0])

stemmed_words = []
stemmer = PorterStemmer()
for word in untagged_words:                                   #stemming the words
    stemmed_words.append(stemmer.stem(word))

text_file = open("StemmedWords.txt", "w")
print("STEMMED WORDS 1")
print("-----------------------------------------------------------------------------")
for x in list(stemmed_words):
    text_file.write(x + " ")     #output file of stemmed words
    print(x + " ")
text_file.close()

#second url below

untagged_words2 = []
for t_k in tagged_words2:                #remove POS tags, no longer needed
    untagged_words2.append(t_k[0])

stemmed_words2 = []
stemmer = PorterStemmer()
for word in untagged_words2:                                   #stemming the words
    stemmed_words2.append(stemmer.stem(word))

text_file = open("StemmedWords2.txt", "w")
print("STEMMED WORDS 2")
print("-----------------------------------------------------------------------------")
for x in list(stemmed_words2):
    text_file.write(x + " ")     #output file of stemmed words
    print(x + " ")
text_file.close()

#End of part six, stemmming, embedded in part five










corpus = [stemmed_words, stemmed_words2]
text = TextCollection(corpus)

stemmed_words = set(stemmed_words)
final_list = []
for word in stemmed_words:
    final_list.append((word, text.tf_idf(word, text)))
    


if pagetitle != "No meta title given": #If the page has a title in the meta tags, operate on the title
    filtered_page_title_words = []
    pagetitlewords = pagetitle.split()
    for w in pagetitlewords:
        if w.lower() not in stopWords:
            filtered_page_title_words.append(w.lower()) #Adds word to list of filtered words unless it's a stop word

filtered_page_title_words.append(pagetitle)
for word in filtered_page_title_words:
    a = (word, text.tf_idf(word, text) + .75)
    final_list.append(a)

final_list.sort(key=operator.itemgetter(1))
final_list.reverse()
text_file = open("FirstURLIndexWords.txt", "w")
text_file.write("Index tf_idf" + "\n")
print("TF-IDF INDEX 1")
print("-----------------------------------------------------------------------------")
for x in final_list[0:20]:
    text_file.write(str(final_list.index(x)+1)+". " + str(x[0]) + ", " + str(x[1]) + "\n")     #output file of words with POS tags
    print(str(final_list.index(x)+1)+". " + str(x[0]) + ", " + str(x[1]) + "\n")
text_file.close()



#second url below




stemmed_words2 = set(stemmed_words2)
final_list2 = []
for word in stemmed_words2:
    final_list2.append((word, text.tf_idf(word, text)))
    

if pagetitle2 != "No meta title given": #If the page has a title in the meta tags, operate on the title
    filtered_page_title_words2 = []
    pagetitlewords2 = pagetitle2.split()
    for w in pagetitlewords2:
        if w.lower() not in stopWords:
            filtered_page_title_words2.append(w.lower()) #Adds word to list of filtered words unless it's a stop word

filtered_page_title_words2.append(pagetitle2)
for word in filtered_page_title_words2:
    a = (word, text.tf_idf(word, text) + .75)
    final_list2.append(a)

final_list2.sort(key=operator.itemgetter(1))
final_list2.reverse()
text_file = open("SecondURLIndexWords.txt", "w")
text_file.write("Index tf_idf" + "\n")
print("TF-IDF INDEX 2")
print("-----------------------------------------------------------------------------")
for x in final_list2[0:20]:
    text_file.write(str(final_list2.index(x)+1)+". " + str(x[0]) + ", " + str(x[1]) + "\n")     #output file of words with POS tags
    print(str(final_list2.index(x)+1)+". " + str(x[0]) + ", " + str(x[1]) + "\n")
text_file.close()


#End of part five




