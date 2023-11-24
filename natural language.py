question 1

#NLP
import nltk
nltk.download('punkt')
#Tokenizing
from nltk.tokenize import *
text="""A newspaper is the strongest medium for news. People are reading newspapers
for decades. It has a huge contribution to globalization. Right now because of easy
internet connection, people don’t read printed newspapers often. They read the online
version."""
print("Sample text : \n ",text,"\n")
sent_tokenized=sent_tokenize(text)
print("Tokenizing by sentence : \n",sent_tokenized,"\n")
word_tokenized=word_tokenize(text)
print("Tokenizing by word : \n ",word_tokenized,"\n")


output
Sample text : 
  A newspaper is the strongest medium for news. People are reading newspapers
for decades. It has a huge contribution to globalization. Right now because of easy
internet connection, people don’t read printed newspapers often. They read the online
version. 

Tokenizing by sentence : 
 ['A newspaper is the strongest medium for news.', 'People are reading newspapers\nfor decades.', 'It has a huge contribution to globalization.', 'Right now because of easy\ninternet connection, people don’t read printed newspapers often.', 'They read the online\nversion.'] 

Tokenizing by word : 
  ['A', 'newspaper', 'is', 'the', 'strongest', 'medium', 'for', 'news', '.', 'People', 'are', 'reading', 'newspapers', 'for', 'decades', '.', 'It', 'has', 'a', 'huge', 'contribution', 'to', 'globalization', '.', 'Right', 'now', 'because', 'of', 'easy', 'internet', 'connection', ',', 'people', 'don', '’', 't', 'read', 'printed', 'newspapers', 'often', '.', 'They', 'read', 'the', 'online', 'version', '.'] 

[nltk_data] Downloading package punkt to /home/ccn/nltk_data...
[nltk_data]   Package punkt is already up-to-date!




program2


#Filtering stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from string import punctuation
stopwords=stopwords.words('english')
punctuation=list(punctuation)
print("After filtering the stop words and punctuation : ")
for word in word_tokenized:
 if word.casefold() not in stopwords and word.casefold() not in punctuation:
   print(word)
#new_list=[word for word in word_tokenized if word.casefold() not in stopwords and word not in punctuation]
#print(new_list,"\n")


output

After filtering the stop words and punctuation : 
newspaper
strongest
medium
news
People
reading
newspapers
decades
huge
contribution
globalization
Right
easy
internet
connection
people
’
read
printed
newspapers
often
read
online
version
[nltk_data] Downloading package stopwords to /home/ccn/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!



program3

#Stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
words = ["reading", "globalization", "Being","Went","gone","going"]
print("Given words : ",words)
stemm=[ps.stem(i) for i in words ]
print("After stemming : ",stemm,"\n")


output
Given words :  ['reading', 'globalization', 'Being', 'Went', 'gone', 'going']
After stemming :  ['read', 'global', 'be', 'went', 'gone', 'go'] 


program4

#Lemmatization
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
lem= WordNetLemmatizer()
print("rocks :", lem.lemmatize("rocks"))
print("corpora :", lem.lemmatize("corpora"))
print("better :", lem.lemmatize("better"))
print("believes :", lem.lemmatize("believes"),"\n")
print("better :", lem.lemmatize("went",pos="a"))
print("better :", lem.lemmatize("went",pos="v"))
print("better :", lem.lemmatize("went",pos="n"),"\n")

output

[nltk_data] Downloading package wordnet to /home/ccn/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package omw-1.4 to /home/ccn/nltk_data...
[nltk_data]   Package omw-1.4 is already up-to-date!
rocks : rock
corpora : corpus
better : better
believes : belief 

better : went
better : go
better : went 


program 5


nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import RegexpParser
from nltk.tree import *
#POS Tag
postag=nltk.pos_tag(word_tokenized)
print("POS tagging : \n")
for i in postag:
 print(i)
#Chunking
print("\n")
grammar = "NP: {<DT>?<JJ>*<NN>}"
chunker = RegexpParser(grammar)
output = chunker.parse(postag)
print("After Chunking:\n",output)
output.pretty_print()


output

POS tagging : 

('A', 'DT')
('newspaper', 'NN')
('is', 'VBZ')
('the', 'DT')
('strongest', 'JJS')
('medium', 'NN')
('for', 'IN')
('news', 'NN')
('.', '.')
('People', 'NNS')
('are', 'VBP')
('reading', 'VBG')
('newspapers', 'NNS')
('for', 'IN')
('decades', 'NNS')
('.', '.')
('It', 'PRP')
('has', 'VBZ')
('a', 'DT')
('huge', 'JJ')
('contribution', 'NN')
('to', 'TO')
('globalization', 'NN')
('.', '.')
('Right', 'RB')
('now', 'RB')
('because', 'IN')
('of', 'IN')
('easy', 'JJ')
('internet', 'JJ')
('connection', 'NN')
(',', ',')
('people', 'NNS')
('don', 'VBP')
('’', 'JJ')
('t', 'NN')
('read', 'NN')
('printed', 'VBD')
('newspapers', 'NNS')
('often', 'RB')
('.', '.')
('They', 'PRP')
('read', 'VBD')
('the', 'DT')
('online', 'JJ')
('version', 'NN')
('.', '.')


After Chunking:
 (S
  (NP A/DT newspaper/NN)
  is/VBZ
  the/DT
  strongest/JJS
  (NP medium/NN)
  for/IN
  (NP news/NN)
  ./.
  People/NNS
  are/VBP
  reading/VBG
  newspapers/NNS
  for/IN
  decades/NNS
  ./.
  It/PRP
  has/VBZ
  (NP a/DT huge/JJ contribution/NN)
  to/TO
  (NP globalization/NN)
  ./.
  Right/RB
  now/RB
  because/IN
  of/IN
  (NP easy/JJ internet/JJ connection/NN)
  ,/,
  people/NNS
  don/VBP
  (NP ’/JJ t/NN)
  (NP read/NN)
  printed/VBD
  newspapers/NNS
  often/RB
  ./.
  They/PRP
  read/VBD
  (NP the/DT online/JJ version/NN)
  ./.)
                                                                                                                                                                                                  S                                                                                                                                                                                                                                 
   _______________________________________________________________________________________________________________________________________________________________________________________________|_________________________________________________________________________________________________________________________________________________________________________________________________________________                 
  |      |          |         |     |      |         |         |            |          |         |       |    |       |      |    |     |       |        |        |    |      |         |         |            |           |      |     |        |      |        NP                  NP       NP           NP                          NP                     NP                         NP         NP              NP              
  |      |          |         |     |      |         |         |            |          |         |       |    |       |      |    |     |       |        |        |    |      |         |         |            |           |      |     |        |      |    ____|_______            |        |      ______|___________                |             _________|____________          ____|___       |       ________|_________       
is/VBZ the/DT strongest/JJS for/IN ./. People/NNS are/VBP reading/VBG newspapers/NNS for/IN decades/NNS ./. It/PRP has/VBZ to/TO ./. Right/RB now/RB because/IN of/IN ,/, people/NNS don/VBP printed/VBD newspapers/NNS often/RB ./. They/PRP read/VBD ./. A/DT     newspaper/NN medium/NN news/NN a/DT huge/JJ contribution/NN globalization/NN easy/JJ internet/JJ connection/NN ’/JJ     t/NN read/NN the/DT online/JJ version/NN

[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     /home/ccn/nltk_data...
[nltk_data]   Package averaged_perceptron_tagger is already up-to-
[nltk_data]       date!
[nltk_data] Downloading package maxent_ne_chunker to
[nltk_data]     /home/ccn/nltk_data...
[nltk_data]   Package maxent_ne_chunker is already up-to-date!
[nltk_data] Downloading package words to /home/ccn/nltk_data...
[nltk_data]   Package words is already up-to-date!



program 6

#NER
Text = "The russian president Vladimir Putin is in the Kremlin"
Tokenize = nltk.word_tokenize(Text)
POS_tags = nltk.pos_tag(Tokenize)
NameEn = nltk.ne_chunk(POS_tags)
print(NameEn)


output

(S
  The/DT
  russian/JJ
  president/NN
  (PERSON Vladimir/NNP Putin/NNP)
  is/VBZ
  in/IN
  the/DT
  (FACILITY Kremlin/NNP))


program 7


#Word grouping
#bigram
print(list(nltk.bigrams(word_tokenized)),"\n")
#trigram
print(list(nltk.trigrams(word_tokenized)),"\n")
#n-gram
print(list(nltk.ngrams(word_tokenized,5)),"\n")


output

[('A', 'newspaper'), ('newspaper', 'is'), ('is', 'the'), ('the', 'strongest'), ('strongest', 'medium'), ('medium', 'for'), ('for', 'news'), ('news', '.'), ('.', 'People'), ('People', 'are'), ('are', 'reading'), ('reading', 'newspapers'), ('newspapers', 'for'), ('for', 'decades'), ('decades', '.'), ('.', 'It'), ('It', 'has'), ('has', 'a'), ('a', 'huge'), ('huge', 'contribution'), ('contribution', 'to'), ('to', 'globalization'), ('globalization', '.'), ('.', 'Right'), ('Right', 'now'), ('now', 'because'), ('because', 'of'), ('of', 'easy'), ('easy', 'internet'), ('internet', 'connection'), ('connection', ','), (',', 'people'), ('people', 'don'), ('don', '’'), ('’', 't'), ('t', 'read'), ('read', 'printed'), ('printed', 'newspapers'), ('newspapers', 'often'), ('often', '.'), ('.', 'They'), ('They', 'read'), ('read', 'the'), ('the', 'online'), ('online', 'version'), ('version', '.')] 

[('A', 'newspaper', 'is'), ('newspaper', 'is', 'the'), ('is', 'the', 'strongest'), ('the', 'strongest', 'medium'), ('strongest', 'medium', 'for'), ('medium', 'for', 'news'), ('for', 'news', '.'), ('news', '.', 'People'), ('.', 'People', 'are'), ('People', 'are', 'reading'), ('are', 'reading', 'newspapers'), ('reading', 'newspapers', 'for'), ('newspapers', 'for', 'decades'), ('for', 'decades', '.'), ('decades', '.', 'It'), ('.', 'It', 'has'), ('It', 'has', 'a'), ('has', 'a', 'huge'), ('a', 'huge', 'contribution'), ('huge', 'contribution', 'to'), ('contribution', 'to', 'globalization'), ('to', 'globalization', '.'), ('globalization', '.', 'Right'), ('.', 'Right', 'now'), ('Right', 'now', 'because'), ('now', 'because', 'of'), ('because', 'of', 'easy'), ('of', 'easy', 'internet'), ('easy', 'internet', 'connection'), ('internet', 'connection', ','), ('connection', ',', 'people'), (',', 'people', 'don'), ('people', 'don', '’'), ('don', '’', 't'), ('’', 't', 'read'), ('t', 'read', 'printed'), ('read', 'printed', 'newspapers'), ('printed', 'newspapers', 'often'), ('newspapers', 'often', '.'), ('often', '.', 'They'), ('.', 'They', 'read'), ('They', 'read', 'the'), ('read', 'the', 'online'), ('the', 'online', 'version'), ('online', 'version', '.')] 

[('A', 'newspaper', 'is', 'the', 'strongest'), ('newspaper', 'is', 'the', 'strongest', 'medium'), ('is', 'the', 'strongest', 'medium', 'for'), ('the', 'strongest', 'medium', 'for', 'news'), ('strongest', 'medium', 'for', 'news', '.'), ('medium', 'for', 'news', '.', 'People'), ('for', 'news', '.', 'People', 'are'), ('news', '.', 'People', 'are', 'reading'), ('.', 'People', 'are', 'reading', 'newspapers'), ('People', 'are', 'reading', 'newspapers', 'for'), ('are', 'reading', 'newspapers', 'for', 'decades'), ('reading', 'newspapers', 'for', 'decades', '.'), ('newspapers', 'for', 'decades', '.', 'It'), ('for', 'decades', '.', 'It', 'has'), ('decades', '.', 'It', 'has', 'a'), ('.', 'It', 'has', 'a', 'huge'), ('It', 'has', 'a', 'huge', 'contribution'), ('has', 'a', 'huge', 'contribution', 'to'), ('a', 'huge', 'contribution', 'to', 'globalization'), ('huge', 'contribution', 'to', 'globalization', '.'), ('contribution', 'to', 'globalization', '.', 'Right'), ('to', 'globalization', '.', 'Right', 'now'), ('globalization', '.', 'Right', 'now', 'because'), ('.', 'Right', 'now', 'because', 'of'), ('Right', 'now', 'because', 'of', 'easy'), ('now', 'because', 'of', 'easy', 'internet'), ('because', 'of', 'easy', 'internet', 'connection'), ('of', 'easy', 'internet', 'connection', ','), ('easy', 'internet', 'connection', ',', 'people'), ('internet', 'connection', ',', 'people', 'don'), ('connection', ',', 'people', 'don', '’'), (',', 'people', 'don', '’', 't'), ('people', 'don', '’', 't', 'read'), ('don', '’', 't', 'read', 'printed'), ('’', 't', 'read', 'printed', 'newspapers'), ('t', 'read', 'printed', 'newspapers', 'often'), ('read', 'printed', 'newspapers', 'often', '.'), ('printed', 'newspapers', 'often', '.', 'They'), ('newspapers', 'often', '.', 'They', 'read'), ('often', '.', 'They', 'read', 'the'), ('.', 'They', 'read', 'the', 'online'), ('They', 'read', 'the', 'online', 'version'), ('read', 'the', 'online', 'version', '.')] 




