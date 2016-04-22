# Named Entity Recognition and Classification for Entity Extraction
**Using NERC to Extract Entities from Peer-Reviewed Journals**

By Linwood Creekmore III  

The overwhelming amount of unstructured text data available today from traditional media sources as well as newer ones, like social media, provides a rich source of information if the data can be structured.  Named entity extraction forms a core subtask to build knowledge from semi-structured and unstructured text sources<sup><a href="#fn1" id="ref1">1</a></sup>.  Some of the first researchers working to extract information from unstructured texts recognized the importance of “units of information” like names, including person, organization and location names, and numeric expressions including time, date, money and percent expressions.  They coined the term “Named Entity” in 1996 to represent these. Considering recent increases in computing power and decreases in the costs of data storage, data scientists and developers can build large knowledge bases that contain millions of entities and hundreds of millions of facts about them. These knowledge bases are key contributors to intelligent computer behavior<sup><a href="#fn2" id="ref2">2</a></sup>.  Not surprisingly, named entity extraction operates at the core of several popular technologies such as smart assistants ([Siri](http://www.apple.com/ios/siri/), [Google Now](https://www.google.com/landing/now/)), machine reading, and deep interpretation of natural language<sup><a href="#fn3" id="ref3">3</a></sup>.

This post explores how to perform named entity extraction, formally known as “[Named Entity Recognition and Classification (NERC)](https://benjamins.com/catalog/bct.19).  In addition, the article surveys open-source NERC tools that work with Python and compares the results obtained using them against hand-labeled data. The specific steps include: preparing semi-structured natural language data for ingestion using regular expressions; creating a custom corpus in the [Natural Language Toolkit](http://www.nltk.org/); using a suite of open source NERC tools to extract entities and store them in JSON format; comparing the performance of the NERC tools, and implementing a simplistic ensemble classifier. The information extraction concepts and tools in this article constitute a first step in the overall process of structuring unstructured data.  They can be used to perform more complex natural language processing to derive unique insights from large collections of unstructured data.
<br>
# Environment Set Up

To recreate the work in this article, use [Anaconda](https://www.continuum.io/why-anaconda), which is an easy-to-install, free, enterprise-ready Python distribution for data analytics, processing, and scientific computing (reference). With a few lines of code, you can have all the dependencies used in this post with the exception of one function (email extractor). 

*  [Install Anaconda](http://docs.continuum.io/anaconda/install)

*  Download the [namedentity_requirements.yml](https://github.com/linwoodc3/LC3-Creations/blob/master/DDL/namedentityblog/namedentity_requirements.yml) (remember where you saved it on your computer)
*  Follow the ["Use Environment from file"](http://conda.pydata.org/docs/using/envs.html#use-environment-from-file) instructions on Anaconda's website.

If you use an alternative method to set up a virtual environment, make sure you have all the files installed from the yml file. The one dependency not in the yml file is the email extractor. [Cut and paste the function from this website](https://gist.github.com/dideler/5219706), save it to a .py file, and make sure it is in your sys.path or environment path.  If you are [running this as an iPython notebook](https://github.com/linwoodc3/LC3-Creations/blob/master/DDL/namedentityblog/blogdraft.ipynb), stop here. Go to the Appendix and run all of the blocks of code before continuing.


### Data Source
The proceedings from the Knowledge Discovery and Data Mining (KDD) conferences in [New York City (2014)](http://www.kdd.org/kdd2014/) and [Sydney, Australia (2015)](http://www.kdd.org/kdd2015/) serve as our source of unstructured text and contain over 230 peer reviewed journal articles and keynote speaker abstracts on data mining, knowledge discovery, big data, data science and their applications. The full conference proceedings can be purchased for $60 at the [Association for Computing Machinery's Digital Library](https://dl.acm.org/purchase.cfm?id=2783258&CFID=740512201&CFTOKEN=34489585) (includes ACM membership). This post will work with a dataset that is equivalent to the combined conference proceedings and takes the semi-structured data that is in the form of PDF journal articles and abstracts, extracts text from these files, and adds structure to the data to facilitate follow-on analysis.  Interested parties looking for a free option can use the [`beautifulsoup`](https://pypi.python.org/pypi/beautifulsoup4/4.4.1) and [`requests`](https://pypi.python.org/pypi/requests/2.9.1)libraries to [scrape the ACM website for KDD 2015 conference data](https://github.com/linwoodc3/LC3-Creations/blob/master/DDL/namedentityblog/KDDwebscrape.ipynb).

### Initial Data Exploration
Visual inspection reveals that the target filenames begin with a “p” and end with “pdf.” As a first step, we determine the number of files and the naming conventions by using a loop to iterate over the files in the directory and printing out the filenames.  Each filename also gets saved to a list, and the length of the list tells us the total number of files in the dataset.


```python
##############################################
# Administrative code: Import what we need
##############################################
import os
import time
from os import walk

###############################################
# Set the Path
##############################################

path        = os.path.abspath(os.getcwd())

# Path to directory where KDD files are
TESTDIR     = os.path.normpath(os.path.join(os.path.expanduser("~"),"Desktop","KDD_15","docs"))

# Establish an empty list to append filenames as we iterate over the directory with filenames
files = []


###############################################
# Code to iterate over files in directory
##############################################

'''Iterate over the directory of filenames and add to list.  
Inspection shows our target filenames begin with 'p' and end with 'pdf' '''

for dirName, subdirList, fileList in os.walk(TESTDIR):
    for fileName in fileList:
        if fileName.startswith('p') and fileName.endswith('.pdf'):
            files.append(fileName)
end_time = time.time()

###############################################
# Output
###############################################print
print len(files) # Print the number of files
print 
#print '[%s]' % ', '.join(map(str, files)) # print the list of filenames
```

<br>A total of 253 files exist in the directory. Opening one of these reveals that our data is in PDF format and it's semi-structured (follows journal article format with separate sections for "abstract" and "title"). While PDFs provide an easily readable presentation of data, they are extremely difficult to work with in data analysis. In your work, if you have an option to get to data before conversion to a PDF format, be sure to take that option.<br><br>

### Creating a Custom NLTK Corpus

We used several Python tools to ingest our data including: [`pdfminer`](https://pypi.python.org/pypi/pdfminer/), [`subprocess`](https://docs.python.org/2/library/subprocess.html),  [`nltk`](http://www.nltk.org/), [`string`](https://docs.python.org/2/library/string.html), and [`unicodedata`](https://docs.python.org/2/library/unicodedata.html).  Pdfminer contains a command line tool called “pdf2txt.py” that extracts text contents from a PDF file (visit the [`pdfminer homepage`](http://euske.github.io/pdfminer/index.html#pdf2txt) for download instructions).  Subprocess, a standard library module, allows us to invoke the “pdf2txt.py” command line tool within our code.  The Natural Language Tool Kit, or NLTK, serves as one of Python’s leading platforms to analyze natural language data.  The string module provides variable substitutions and value formatting to strip non-printable characters from the output of the text extracted from our journal article PDFs.  Finally, the unicodedata library allows Latin Unicode characters to degrade gracefully into ASCII.  This is an important feature because some Unicode characters won’t extract nicely.

Our task begins by iterating over the files in the directory with names that begin with 'p' and end with 'pdf.' This time, however, we will strip the text from the pdf file, write the .txt file to a newly created directory, and use the fileName variable to name the files we write to disk. Keep in mind that this task may take a few minutes depending on the processing power of your computer. Next, we use the simple instructions from Section 1.9, Chapter 2 of NLTK's Book to build a custom corpus. Having our target documents loaded as an NLTK corpus brings the power of NLTK to our analysis goals. Here's the code to accomplish what's discussed above:


```python
###############################################
# Importing what we need
###############################################

import string
import unicodedata
import subprocess
import nltk
import os, os.path
import re

###############################################
# Create the directory we will write the .txt files to after stripping text
###############################################

# path where KDD journal files exist on disk or cloud drive access
corpuspath = os.path.normpath(os.path.expanduser('~/Desktop/KDD_corpus/'))
if not os.path.exists(corpuspath):
    os.mkdir(corpuspath)
```


```python


###############################################
# Core code to iterate over files in the directory
###############################################

# We start from the code to iterate over the files
%timeit

for dirName, subdirList, fileList in os.walk(TESTDIR):
    for fileName in fileList:
        if fileName.startswith('p') and fileName.endswith('.pdf'):
            if os.path.exists(os.path.normpath(os.path.join(corpuspath,fileName.split(".")[0]+".txt"))):
                pass
            else:
            
            
###############################################
# This code strips the text from the PDFs
###############################################
                try:
                    document = filter(lambda x: x in string.printable,
                                      unicodedata.normalize('NFKD', 
                                                            (unicode(subprocess.check_output(['pdf2txt.py',str(os.path.normpath(os.path.join(TESTDIR,fileName)))]),
                                                                     errors='ignore'))).encode('ascii','ignore').decode('unicode_escape').encode('ascii','ignore'))
                except UnicodeDecodeError:
                    document = unicodedata.normalize('NFKD',
                                                     unicode(subprocess.check_output(['pdf2txt.py',str(os.path.normpath(os.path.join(TESTDIR,fileName)))]),errors='ignore')).encode('ascii','ignore')    

                if len(document)<300:
                    pass
                else:
                    # used this for assistance http://stackoverflow.com/questions/2967194/open-in-python-does-not-create-a-file-if-it-doesnt-exist
                    if not os.path.exists(os.path.normpath(os.path.join(corpuspath,fileName.split(".")[0]+".txt"))):
                        file = open(os.path.normpath(os.path.join(corpuspath,fileName.split(".")[0]+".txt")), 'w+')
                        file.write(document)
                    else:
                        pass

```


```python
# This code builds our custom corpus.  The corpus path is a path to where we saved all of our .txt files of stripped text                    
kddcorpus= nltk.corpus.PlaintextCorpusReader(corpuspath, '.*\.txt')
```

<br>
We now have a semi-structured dataset in a format that we can query and analyze the different pieces of data. Let's see how many words (including stop words) we have in our entire corpus. <br><br>


```python
# Mapping, setting count to zero for start
wordcount = 0

#Iterating over list and files and counting length
for fileid in kddcorpus.fileids():
    wordcount += len(kddcorpus.words(fileid))
print wordcount
```

    2795267


<br>The NLTK book has an excellent [section on processing raw text and unicode issues](http://www.nltk.org/book/ch03.html#fig-unicode). It provides a helpful discussion of some problems you may encounter. 

###Using Regular Expressions to Extract Specific Sections 

<br>
To begin our exploration of regular expressions (aka "regex"), it's important to point out some good resources for those new to the topic. An excellent resource may be found in [Videos 1-3, Week 4, Getting and Cleaning Data, Data Science Specialization Track from Johns Hopkins University](https://www.coursera.org/learn/data-cleaning).   Additional resources appear in the Appendix.  As a simple example, let’s extract titles from the first 26 documents. 


```python
# Using metacharacters vice literal matches
p=re.compile('^(.*)([\s]){2}[A-z]+[\s]+[\s]?.+')

for fileid in kddcorpus.fileids()[:25]:
    print re.search('^(.*)[\s]+[\s]?(.*)?',kddcorpus.raw(fileid)).group(1).strip()+" "+re.search('^(.*)[\s]+[\s]?(.*)?',kddcorpus.raw(fileid)).group(2).strip()
```

    Online Controlled Experiments: Lessons from Running A/B/n Tests for 12 Years
    Mining Frequent Itemsets through Progressive Sampling with Rademacher Averages
    Why It Happened: Identifying and Modeling the Reasons of the Happening of Social Events
    Matrix Completion with Queries Natali Ruchansky
    Stochastic Divergence Minimization for Online Collapsed Variational Bayes Zero Inference
    Bayesian Poisson Tensor Factorization for Inferring Multilateral Relations from Sparse Dyadic Event Counts
    TimeCrunch: Interpretable Dynamic Graph Summarization Neil Shah
    Inside Jokes: Identifying Humorous Cartoon Captions Dafna Shahaf
    Community Detection based on Distance Dynamics Junming Shao
    Discovery of Meaningful Rules in Time Series Mohammad Shokoohi-Yekta    Yanping Chen    Bilson Campana    Bing Hu
    On the Formation of Circles in Co-authorship Networks Tanmoy Chakraborty1, Sikhar Patranabis2, Pawan Goyal3, Animesh Mukherjee4
    An Evaluation of Parallel Eccentricity Estimation Algorithms on Undirected Real-World Graphs
    Efcient Latent Link Recommendation in Signed Networks
    Turn Waste into Wealth: On Simultaneous Clustering and Cleaning over Dirty Data
    Set Cover at Web Scale Stergios Stergiou
    Exploiting Relevance Feedback in Knowledge Graph Search
    LINKAGE: An Approach for Comprehensive Risk Prediction for Care Management
    Transitive Transfer Learning Ben Tan
    PTE: Predictive Text Embedding through Large-scale Heterogeneous Text Networks
    An Effective Marketing Strategy for Revenue Maximization with a Quantity Constraint
    Scaling Up Stochastic Dual Coordinate Ascent Kenneth Tran
    Heterogeneous Network Embedding via Deep Architectures
    Discovering Valuable Items from Massive Data Hastagiri P Vanchinathan
    Deep Learning Architecture with Dynamically Programmed Layers for Brain Connectome Prediction
    Incorporating World Knowledge to Document Clustering via Heterogeneous Information Networks


<br>This code extracts the titles, but some author names get caught up in the extraction as well. 

For simplicity, let's focus on wrangling the data to use the NERC tools on two sections of the paper: the “top” section and the “references” section.  The “top” section includes the names of authors and schools.  This section represents all of the text above the article’s abstract.  The “references” section appears at the end of the article. The regex tools of choice to extract sections are the [`positive lookbehind` and `positive lookahead`](https://docs.python.org/2/library/re.html) expressions.  We build two functions designed to extract the “top” and “references” sections of each document. 

First a few words about the data.  When working with natural language, one should always be prepared to deal with irregularities in the data set. This corpus is no exception. It comes from a top-notch data mining organization, but human error and a lack of standardization makes its way into the picture.  For example, in one paper the header section is entitled “Categories and Subject Descriptors,” while in another the title is “Categories & Subject Descriptors.” While that may seem like a small difference, these types of differences cause significant problems.  There are also some documents that will be missing sections altogether, i.e. keynote speaker documents do not contain a “references” section. When encountering similar issues in your work, you must decide whether to account for these differences or ignore them. I worked to include as much of the 253-document corpus as possible. 

In addition to extracting the relevant sections of the documents, our two functions will obtain a character count for each section, extract emails, count the number of references and store that value, calculate a word per reference count, and store all the above data as a nested dictionary with filenames as the key. For simplicity, we show below the code to extract the “references” section and include the function for extracting the “top” section in the Appendix. <br>


```python


# Code to pull the references section only, store a character count, number of references, and "word per reference" calculation

def refpull(docnum=None,section='references',full = False):
    
    # Establish an empty dictionary to hold values
    ans={}
    
    # Establish an empty list to hold document ids that don't make the cut (i.e. missing reference section or different format)
    # This comes in handy when you are trying to improve your code to catch outliers
    failids = []
    section = section.lower()    
    
    # Admin code to set default values and raise an exception if there's human error on input
    if docnum is None and full == False:
        raise BaseException("Enter target file to extract data from")
    
    if docnum is None and full == True:
        
        # Setting the target document and the text we will extract from 
        text=kddcorpus.raw(docnum)
        
        
        # This first condtional is for pulling the target section for ALL documents in the corpus
        if full == True:
            
            # Iterate over the corpus to get the id; this is possible from loading our docs into a custom NLTK corpus
            for fileid in kddcorpus.fileids():
                text = kddcorpus.raw(fileid)
                
                # These lines of code build our regular expression.
                # In the other functions for abstract or keywords, you see how I use this technique to create different regex arugments
                if section == "references":
                    section1=["REFERENCES"] 
                    
                    # Just in case, making sure our target string is empty before we pass data into it; just a check
                    target = ""   

                    #We now build our lists iteratively to build our regex
                    for sect in section1:
                        
                        # We embed exceptions to remove the possibility of our code stopping; we pass failed passes into a list
                        try:
                            
                            # our machine built regex
                            part1= "(?<="+sect+")(.+)"
                            p=re.compile(part1)
                            target=p.search(re.sub('[\s]'," ",text)).group(1)
                            
                            # Conditoin to make sure we don't get any empty string
                            if len(target) > 50:

                                # calculate the number of references in a journal; finds digits between [] in references section only
                                try:
                                    refnum = len(re.findall('\[(\d){1,3}\]',target))+1
                                except:
                                    print "This file does not appear to have a references section"
                                    pass
                                
                                #These are all our values; we build a nested dictonary and store the calculated values
                                ans[str(fileid)]={}
                                ans[str(fileid)]["references"]=target.strip()
                                ans[str(fileid)]["charcount"]=len(target)
                                ans[str(fileid)]["refcount"]= refnum
                                ans[str(fileid)]["wordperRef"]=round(float(len(nltk.word_tokenize(text)))/float(refnum))
                                #print [fileid,len(target),len(text), refnum, len(nltk.word_tokenize(text))/refnum]
                                break
                            else:

                                pass
                        except AttributeError:
                            failids.append(fileid)
                            pass

            return ans
            return failids
                              
        # This is to perform the same operations on just one document; same functionality as above.
    else:
        ans = {}
        failids=[]
        text = kddcorpus.raw(docnum)
        
        if section == "references":
            section1=["REFERENCES"] 
            target = ""   
            for sect in section1:
                try:
                    part1= "(?<="+sect+")(.+)"
                    p=re.compile(part1)
                    target=p.search(re.sub('[\s]'," ",text)).group(1)
                    if len(target) > 50:
                        # calculate the number of references in a journal; finds digits between [] in references section only
                        try:
                            refnum = len(re.findall('\[(\d){1,3}\]',target))+1
                        except:
                            print "This file does not appear to have a references section"
                            pass
                        ans[str(docnum)]={}
                        ans[str(docnum)]["references"]=target.strip()
                        ans[str(docnum)]["charcount"]=len(target)
                        ans[str(docnum)]["refcount"]= refnum
                        ans[str(docnum)]["wordperRef"]=float(len(nltk.word_tokenize(text)))/float(refnum)


                        #print [fileid,len(target),len(text), refnum, len(nltk.word_tokenize(text))/refnum]
                        break
                    else:

                        pass
                except AttributeError:
                    failids.append(docnum)
                    pass
        
        
        
        return ans
        return failids
```

<br>
The above code also makes use of the `nltk.word_tokenize` tool to create the "word per reference" statistic (takes time to run). 
Let's test the “references” extraction function and look at the output by obtaining the first 10 entries of the dictionary created by the function.  This dictionary holds all the extracted data and various calculations.


```python
# call our function, setting "full=True" extracts ALL references in corpus
test = refpull(full=True)

# To get a quick glimpse, I use the example from this page: http://stackoverflow.com/questions/7971618/python-return-first-n-keyvalue-pairs-from-dict
import itertools
import collections

man = collections.OrderedDict(test)

x = itertools.islice(man.items(), 0, 10)
```

The [`tabulate`](https://pypi.python.org/pypi/tabulate) module is a great tool to visualize descriptive outputs in table format. 


```python
from tabulate import tabulate

# A quick list comprehension to follow the example on the tabulate pypi page
table = [[key,value['charcount'],value['refcount'], value['wordperRef']] for key,value in x]

# print the pretty table; we invoke the "header" argument and assign custom header!!!!
print tabulate(table,headers=["filename","Character Count", "Number of references","Words per Reference"])
```

    filename      Character Count    Number of references    Words per Reference
    ----------  -----------------  ----------------------  ---------------------
    p2277.txt                6295                      33                    326
    p835.txt                 5347                      38                    319
    p865.txt                 5269                      27                    399
    p2089.txt                8734                      45                    181
    p1759.txt                3677                      31                    405
    p29.txt                  5101                      40                    265
    p2227.txt               10345                      36                    332
    p2099.txt                3949                      28                    374
    p725.txt                 5771                      37                    304
    p2019.txt                9101                      60                    171


<br>

### Open Source NERC Tools: NLTK, Stanford NER and Polyglot


Now that we have a method to obtain the corpus from the “top” and “references” sections of each article in the dataset, we are ready to perform the named entity extractions.  In this post, we examine three popular, open source NERC tools. The tools are NLTK, Stanford NER, and Polyglot.  A brief description of each follows.

[`NLTK has a chunk package`](http://www.nltk.org/api/nltk.chunk.html) that uses NLTK’s recommended named entity chunker to chunk the given list of tagged tokens. A string is tokenized and tagged with parts of speech (POS) tags.  The NLTK chunker then identifies non-overlapping groups and assigns them to an entity class. You can read more about NLTK's chunking capabilities in [the NLTK book](http://www.nltk.org/book/ch07.html).

[`Standard's Named Entity Recognizer`](http://nlp.stanford.edu/software/CRF-NER.shtml), often called Stanford NER, is a Java implementation of linear chain Conditional Random Field (CRF) sequence models functioning as a Named Entity Recognizer. Named Entity Recognition (NER) labels sequences of words in a text that are the names of things, such as person and company names, or gene and protein names. NLTK contains an [interface to Stanford NER](http://www.nltk.org/_modules/nltk/tag/stanford.html) written by Nitin Madnani. Details for [using the Stanford NER tool](http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford) are on the NLTK page and the required jar files can be downloaded [here](http://nlp.stanford.edu/software/index.shtml).

[`Polyglot`](http://polyglot.readthedocs.org/en/latest/index.html) is a natural language pipeline that supports massive multilingual (i.e. language) applications. It supports tokenization in 165 languages, language detection in 196 languages, named entity recognition in 40 languages, part of speech tagging in 16 languages, sentiment analysis in 136 languages, word embeddings in 137 languages, morphological analysis in 135 languages, and transliteration in 69 languages. It is a powerhouse tool for natural language processing. We will use the named entity recognition feature for English language in this exercise. Polyglot is available via pypi.


We can now test how well these open source NERC tools extract entities from the “top” and “reference” sections of our corpus. For two documents, I hand labeled authors, organizations, and locations from the “top” section of the article (section before the abstract) and the list of all authors from the “references” section.  I also created a combined list of the authors, joining the lists from the “top” and “references” sections. Hand labeling is a time consuming and tedious process. For just the two (2) documents, this involved 295 cut-and-pastes of names or organizations. The annotated list appears in the Appendix.

An easy test for the accuracy of a NERC tool is to compare the entities extracted by the tools to the hand-labeled extractions.  Before beginning, we take advantage of the NLTK functionality to obtain the “top” and “references” sections of the two documents used for the hand labeling:



```python
# We need the top and references sections from p19.txt and p29.txt

p19={'top': toppull("p19.txt")['p19.txt']['top'], 'references':refpull("p19.txt")['p19.txt']['references']}
p29={'top': toppull("p29.txt")['p29.txt']['top'], 'references':refpull("p29.txt")['p29.txt']['references']}
```


<br>In this next block of code, we will apply the NLTK standard chunker, Stanford Named Entity Recognizer, and Polyglot extractor to our corpus. For each NERC tool, I created functions (available in the Appendix) to extract entities and return classes of objects in different lists. If you are following along, you should have run all the code blocks in the Appendix. If not, go there and do it now. The functions (in appendix) are:

*  **nltktreelist** - NLTK Standard Chunker
*  **get_continuous_chunks** - Stanford Named Entity Recognizer
*  **extraction** - Polyglot Extraction tool

For illustration, the Polyglot Extraction tool function, extraction, appears below:<br><br>



```python
def extraction(corpus):
    import itertools
    import unicodedata
    from polyglot.text import Text
    
    corpus=corpus
    # extract entities from a single string; remove whitespace characters
    try:
        e = Text(corpus).entities
    except:
        pass #e = Text(re.sub("(r'(x0)'," ","(re.sub('[\s]'," ",corpus)))).entities
    
    current_person =[]
    persons =[]
    current_org=[]
    organizations=[]
    current_loc=[]
    locations=[]

    for l in e:
        if l.tag == 'I-PER':
            for m in l:
                current_person.append(unicodedata.normalize('NFKD', m).encode('ascii','ignore'))
            else:
                    if current_person: # if the current chunk is not empty
                        persons.append(" ".join(current_person))
                        current_person = []
        elif l.tag == 'I-ORG':
            for m in l:
                current_org.append(unicodedata.normalize('NFKD', m).encode('ascii','ignore'))
            else:
                    if current_org: # if the current chunk is not empty
                        organizations.append(" ".join(current_org))
                        current_org = []
        elif l.tag == 'I-LOC':
            for m in l:
                current_loc.append(unicodedata.normalize('NFKD', m).encode('ascii','ignore'))
            else:
                    if current_loc: # if the current chunk is not empty
                        locations.append(" ".join(current_loc))
                        current_loc = []
    results = {}
    results['persons']=persons
    results['organizations']=organizations
    results['locations']=locations
    
    return results
```

<br>We pass our data, the “top” and “references” section of the two documents of interest, into the functions created with each NERC tool and build a nested dictionary of the extracted entities—author names, locations, and organization names. This code may take a bit of time to run (30 secs to a minute). <br><br>


```python
###############################################
    #  NLTK Standard Chunker
###############################################
    
nltkstandard_p19ents = {'top': nltktreelist(p19['top']),'references': nltktreelist(p19['references'])}
nltkstandard_p29ents = {'top': nltktreelist(p29['top']),'references': nltktreelist(p29['references'])}

###############################################
# Stanford NERC Tool
################################################
from nltk.tag import StanfordNERTagger, StanfordPOSTagger
stner = StanfordNERTagger('/Users/linwood/stanford-corenlp-full/classifiers/english.muc.7class.distsim.crf.ser.gz',
       '/Users/linwood/stanford-corenlp-full/stanford-corenlp-3.5.2.jar',
       encoding='utf-8')
stpos = StanfordPOSTagger('/Users/linwood/stanford-postagger-full/models/english-bidirectional-distsim.tagger','/Users/linwood/stanford-postagger-full/stanford-postagger.jar') 

stan_p19ents = {'top': get_continuous_chunks(p19['top']), 'references': get_continuous_chunks(p19['references'])}
stan_p29ents = {'top': get_continuous_chunks(p29['top']), 'references': get_continuous_chunks(p29['references'])}

###############################################
# Polyglot NERC Tool
###############################################
poly_p19ents = {'top': extraction(p19['top']), 'references': extraction(p19['references'])}
poly_p29ents = {'top': extraction(p29['top']), 'references': extraction(p29['references'])}
```

<br>
We will focus specifically on the "persons" entity extractions from the “top” section of the documents to estimate performance.  However, a similar exercise is possible with the extractions of “organizations” entity extractions or “locations” entity extractions too, as well as from the “references” section. To get a better look at how each NERC tool performed on the named person entities, we will use the `Pandas` dataframe.[`Pandas`](http://pandas.pydata.org/) is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language. The dataframe provides a visual comparison of the extractions from each NERC tool and the hand-labeled extractions. Just a few lines of code accomplish the task:



```python
#################################################################
# Administrative code, importing necessary library or module
#################################################################

import pandas as pd

#################################################################
# Create pandas series for each NERC tool entity extraction group
#################################################################

df1 = pd.Series(poly_p19ents['top']['persons'], index=None, dtype=None, name='Polyglot NERC Authors', copy=False, fastpath=False)
df2=pd.Series(stan_p19ents['top']['persons'], index=None, dtype=None, name='Stanford NERC Authors', copy=False, fastpath=False)
df3=pd.Series(nltkstandard_p19ents['top']['persons'], index=None, dtype=None, name='NLTKStandard NERC Authors', copy=False, fastpath=False)
df4 = pd.Series(p19pdf_authors, index=None, dtype=None, name='Hand-labeled True Authors', copy=False, fastpath=False)
met = pd.concat([df4,df3,df2,df1], axis=1).fillna('')
met
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hand-labeled True Authors</th>
      <th>NLTKStandard NERC Authors</th>
      <th>Stanford NERC Authors</th>
      <th>Polyglot NERC Authors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tim Althoff</td>
      <td>Timeline</td>
      <td>Tim Althoff</td>
      <td>Tim Althoff</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Xin Luna Dong</td>
      <td>Tim Althoff</td>
      <td>Xin Luna Dong</td>
      <td>Xin Luna Dong</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kevin Murphy</td>
      <td>Xin Luna Dong</td>
      <td>Kevin Murphy</td>
      <td>Kevin Murphy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Safa Alai</td>
      <td>Kevin Murphy</td>
      <td></td>
      <td>Safa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Van Dang</td>
      <td>Safa Alai</td>
      <td></td>
      <td>Van Dang</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Wei Zhang</td>
      <td>Van Dang</td>
      <td></td>
      <td>Wei Zhang</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Wei Zhang</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>Stanford</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td>Mountain View</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



<br>
The above dataframe illustrates the mixed results from the NERC tools.  NLTK Standard NERC appears to have extracted 3 false positives while the Stanford NERC missed 3 true positives and the Polyglot NERC extracted all but one true positive (partially extracted; returned first name only). Let's calculate some key performance metrics:<br>
1.  **TN or True Negative**: case was negative and predicted negative <br>
2.  **TP or True Positive**: case was positive and predicted positive <br>
3.  **FN or False Negative**: case was positive but predicted negative <br>
4.  **FP or False Positive**: case was negative but predicted positive<br>

The following function calculates the above metrics for the three NERC tools:



```python
# Calculations and logic from http://www.kdnuggets.com/faq/precision-recall.html

def metrics(truth,run):
    truth = truth
    run = run
    TP = float(len(set(run) & set(truth)))

    if float(len(run)) >= float(TP):
        FP = len(run) - TP
    else:
        FP = TP - len(run)
    TN = 0
    if len(truth) >= len(run):
        FN = len(truth) - len(run)
    else:
        FN = 0

    accuracy = (float(TP)+float(TN))/float(len(truth))
    recall = (float(TP))/float(len(truth))
    precision = float(TP)/(float(FP)+float(TP))
    print "The accuracy is %r" % accuracy
    print "The recall is %r" % recall
    print "The precision is %r" % precision
    
    d = {'Predicted Negative': [TN,FN], 'Predicted Positive': [FP,TP]}
    metricsdf = pd.DataFrame(d, index=['Negative Cases','Positive Cases'])
    
    return metricsdf 
```

<br>Now let's pass our values into the function to calculate the performance metrics:<br><br>


```python
print
print
str1 = "NLTK Standard NERC Tool Metrics"

print str1.center(40, ' ')
print
print
metrics(p19pdf_authors,nltkstandard_p19ents['top']['persons'])
```

    
    
        NLTK Standard NERC Tool Metrics     
    
    
    The accuracy is 1.0
    The recall is 1.0
    The precision is 0.6666666666666666





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted Negative</th>
      <th>Predicted Positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Negative Cases</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Positive Cases</th>
      <td>0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
print
print
str2 = "Stanford NERC Tool Metrics"

print str2.center(40, ' ')
print
print
metrics(p19pdf_authors, stan_p19ents['top']['persons'])
```

    
    
           Stanford NERC Tool Metrics       
    
    
    The accuracy is 0.5
    The recall is 0.5
    The precision is 1.0





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted Negative</th>
      <th>Predicted Positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Negative Cases</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Positive Cases</th>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
print
print
str3 = "Polyglot NERC Tool Metrics"

print str3.center(40, ' ')
print
print
metrics(p19pdf_authors,poly_p19ents['top']['persons'])
```

    
    
           Polyglot NERC Tool Metrics       
    
    
    The accuracy is 0.8333333333333334
    The recall is 0.8333333333333334
    The precision is 0.8333333333333334





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted Negative</th>
      <th>Predicted Positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Negative Cases</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Positive Cases</th>
      <td>0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



**Note to Ben from Selma** - *I think there might be a mistake in the table for the Polyglot NERC. Missing a 1 in the lower left maybe?* 

The basic metrics above reveal some quick takeaways about each tool based on the specific extraction task. The `NLTK Standard Chunker` has perfect accuracy and recall but lacks in precision. It successfully extracted all the authors for the document, but also extracted 3 false entities. NLTK's chunker would serve well in an entity extraction pipeline where the data scientist is concerned with identifying all possible entities

The `Stanford NER tool` is very precise (specificity vs sensitivity). The entities it extracts were 100% accurate, but it failed to identify half of the true entities. The Stanford NER tool would be best used when a data scientist wanted to extract only those entities that have a high likelihood of being named entities, suggesting an unconscious acceptance of leaving behind some information.

The `Polyglot Named Entity Recognizer` identified five named entities exactly, but only partially identified the sixth (first name returned only). The data scientist looking for a balance between sensitivity and specificity would likely use Polyglot, as it will balance extracting the 100% accurate entities and those which may not necessarily be a named entity.

### A Simple Ensemble Classifier

In our discussion above, we notice the varying levels of performance by the different NERC tools. Using the idea that combining the outputs from various classifiers in an ensemble method can improve the reliability of classifications, we can improve the performance of our named entity extractor tools by creating an ensemble classifier. Each NERC tool had at least 3 named persons that were true positives, but no two NERC tools had the same false positive or false negative. Our ensemble classifier "voting" rule is very simple: “Return all named entities that exist in at least two of the true positive named entity result sets from our NERC tools.
We implement this rule using the `set` module. We first do an `intersection` operation of the NERC results vs the hand labeled entities to get our "true positive" set. Here is our code to accomplish the task:



```python
# Create intersection of true authors from NLTK standard output
a =set(sorted(nltkstandard_p19ents['top']['persons'])) & set(p19pdf_authors)

# Create intersection of true authors from Stanford NER output
b =set(sorted(stan_p19ents['top']['persons'])) & set(p19pdf_authors)

# Create intersection of true authors from Polyglot output
c = set(sorted(poly_p19ents['top']['persons'])) & set(p19pdf_authors)

# Create union of all true positives from each NERC output
(a.union(b)).union(c)
```




    {'Kevin Murphy',
     'Safa Alai',
     'Tim Althoff',
     'Van Dang',
     'Wei Zhang',
     'Xin Luna Dong'}



To get a visual comparison of the extractions for each tool and the ensemble set side by side, we return to our dataframe from earlier. In this case, we use the `concat` operation in `pandas` to append the new ensemble set to the dataframe. Our code to accomplish the task is:


```python
dfensemble = pd.Series(list((a.union(b)).union(c)), index=None, dtype=None, name='Ensemble Method Authors', copy=False, fastpath=False)
met = pd.concat([df4,dfensemble,df3,df2,df1], axis=1).fillna('')
met
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hand-labeled True Authors</th>
      <th>Ensemble Method Authors</th>
      <th>NLTKStandard NERC Authors</th>
      <th>Stanford NERC Authors</th>
      <th>Polyglot NERC Authors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tim Althoff</td>
      <td>Wei Zhang</td>
      <td>Timeline</td>
      <td>Tim Althoff</td>
      <td>Tim Althoff</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Xin Luna Dong</td>
      <td>Tim Althoff</td>
      <td>Tim Althoff</td>
      <td>Xin Luna Dong</td>
      <td>Xin Luna Dong</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kevin Murphy</td>
      <td>Xin Luna Dong</td>
      <td>Xin Luna Dong</td>
      <td>Kevin Murphy</td>
      <td>Kevin Murphy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Safa Alai</td>
      <td>Van Dang</td>
      <td>Kevin Murphy</td>
      <td></td>
      <td>Safa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Van Dang</td>
      <td>Kevin Murphy</td>
      <td>Safa Alai</td>
      <td></td>
      <td>Van Dang</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Wei Zhang</td>
      <td>Safa Alai</td>
      <td>Van Dang</td>
      <td></td>
      <td>Wei Zhang</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td></td>
      <td>Wei Zhang</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td></td>
      <td>Stanford</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td></td>
      <td>Mountain View</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



And we get a look at the performance metrics to see if we push our scores up in all categories:


```python
print
print
str = "Ensemble NERC Metrics"

print str.center(40, ' ')
print
print
metrics(p19pdf_authors,list((a.union(b)).union(c)))
```

    
    
             Ensemble NERC Metrics          
    
    
    The accuracy is 1.0
    The recall is 1.0
    The precision is 1.0





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted Negative</th>
      <th>Predicted Positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Negative Cases</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Positive Cases</th>
      <td>0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




<br>Exactly as expected, we see improved performance across all performance metric scores and in the end get a perfect extraction of all named persons from this document. Before we go ANY further, the idea of moving from "okay" to "perfect" is unrealistic. Moreover, this is a very small sample and only intended to show the application of an ensemble method. Applying this method to other sections of the journal articles will not lead to a perfect extraction, but it will indeed improve the performance of the extraction considerably.

### Getting Your Data in Open File Format

A good rule for any data analytics project is to store the results or output in an open file format. Why? An [open file format is a published specification for storing digital data, usually maintained by a standards organization, and which can be used and implemented by anyone](https://en.wikipedia.org/wiki/Open_format).  I selected [`JavaScript Object Notation(JSON)`](https://en.wikipedia.org/wiki/JSON), which is an open standard format that uses human-readable text to transmit data objects consisting of attribute–value pairs. We take our list of persons from the ensemble results, store it as a Python dictionary, and then convert it to JSON. Alternatively, we could use the `dumps` function from the `json` module to return dictionaries, and ensure we get the open file format at every step. 

In this way, other data scientists or users could pick and choose what portions of code to use in their projects. Here is our code to accomplish the task:



```python
import json

# Add ensemble results for author to the nested python dictionary
p19['authors']= list((a.union(b)).union(c))

# covert nested dictionary to json for open data storage
# json can be stored in mongodb or any other disk store
output = json.dumps(p19, ensure_ascii=False,indent=3)

# print out the authors section we just created in our json
print json.dumps(json.loads(output)['authors'],indent=3)

# uncomment to see full json output
#print json.dumps((json.loads(output)),indent=3)
```

    [
       "Wei Zhang", 
       "Tim Althoff", 
       "Xin Luna Dong", 
       "Van Dang", 
       "Kevin Murphy", 
       "Safa Alai"
    ]


### Conclusion

We covered the entire data science pipeline in a natural language processing job that compared the performance of three different NERC tools. A core task in this pipeline involved ingesting plaintext into an NLTK corpus so that we could easily retrieve and manipulate the corpus. Finally, we used the results from the various NERC tools to create a simplistic ensemble classifier that improved the overall performance.

The techniques in this post can be applied to other domains, larger datasets or any other corpus. Everything I used in this post (with the exception of the Regular expression resource from Coursera) was not taught in a classroom or structured learning experience. It all came from online resources, posts from others, and books (that includes learning how to code in Python). If you have the motivation, you can do it. 

Throughout the article, there are hyperlinks to resources and reading materials for reference, but here is a central list:

* [Requirements to run this code in iPython notebook or on your machine](https://github.com/linwoodc3/LC3-Creations/blob/master/DDL/namedentityblog/namedentity_requirements.yml)
* [Natural Language Toolkit Book (free online resource)](http://www.nltk.org/book/) and the [NLTK Standard Chunker](http://www.nltk.org/_modules/nltk/chunk/named_entity.html) and a [post on how to use the chunker](http://stackoverflow.com/questions/19312573/nltk-for-named-entity-recognition)
* [Polyglot natural language pipeline for massive muliligual applications](https://pypi.python.org/pypi/polyglot) and the [journal article describing the word classification model](http://arxiv.org/pdf/1410.3791.pdf)
* [Stanford Named Entity Recognizer](http://nlp.stanford.edu/software/CRF-NER.shtml) and the [NLTK interface to the Stanford NER](http://www.nltk.org/_modules/nltk/tag/stanford.html) and a [post on how to use the interface](http://textminingonline.com/how-to-use-stanford-named-entity-recognizer-ner-in-python-nltk-and-other-programming-languages)
* [Python Pandas](http://pandas.pydata.org/) is a must have tool for anyone who does analysis in Python.  The best book I've used to date is [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://play.google.com/store/books/details?id=v3n4_AK8vu0C&source=productsearch&utm_source=HA_Desktop_US&utm_medium=SEM&utm_campaign=PLA&pcampaignid=MKTAD0930BO1&gl=US&gclid=COnf8Z_BncoCFYKvNwodVA4ItA&gclsrc=ds)
* [Intuitive description and examples of Python's standard library set module](http://www.linuxtopia.org/online_books/programming_books/python_programming/python_ch16s03.html)
* [Discussion of ensemble classifiers](http://arxiv.org/pdf/1404.4088.pdf)
* [Nice module to print tables in standard python output called tablulate](https://pypi.python.org/pypi/tabulate)
* [Regular expression training (more examples in earlier sections)](http://regexone.com/)
* [Python library to extract text from PDF](http://euske.github.io/pdfminer/index.html) and [post on available Python tools to extract text from a PDF](https://www.binpress.com/tutorial/manipulating-pdfs-with-python/167)
* [ACM Digital Library](http://dl.acm.org/) to [purchase journal articles to completely recreate this exercise](https://dl.acm.org/purchase.cfm?id=2783258&CFID=740512201&CFTOKEN=34489585)
* My [quick web scrap code to pull back abstracts and authors from KDD 2015](https://github.com/linwoodc3/LC3-Creations/blob/master/DDL/namedentityblog/KDDwebscrape.ipynb); can apply this same analysis to web acquired dataset

If you liked this post, make sure to go to the [blog home page](http://districtdatalabs.silvrback.com/) and click the **Subscribe** button so that you don't miss any of our future posts. We're also always looking for blog contributors, so if you have data science skills and want to get some exposure, [apply here](http://www.districtdatalabs.com/#!blog-contributor/c4m8).

### References

<sup id="fn1">1. [(2014). Text Mining and its Business Applications - CodeProject. Retrieved December 26, 2015, from http://www.codeproject.com/Articles/822379/Text-Mining-and-its-Business-Applications.]<a href="#ref1" title="Jump back to footnote 1 in the text.">↩</a></sup>

<sup id="fn2">2. [Suchanek, F., & Weikum, G. (2013). Knowledge harvesting in the big-data era. Proceedings of the 2013 ACM SIGMOD International Conference on Management of Data. ACM.]<a href="#ref2" title="Jump back to footnote 2 in the text.">↩</a></sup>


<sup id ="fn3">3. [Nadeau, D., & Sekine, S. (2007). A survey of named entity recognition and classification. Lingvisticae Investigationes, 30(1), 3-26.]<a href="#ref3" title = "Jump back to footnote 3 in the text">↩</a></sup>

<sup id ="fn4">4. [Ojeda, Tony, Sean Patrick Murphy, Benjamin Bengfort, and Abhijit Dasgupta. [Practical Data Science Cookbook: 89 Hands-on Recipes to Help You Complete Real-world Data Science Projects in R and Python](https://www.packtpub.com/big-data-and-business-intelligence/practical-data-science-cookbook). N.p.: n.p., n.d. Print.]<a href="#ref4" title = "Jump back to footnote 4 in the text">↩</a></sup>


# Appendix

Create all the functions in the appendix before running this code in a notebook.

### Hand labeled entities from two journal articles


```python

p19pdf_authors=['Tim Althoff','Xin Luna Dong','Kevin Murphy','Safa Alai','Van Dang','Wei Zhang']
p19pdf_author_organizations=['Computer Science Department','Stanford University','Google']
p19pdf_author_locations=['Stanford, CA','1600 Amphitheatre Parkway, Mountain View, CA 94043','Mountain View']

p19pdf_references_authors =['A. Ahmed', 'C. H. Teo', 'S. Vishwanathan','A. Smola','J. Allan', 'R. Gupta', 'V. Khandelwal',
                           'D. Graus', 'M.-H. Peetz', 'D. Odijk', 'O. de Rooij', 'M. de Rijke','T. Huet', 'J. Biega', 
                            'F. M. Suchanek','H. Ji', 'T. Cassidy', 'Q. Li','S. Tamang', 'A. Kannan', 'S. Baker', 'K. Ramnath', 
                            'J. Fiss', 'D. Lin', 'L. Vanderwende',  'R. Ansary', 'A. Kapoor', 'Q. Ke', 'M. Uyttendaele',
                           'S. M. Katz','A. Krause','D. Golovin','J. Leskovec', 'A. Krause', 'C. Guestrin', 'C. Faloutsos', 
                            'J. VanBriesen','N. Glance','J. Li','C. Cardie','J. Li','C. Cardie','C.-Y. Lin','H. Lin','J. A. Bilmes'
                           'X. Ling','D. S. Weld', 'A. Mazeika', 'T. Tylenda','G. Weikum','M. Minoux', 'G. L. Nemhauser', 'L. A. Wolsey',
                            'M. L. Fisher','R. Qian','D. Shahaf', 'C. Guestrin','E. Horvitz','T. Althoff', 'X. L. Dong', 'K. Murphy', 'S. Alai',
                            'V. Dang','W. Zhang','R. A. Baeza-Yates', 'B. Ribeiro-Neto', 'D. Shahaf', 'J. Yang', 'C. Suen', 'J. Jacobs', 'H. Wang', 'J. Leskovec',
                           'W. Shen', 'J. Wang', 'J. Han','D. Bamman', 'N. Smith','K. Bollacker', 'C. Evans', 'P. Paritosh', 'T. Sturge', 'J. Taylor',
                           'R. Sipos', 'A. Swaminathan', 'P. Shivaswamy', 'T. Joachims','K. Sprck Jones','G. Calinescu', 'C. Chekuri', 'M. Pl','J. Vondrk',
                           'F. M. Suchanek', 'G. Kasneci','G. Weikum', 'J. Carbonell' ,'J. Goldstein','B. Carterette', 'P. N. Bennett', 'D. M. Chickering',
                            'S. T. Dumais','A. Dasgupta', 'R. Kumar','S. Ravi','Q. X. Do', 'W. Lu', 'D. Roth','X. Dong', 'E. Gabrilovich', 'G. Heitz', 'W. Horn', 
                            'N. Lao', 'K. Murphy',  'T. Strohmann', 'S. Sun','W. Zhang', 'M. Dubinko', 'R. Kumar', 'J. Magnani', 'J. Novak', 'P. Raghavan','A. Tomkins',
                           'U. Feige','F. M. Suchanek','N. Preda','R. Swan','J. Allan', 'T. Tran', 'A. Ceroni', 'M. Georgescu', 'K. D. Naini', 'M. Fisichella',
                           'T. A. Tuan', 'S. Elbassuoni', 'N. Preda','G. Weikum','Y. Wang', 'M. Zhu', 'L. Qu', 'M. Spaniol', 'G. Weikum',
                           'G. Weikum', 'N. Ntarmos', 'M. Spaniol', 'P. Triantallou', 'A. A. Benczr',  'S. Kirkpatrick', 'P. Rigaux','M. Williamson',
                           'X. W. Zhao', 'Y. Guo', 'R. Yan', 'Y. He','X. Li']

p19pdf_allauthors=['Tim Althoff','Xin Luna Dong','Kevin Murphy','Safa Alai','Van Dang','Wei Zhang','A. Ahmed', 'C. H. Teo', 'S. Vishwanathan','A. Smola','J. Allan', 'R. Gupta', 'V. Khandelwal',
                           'D. Graus', 'M.-H. Peetz', 'D. Odijk', 'O. de Rooij', 'M. de Rijke','T. Huet', 'J. Biega', 
                            'F. M. Suchanek','H. Ji', 'T. Cassidy', 'Q. Li','S. Tamang', 'A. Kannan', 'S. Baker', 'K. Ramnath', 
                            'J. Fiss', 'D. Lin', 'L. Vanderwende',  'R. Ansary', 'A. Kapoor', 'Q. Ke', 'M. Uyttendaele',
                           'S. M. Katz','A. Krause','D. Golovin','J. Leskovec', 'A. Krause', 'C. Guestrin', 'C. Faloutsos', 
                            'J. VanBriesen','N. Glance','J. Li','C. Cardie','J. Li','C. Cardie','C.-Y. Lin','H. Lin','J. A. Bilmes'
                           'X. Ling','D. S. Weld', 'A. Mazeika', 'T. Tylenda','G. Weikum','M. Minoux', 'G. L. Nemhauser', 'L. A. Wolsey',
                            'M. L. Fisher','R. Qian','D. Shahaf', 'C. Guestrin','E. Horvitz','T. Althoff', 'X. L. Dong', 'K. Murphy', 'S. Alai',
                            'V. Dang','W. Zhang','R. A. Baeza-Yates', 'B. Ribeiro-Neto', 'D. Shahaf', 'J. Yang', 'C. Suen', 'J. Jacobs', 'H. Wang', 'J. Leskovec',
                           'W. Shen', 'J. Wang', 'J. Han','D. Bamman', 'N. Smith','K. Bollacker', 'C. Evans', 'P. Paritosh', 'T. Sturge', 'J. Taylor',
                           'R. Sipos', 'A. Swaminathan', 'P. Shivaswamy', 'T. Joachims','K. Sprck Jones','G. Calinescu', 'C. Chekuri', 'M. Pl','J. Vondrk',
                           'F. M. Suchanek', 'G. Kasneci','G. Weikum', 'J. Carbonell' ,'J. Goldstein','B. Carterette', 'P. N. Bennett', 'D. M. Chickering',
                            'S. T. Dumais','A. Dasgupta', 'R. Kumar','S. Ravi','Q. X. Do', 'W. Lu', 'D. Roth','X. Dong', 'E. Gabrilovich', 'G. Heitz', 'W. Horn', 
                            'N. Lao', 'K. Murphy',  'T. Strohmann', 'S. Sun','W. Zhang', 'M. Dubinko', 'R. Kumar', 'J. Magnani', 'J. Novak', 'P. Raghavan','A. Tomkins',
                           'U. Feige','F. M. Suchanek','N. Preda','R. Swan','J. Allan', 'T. Tran', 'A. Ceroni', 'M. Georgescu', 'K. D. Naini', 'M. Fisichella',
                           'T. A. Tuan', 'S. Elbassuoni', 'N. Preda','G. Weikum','Y. Wang', 'M. Zhu', 'L. Qu', 'M. Spaniol', 'G. Weikum',
                           'G. Weikum', 'N. Ntarmos', 'M. Spaniol', 'P. Triantallou', 'A. A. Benczr',  'S. Kirkpatrick', 'P. Rigaux','M. Williamson',
                           'X. W. Zhao', 'Y. Guo', 'R. Yan', 'Y. He','X. Li']
```

### Function to pull from top section of document


```python
# attempting function with gold top section...Normal case done

def toppull(docnum=None,section='top',full = False):
    from emailextractor import file_to_str, get_emails # paste code to .py file from following link and save within your environment path to call it: https://gist.github.com/dideler/5219706

    ans={}
    failids = []
    section = section.lower()    
    if docnum is None and full == False:
        raise BaseException("Enter target file to extract data from")

    if docnum is None and full == True:

        text=kddcorpus.raw(docnum).lower()
        # to return output from entire corpus

        if full == True:
            if section == 'top':
                section = ["ABSTRACT","Abstract","Bio","Panel Summary"]
                for fileid in kddcorpus.fileids():
                    text = kddcorpus.raw(fileid)
                    for sect in section:
                        try:
                            part1="(.+)(?="+sect+")"
                            #print "re.compile"+"("+part1+")"
                            p=re.compile(part1)
                            target = p.search(re.sub('[\s]'," ", text)).group()
                            #print docnum,len(target),len(text)

                            emails = tuple(get_emails(target))
                            ans[str(fileid)]={}
                            ans[str(fileid)]["top"]=target.strip()
                            ans[str(fileid)]["charcount"]=len(target)
                            ans[str(fileid)]["emails"]=emails
                            #print [fileid,len(target),len(text)]
                            break
                        except AttributeError:
                            failids.append(fileid)
                            pass
        return ans
        return failids

        # to return output from one document
    else:
        ans = {}
        failids=[]
        text = kddcorpus.raw(docnum)

        if section == "top":
            section = ["ABSTRACT","Abstract","Bio","Panel Summary"]
            text = kddcorpus.raw(docnum)
            for sect in section:
                try:
                    part1="(.+)(?="+sect+")"
                    #print "re.compile"+"("+part1+")"
                    p=re.compile(part1)
                    target = p.search(re.sub('[\s]'," ", text)).group()
                    #print docnum,len(target),len(text)

                    emails = tuple(get_emails(target))
                    ans[str(docnum)]={}
                    ans[str(docnum)]["top"]=target.strip()
                    ans[str(docnum)]["charcount"]=len(target)
                    ans[str(docnum)]["emails"]=emails
                    #print [fileid,len(target),len(text)]
                    break

                except AttributeError:
                    failids.append(fileid)
                    pass

        return ans
        return failids
```

### Function to build list of named entity classes from Standard NLTK Chunker


```python
def nltktreelist(text):
    from operator import itemgetter

    text = text


    persons = []
    organizations = []
    locations =[]
    genpurp = []

    for l in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text))):
        if isinstance(l,nltk.tree.Tree):
            if l.label() == 'PERSON':
                if len(l)== 1:
                    if l[0][0] in persons:
                        pass
                    else:
                        persons.append(l[0][0])
                else:
                    if " ".join(map(itemgetter(0), l)) in persons:
                        pass
                    else:
                        persons.append(" ".join(map(itemgetter(0), l)).strip("*"))


    for o in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text))):
        if isinstance(o,nltk.tree.Tree):
            if o.label() == 'ORGANIZATION':
                if len(o)== 1:
                    if o[0][0] in organizations:
                        pass
                    else:
                        organizations.append(o[0][0])
                else:
                    if " ".join(map(itemgetter(0), o)) in organizations:
                        pass
                    else:
                        organizations.append(" ".join(map(itemgetter(0), o)).strip("*"))


    for o in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text))):
        if isinstance(o,nltk.tree.Tree):
            if o.label() == 'LOCATION':
                if len(o)== 1:
                    if o[0][0] in locations:
                        pass
                    else:
                        locations.append(o[0][0])
                else:
                    if " ".join(map(itemgetter(0), o)) in locations:
                        pass
                    else:
                        locations.append(" ".join(map(itemgetter(0), o)).strip("*"))

    for e in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text))):
        if isinstance(o,nltk.tree.Tree):
            if o.label() == 'GPE':
                if len(o)== 1:
                    if o[0][0] in genpurp:
                        pass
                    else:
                        genpurp.append(o[0][0])
                else:
                    if " ".join(map(itemgetter(0), o)) in genpurp:
                        pass
                    else:
                        genpurp.append(" ".join(map(itemgetter(0), o)).strip("*"))




    results = {}
    results['persons']=persons
    results['organizations']=organizations
    results['locations']=locations
    results['genpurp'] = genpurp

    return results
```

### Function to get lists of entities from Stanford NER


```python
def get_continuous_chunks(string):
    string = string
    continuous_chunk = []
    current_chunk = []

    for token, tag in stner.tag(string.split()):
        if tag != "O":
            current_chunk.append((token, tag))
        else:
            if current_chunk: # if the current chunk is not empty
                continuous_chunk.append(current_chunk)
                current_chunk = []
    # Flush the final current_chunk into the continuous_chunk, if any.
    if current_chunk:
        continuous_chunk.append(current_chunk)
    named_entities = continuous_chunk
    named_entities_str = [" ".join([token for token, tag in ne]) for ne in named_entities]
    named_entities_str_tag = [(" ".join([token for token, tag in ne]), ne[0][1]) for ne in named_entities]
    persons = []
    for l in [l.split(",") for l,m in named_entities_str_tag if m == "PERSON"]:
        for m in l:
            for n in m.strip().split(","):
                if len(n)>0:
                    persons.append(n.strip("*"))
    organizations = []
    for l in [l.split(",") for l,m in named_entities_str_tag if m == "ORGANIZATION"]:
        for m in l:
            for n in m.strip().split(","):
                n.strip("*")
                if len(n)>0:
                    organizations.append(n.strip("*"))
    locations = []
    for l in [l.split(",") for l,m in named_entities_str_tag if m == "LOCATION"]:
        for m in l:
            for n in m.strip().split(","):
                if len(n)>0:
                    locations.append(n.strip("*"))
    dates = []
    for l in [l.split(",") for l,m in named_entities_str_tag if m == "DATE"]:
        for m in l:
            for n in m.strip().split(","):
                if len(n)>0:
                    dates.append(n.strip("*"))
    money = []
    for l in [l.split(",") for l,m in named_entities_str_tag if m == "MONEY"]:
        for m in l:
            for n in m.strip().split(","):
                if len(n)>0:
                    money.append(n.strip("*"))
    time = []
    for l in [l.split(",") for l,m in named_entities_str_tag if m == "TIME"]:
        for m in l:
            for n in m.strip().split(","):
                if len(n)>0:
                    money.append(n.strip("*"))

    percent = []
    for l in [l.split(",") for l,m in named_entities_str_tag if m == "PERCENT"]:
        for m in l:
            for n in m.strip().split(","):
                if len(n)>0:
                    money.append(n.strip("*"))

    entities={}
    entities['persons']= persons
    entities['organizations']= organizations
    entities['locations']= locations
    #entities['dates']= dates
    #entities['money']= money
    #entities['time']= time
    #entities['percent']= percent

    return entities
```

### Function to pull Keywords section only


```python


# attempting function with gold keywords....

def keypull(docnum=None,section='keywords',full = False):

    ans={}
    failids = []
    section = section.lower()    
    if docnum is None and full == False:
        raise BaseException("Enter target file to extract data from")

    if docnum is None and full == True:

        text=kddcorpus.raw(docnum).lower()



        # to return output from entire corpus
        if full == True:
            for fileid in kddcorpus.fileids():
                text = kddcorpus.raw(fileid).lower()
                if section == "keywords":
                    section1="keywords"
                    target = ""   
                    section2=["1.  introduction  ","1.  introd ","1. motivation","(1. tutorial )"," permission to make ","  permission to make","(  permission to make digital )","    bio  ","abstract:  ","1.motivation" ]

                    part1= "(?<="+str(section1)+")(.+)"
                    for sect in section2:
                        try:
                            part2 = "(?="+str(sect)+")"
                            p=re.compile(part1+part2)
                            target=p.search(re.sub('[\s]'," ",text)).group(1)
                            if len(target) >50:
                                if len(target) > 300:
                                    target = target[:200]
                                else:
                                    target = target

                                ans[str(fileid)]={}
                                ans[str(fileid)]["keywords"]=target.strip()
                                ans[str(fileid)]["charcount"]=len(target)
                                #print [fileid,len(target),len(text)]
                                break
                            else:
                                if len(target)==0:
                                     failids.append(fileid)   
                                pass
                        except AttributeError:
                            failids.append(fileid)
                            pass
            set(failids)
            return ans
        # to return output from one document
    else:
        ans = {}
        text=kddcorpus.raw(docnum).lower()
        if full == False:
            if section == "keywords":
                section1="keywords"
                target = ""   
                section2=["1.  introduction  ","1.  introd ","1. motivation","permission to make ","1.motivation" ]

                part1= "(?<="+str(section1)+")(.+)"

                for sect in section2:
                    try:
                        part2 = "(?="+str(sect)+")"
                        p=re.compile(part1+part2)
                        target=p.search(re.sub('[\s]'," ",text)).group(1)
                        if len(target) >50:
                            if len(target) > 300:
                                target = target[:200]
                            else:
                                target = target
                            ans[docnum]={}
                            ans[docnum]["keywords"]=target.strip()
                            ans[docnum]["charcount"]=len(target)
                            break                  
                    except:
                        pass
    return ans
    return failids

```

### Function to pull Abstract only


```python
# attempting function with gold abstracts...Normal case done

def abpull(docnum=None,section='abstract',full = False):

    ans={}
    failids = []
    section = section.lower()    
    if docnum is None and full == False:
        raise BaseException("Enter target file to extract data from")

    if docnum is None and full == True:

        text=kddcorpus.raw(docnum).lower()
        # to return output from entire corpus
        if full == True:
            for fileid in kddcorpus.fileids():
                text = kddcorpus.raw(fileid)
                if section == "abstract":
                    section1=["ABSTRACT", "Abstract "]
                    target = ""   
                    section2=["Categories and Subject Descriptors","Categories & Subject Descriptors","Keywords","INTRODUCTION"]
                    for fileid in kddcorpus.fileids():
                        text = kddcorpus.raw(fileid)


                        for sect1 in section1:
                            for sect2 in section2:
                                part1= "(?<="+str(sect1)+")(.+)"
                                part2 = "(?="+str(sect2)+")"
                                p = re.compile(part1+part2)
                                try:
                                    target=p.search(re.sub('[\s]'," ",text)).group()
                                    if len(target) > 50:
                                        ans[str(fileid)]={}
                                        ans[str(fileid)]["abstract"]=target.strip()
                                        ans[str(fileid)]["charcount"]=len(target)
                                        #print [fileid,len(target),len(text)]
                                        break
                                    else:
                                        failids.append(fileid)
                                        pass
                                except AttributeError:
                                    pass 

            return ans

        # to return output from one document
    else:
        ans = {}
        failids=[]
        text = kddcorpus.raw(docnum).lower()
        if section == "abstract":
            section1=["ABSTRACT", "Abstract "]
            target = ""   
            section2=["Categories and Subject Descriptors","Categories & Subject Descriptors","Keywords","INTRODUCTION"]
            for sect1 in section1:
                for sect2 in section2:
                    part1= "(?<="+str(sect1)+")(.+?)"
                    part2 = "(?="+str(sect2)+"[\s]?)"
                    p = re.compile(part1+part2)
                    try:
                        target=p.search(re.sub('[\s]'," ",text)).group()
                        if len(target) > 50:
                            ans[str(docnum)]={}
                            ans[str(docnum)]["abstract"]=target.strip()
                            ans[str(docnum)]["charcount"]=len(target)
                            #print [docnum,len(target),len(text)]
                            break
                        else:
                            failids.append(docnum)
                            pass
                    except AttributeError:
                        pass
        return ans
        return failids
```
