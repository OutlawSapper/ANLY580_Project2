# -*- coding: utf-8 -*-
"""
Authorship Detection
"""

#load necessary libraries and set global conditions
# import os
import pandas as pd
import numpy as np
# import json
from datetime import datetime
import pytz
import re
import nltk
import spacy
import matplotlib.pyplot as plt
from collections import Counter 
# import gensim
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD, PCA
# from sklearn.feature_extraction import stop_words
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# from PIL import Image
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import GridSearchCV
# import pyLDAvis
# import pyLDAvis.sklearn
# import matplotlib.pyplot as plt
# import chart_studio
# import chart_studio.plotly as py
# import plotly.graph_objects as go
# import pickle

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

np.set_printoptions(suppress=True)

def dataImport(file):
    '''
    Imports the data and checks the import quality by printing the string
    Inputs: file (str)
    Outputs: DF (Pandas DF)
    '''
    #import the dataset
    DF = pd.read_csv(file)
    
    #check the quality of the import
    print(DF.head())
    
    return DF

def summaryStats(DF):
    '''
    Generate Summmary Statitics for the full feature space
    Input: DF (pd DF)
    Output: None
    '''
    
    print('\n###########', 'Summary Statistics', '###########\n')
    print('Dataframe Structure:', DF.shape)

    print('\nVariable Types:')
    print(DF.dtypes)
    
    print('\nSummary Stats (Quantitative Features)')
    print(DF.describe())
    
    print('\nSummary Stats (Qualitative Features)')
    print(DF.describe(include=['object']))
    
def reformatData(DF):
    '''
    Restructure the data as required (change column names, dtypes, etc) 
    Input: DF (pd DF)
    Output: textDF (pd DF)
    '''
    #rename the columns
    DF.columns = ['Datetime', 'TweetID','FullText', 'Truncated', 'RepliedTo',
                  'GeoTagged', 'Coordinates', 'Location', 'RetweetCount', 
                  'FavoriteCount', 'Sensitive?', 'Lang', 'UserName', 'UserLocation',
                  'UserDescription', 'UserFollowers', 'UserFriends', 'UserStatusCount']
    
    #convert the columns to the appropriate data types
    DF['Datetime']= pd.to_datetime(DF['Datetime']) 
    
    #convert remaining columns to the appropriate type
    DF = DF.astype({'Sensitive?': bool, 'TweetID': str})
    
    #only focused on the text data so only retain columns of interest for
    #textual analysis
    textDF = DF[['Datetime', 'TweetID', 'FullText', 'UserName', 'Lang']]
        
    return textDF

def scoreData(DF):
    '''
    Score the dataframe to ensure that each feature is in the appropriate value range
    and/or has no NA
    Input: DF (pd DF)
    Output: None
    '''
    ##check datetime range
    bad_date = 0
    
    #check date range (June 1, 2019 to Super Tuesday 2020)
    for date in DF['Datetime']:
        if date < datetime(2019, 6, 1, 0, 0, 0, 0, pytz.UTC) or \
            date > datetime(2020, 3, 3, 0, 0, 0, 0, pytz.UTC):
            bad_date += 1
    
    #check for null values
    bad_date += sum(DF['Datetime'].isnull())
    
    #TweetID should be unique
    bad_id = len(DF['TweetID'])-len(DF['TweetID'].drop_duplicates())
    bad_id += sum(DF['TweetID'].isnull())
    
    #FullText cannot be empty
    bad_text = sum(DF['FullText'].isnull()) 
    
    #UserName
    names = ['Amy Klobuchar', 'Bernie Sanders', 'Elizabeth Warren', 'Joe Biden',
             'Pete Buttigieg', 'Tom Steyer', 'Tulsi Gabbard 🌺']
    
    bad_names = 0
    
    for name in DF['UserName']:
        if name not in names:
            bad_names += 1
     
    ## check if the language is not english            
    bad_lang = 0
    
    for language in DF['Lang']:
        if language != 'en':
            bad_lang += 1
    
    ## generate the score
    print('\n######## Data Frame Score by Column #########')   
    print("Date Column: ", bad_date)
    print("TweetID Column: ", bad_id)
    print("Text Column: ", bad_text)
    print("UserNames Column:", bad_names)
    print("Language Column (not English):", bad_lang)
    
    bad_total = bad_date + bad_id + bad_text + bad_names + bad_lang
    score = bad_total / (DF.shape[0]*DF.shape[1])
    
    print('\n######## Total DF Score #########')   
    print('Total DF Cleanliness Score:', 100-round(score * 100, 2))
    
def cleanData(DF):
    '''
    Clean the dataset by dropping duplicate tweets and removing those that 
    are not in english
    Input: DF (pd DF)
    Output: DF (pd DF)
    '''
    #drop columns with duplicate tweet ID
    DF = DF.drop_duplicates('TweetID', keep = 'last')

    #drop tweets that are not in English    
    DF = DF[DF['Lang'] == 'en']
    
    return DF

def keepChars(tweet):
    '''
    Cleans an invididual tweet by removing non-character features, any url link, 
    etc. Designed to be used with a map function.
    Input: tweet (list of tokens)
    Output: charTweet (list of tokens)
    '''
    ## initialize a list to store the characters only
    charTweet = []
    
    ## do I want to filter NER before I begin?????
    
    for token in tweet:
        ## remove token if it contains a url link
        if 'http' in token.text:
            continue
        
        ## remove a token if it contains a mention, hastag, punctuation, etc.
        elif re.match(r'[^0-9a-zA-Z_]', token.text):
            continue
        
        ## remove token if it is less than one character long
        elif len(token.text) < 2:
            continue
        
        ## append the new token to the character only token list
        else:
            charTweet.append(token)
        
    return charTweet

def mendenhallAnalysis(nlp, DF):
    '''
    Generate Mendenhall's Curve to understand the stylometric differences
    between the candidate's tweeting styles
    Input: nlp (spaCy) and DF (pandas)
    Output: None
    '''
    ## initialize a dictionary to store the results
    totalDict = {}
    
    ## for each candidate count the number of times that words of each
    ## word length are used
    for name in DF['UserName'].unique():
        text = DF[DF['UserName'] == name]['TextOnly']
        
        ## the candidate's word length dictionary
        candDict = {}
        
        ## for each tweet calculate the length of each word
        for tweet in text:
            for token in tweet:
                tokenLen = len(token)
                
                ## if that word length exists in the dictionary add one
                ## otherwise add a new value
                if tokenLen in candDict.keys():
                    candDict[tokenLen] += 1
                else:
                    candDict[tokenLen] = 0
        
        ## sum the candidate's total tokens (to generate relative value)
        sumTokens = sum(candDict.values())            
        
        ## generate the relative useage rate and sort by key (word length)
        totalDict[name] = {k: v/sumTokens for k, v in sorted(candDict.items(), key=lambda item: item[0], reverse = False)}
        
    ## generate mendenhall's curve
    fig, ax = plt.subplots()

    for cand in totalDict.keys():
        ## if the candidate's word length range was relatively unique
        ## assign a random color and associate it with their name
        ## otherwise color their curve grey and give the generic 'Other Candidate Label'
        if totalDict[cand][3] > .22:
            ax.plot(totalDict[cand].keys(), totalDict[cand].values(), label = cand)
        elif totalDict[cand][4] < .17:
            ax.plot(totalDict[cand].keys(), totalDict[cand].values(), label = cand)
        else:
            try:
                i
            except:
                i = 0
            ax.plot(totalDict[cand].keys(), totalDict[cand].values(), color = '#D3D3D3',
                    alpha = .6, label = 'Other Candidates' if i == 0 else '')
            i += 1
    
    ## clean the plot and its formatting
    ax.set_xlim(2, 15)    
    ax.set_ylim(0, .24)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Word Length')
    ax.set_ylabel('Percent of Total Words')
    ax.set_yticklabels(["{:.0%}".format(i) for i in ax.get_yticks()])
    ax.set_title("Candidate Word Length Tendancies: Mendenhall's Characteristic Curve", 
                 loc = 'left', fontweight = 'semibold', fontsize = 14)
    ax.legend(title = '$\\bf{Candidate}$', edgecolor = 'none')      

def wordCounter(text):
    return Counter(text)
        
    
def kilgariffMethod(nlp, DF):
    
    pass

def main():
    ## Load the Data
    DF = dataImport('Data/DemTweetsReformat.csv')
    
    ## generate basic summary statistics
    summaryStats(DF)
    
    ## get the number of tweets by user
    print(DF['user.name'].value_counts())
    
    ## reformat the dataframe--removing unneeded column
    ## modifying data types, etc.
    DF = reformatData(DF)
    
    ## get a preliminary score for the dataset cleanliness
    scoreData(DF)
    
    ## clean the data
    DF = cleanData(DF)
    
    ## rescore the data
    scoreData(DF)
        
    ## load spacy
    nlp = spacy.load("en_core_web_sm", parser = False)
    
    ## load the current list of prefixes & remove # from the list 
    ## (keep hastags in tact)
    prefixes = list(nlp.Defaults.prefixes)
    prefixes.remove('#')
    
    ## recompile a new prefix regex with the smaller list of prefixes
    prefix_regex = spacy.util.compile_prefix_regex(prefixes)

    ## set the tokenizer prefix_search to use the search of the newly compiled regex
    nlp.tokenizer.prefix_search = prefix_regex.search
    
    ## tokenize every tweet in the dataframe & adding a new column in the process
    DF['Tokens'] = DF['FullText'].apply(lambda x: nlp.tokenizer(x))
    
    DF['TextOnly'] = DF['Tokens'].map(keepChars)
    
    # mendenhallAnalysis(nlp, DF)
    
    # kilgariffMethod(nlp, DF)
        
    # something = DF['Tokens'].map(wordCounter)
    
    # blank = Counter()
    
    # for i in something:
    #     blank += i
        
    # print(blank)
        
    ## convert each candidate's tweets into a single entity (dictionary) ''.join()
    
    ## Mendenhall's Characteristic Curves of Composition
    ## https://programminghistorian.org/en/lessons/introduction-to-stylometry-with-python
    
        ##tweet tokenize-->use spacy w/ affix append for hashtags
        
        ##get token length & count
    
    
    
    
    
    
    
    

if __name__ == '__main__':
    main()
















