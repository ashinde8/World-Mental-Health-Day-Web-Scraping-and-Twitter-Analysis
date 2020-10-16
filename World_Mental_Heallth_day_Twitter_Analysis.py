# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 01:58:37 2020

@author: Lenovo
"""


from selenium import webdriver
import time
import csv
import pandas as pd
import os

"""
A function that uses selenium and chromedriver to get tweets from a Twitter account.
url is the link the account
scrollNum is the number of times we want to scroll to load more tweets.
"""

def getTweets(url):
    #open the browser and visit the url
    
    folder_path = 'D:\\Downloads\\BIA 660\\chromedriver_win32'
    driver = webdriver.Chrome('chromedriver.exe')

    driver.get(url)
    time.sleep(2)

    already_seen=set()
    #keeps track of tweets we have already seen.

    #write the tweets to a f1ile
    file_name = 'mentalhealth.txt'
    fw=open(file_name,'w',encoding='utf8')
 
    writer=csv.writer(fw,lineterminator='\n')#create a csv writer for this file
    
    scrollNum = 25
    
    for i in range(scrollNum):

        print('batch count',i)
        
        #find all elements that have the value "tweet" for the data-testid attribute
        tweets=driver.find_elements_by_css_selector('div[data-testid="tweet"]')#
        print(len(tweets),' tweets found\n')
        
        for tweet in tweets:

            if tweet in already_seen:continue #we have seen this tweet before while scrolling down, ignore
            already_seen.add(tweet) #first time we see this tweet. Mark as seen and process.
        
            txt, comments, likes, retweets ='NA','NA', 'NA', 'NA'
        
            try: 
                txt=tweet.find_element_by_css_selector("div.css-901oao.r-hkyrab.r-1qd0xha.r-a023e6.r-16dba41.r-ad9z0x.r-bcqeeo.r-bnwqim.r-qvutc0").text
                txt=txt.replace('\n', ' ')
            except: 
                print ('no text')     
                
                
            try:
        
                #find the div element that havs the value "retweet" for the data-testid attribute
                replyElement=tweet.find_element_by_css_selector('div[data-testid="reply"]')
 
                #find the span element that has all the specified values (space separated) in its class attribute
                comments=replyElement.find_element_by_css_selector('span.css-901oao.css-16my406.r-1qd0xha.r-ad9z0x.r-bcqeeo.r-qvutc0').text  
                                  
            except:
                print ('no comments')
                
            try:
        
                #find the div element that havs the value "retweet" for the data-testid attribute
                likesElement=tweet.find_element_by_css_selector('div[data-testid="like"]')
 
                #find the span element that has all the specified values (space separated) in its class attribute
                likes=likesElement.find_element_by_css_selector('span.css-901oao.css-16my406.r-1qd0xha.r-ad9z0x.r-bcqeeo.r-qvutc0').text  
                                  
            except:
                print ('no likes')
                
            try:
        
                #find the div element that havs the value "retweet" for the data-testid attribute
                retweetElement=tweet.find_element_by_css_selector('div[data-testid="retweet"]')
 
                #find the span element that has all the specified values (space separated) in its class attribute
                retweets=retweetElement.find_element_by_css_selector('span.css-901oao.css-16my406.r-1qd0xha.r-ad9z0x.r-bcqeeo.r-qvutc0').text  
                                  
            except:
                print ('no retweets')

            
            #only write tweets that have text or retweets (or both). 
            if txt!='NA' or comments!= 'NA' or likes!= 'NA' or retweets!='NA':
                writer.writerow([txt,comments, likes, retweets])


# For finding out the total number of retweets, likes, comments or for comparing these parameters with another twitter profile
# we will first have to check for k if k is there then remove k and then convert the string into a float data type and then multiply it by 1000




        #scroll down twice to load more tweets
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        time.sleep(2)

    fw.close()
    var = 'world_mental_health_day'
        
    new_path = os.path.join(folder_path, var)
    new_path = new_path + '.csv'
    
    tweets_file = pd.read_csv(file_name, names = ['TWEET','COMMENTS','LIKES','RETWEETS'], sep = ',')
    tweets_file.to_csv(new_path, index = False, header=True)
    print('done')


url = 'https://twitter.com/hashtag/worldmentalheathday?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Ehashtag'

getTweets(url)