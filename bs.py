import urllib
import re
import math
from bs4 import BeautifulSoup

##  This is my small project on Sentimental Analysis
##  Please NOTE: You cannot use this code without prior permission

poststext=[]
htmlfile = urllib.urlopen("http://manraj.collegespace.in/sentimentanalysisPY/")
htmltext = htmlfile.read()
htmlfile.close()
soup = BeautifulSoup(htmltext)
post = soup.find_all("post")
for text in post:
    poststext.append(''.join(text.find_all(text=True)))
i=0
patternsplit = re.compile(r"\W+")
list=["love","hate"]
while i<len(poststext):
    words = patternsplit.split(poststext[i].lower())
    sentilist = map(lambda word: word in list,words)
    t = sentilist.count(True)
    f = sentilist.count(False)
    factor = t*100/math.sqrt(len(sentilist)) 
    print str(factor) + ' %'
    i+=1
