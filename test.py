from readability import Document
import requests
from bs4 import BeautifulSoup

# 1. Get web page content
url = 'https://abcnews.go.com/Business/wireStory/zelenskyy-visits-berlin-seeks-support-ukraine-war-russia-122257223'
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
response.encoding = response.apparent_encoding

doc = Document(response.text)
clean_html = doc.summary()
title = doc.title()

# Extract the body plain text using BeautifulSoup
soup = BeautifulSoup(clean_html, 'html.parser')
text = soup.get_text(strip=True)

print("Title:", title)
print("Text:", text)
