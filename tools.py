import requests
from bs4 import BeautifulSoup
from readability import Document
import re

"""Use BeautifulSoup to extract news content"""
def get_text_by_bs4(url):
    try:
        # 1. Get web page content, set timeout to 10 seconds
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.encoding = response.apparent_encoding  # Set correct encoding

        doc = Document(response.text)
        clean_html = doc.summary()
        title = doc.title()

        # Use BeautifulSoup to extract the body plain text
        soup = BeautifulSoup(clean_html, 'html.parser')
        content = soup.get_text(strip=True)
        return content
    except Exception as e:
        print(f"Failed to extract: {e}")
        return None

def get_text_from_url(url):
    return get_text_by_bs4(url)


if __name__ == "__main__":
    url = "https://www.bbc.com/sport/formula1/articles/ce80epy1e2lo"
    url = "https://www.gulf-times.com/article/705302/qatar/doha-meet-calls-for-global-protocol-to-ensure-safe-ai-usage"
    print(get_text_from_url(url))
