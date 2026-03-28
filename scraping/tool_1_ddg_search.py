import requests
from termcolor import colored

# define the query and API URL
url = f"https://api.duckduckgo.com"

# Send the Get Request to DuckduckGo's API
response = requests.get(url, params={
    "q": "world War",
    "format":"json"
})

# Parse the json response
data = response.json()

# Extract and print relevant information
related_topics = data.get("RelatedTopics", [])
for topic in related_topics[:5]:  # Slice to get only the first 3 topics
    if "Text" in topic and "FirstURL" in topic:
        print(colored(f"Title: {topic['Text']}", "cyan"))
        print(colored(f"URL {topic['FirstURL']}", "yellow"))
        print()
