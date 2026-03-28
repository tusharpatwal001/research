import requests
from termcolor import colored

# your serper api key
api_key = "a6bb099fbbed21eb01c587969b1770ed87cc9f20"

# define the query and API URL
query = "Iran War"
url = "https://google.serper.dev/search"

# Header with API Key
headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

# request Parameters
params = {"q": query, "num": 5, "tbs": "qdr:d"}  # limiting top 5 results

# send the GET request to serper API
response = requests.get(url, headers=headers, params=params)

# Check if the request was successfull
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # extract and print relevant information
    print(data)

# whole output
{
    "searchParameters": {
        "q": "Iran War",
        "type": "search",
        "num": 5,
        "engine": "google",
    },
    "organic": [
        {
            "title": "2026 Iran war - Wikipedia",
            "link": "https://en.wikipedia.org/wiki/2026_Iran_war",
            "snippet": "On 28 February 2026, a war began when the United States and Israel launched surprise airstrikes on multiple sites and cities across Iran, killing Supreme Leader ...",
            "date": "3 hours ago",
            "position": 1,
        },
        {
            "title": "Latest Analysis: War with Iran | CSIS",
            "link": "https://www.csis.org/programs/latest-analysis-war-iran",
            "snippet": "The war with Iran has had significant consequences for food security, economic security, and humanitarian crises in the Middle East and ...",
            "position": 2,
        },
        {
            "title": "Iran War: Latest Breaking News, Updates & Analysis | Reuters",
            "link": "https://www.reuters.com/world/iran/",
            "snippet": "Real-time Reuters coverage of the Iran war: US-Israel strikes, Iranian retaliation, nuclear threats, oil market shocks, and regional war risks.",
            "date": "3 hours ago",
            "position": 3,
        },
        {
            "title": "The War in Iran: Operational Progress, but Challenges Remain",
            "link": "https://understandingwar.org/research/middle-east/the-war-in-iran-operational-progress-but-challenges-remain/",
            "snippet": "The United States is steadily destroying Iran's ability to use its most essential tool in the war: drone and missile strikes.",
            "date": "Mar 15, 2026",
            "position": 4,
        },
        {
            "title": "Houthis Announce Entry Into Iran War, Launching Strikes On Israel",
            "link": "https://www.youtube.com/watch?v=8HlwUZQtXEY",
            "snippet": "On Saturday morning, the Houthis of Yemen launched strikes on Israel as they enter into the Iran war. Stay Connected Forbes Breaking News on ...",
            "date": "2 hours ago",
            "position": 5,
        },
    ],
    "credits": 1,
}

# 24 hours constraint
{
    "searchParameters": {
        "q": "Iran War",
        "type": "search",
        "num": 5,
        "tbs": "qdr:d",
        "engine": "google",
    },
    "organic": [
        {
            "title": "Iran war live: US-Israeli war on Iran widens with first attack from Yemen",
            "link": "https://www.aljazeera.com/news/liveblog/2026/3/28/iran-war-live-trump-again-slams-natos-lack-of-support-for-war-on-tehran",
            "snippet": "Yemen's Iran-backed Houthi rebels have confirmed their first attack on Israel since the United States-Israeli war on Iran began.",
            "date": "8 hours ago",
            "position": 1,
        },
        {
            "title": "No end to war in sight after one month as Iran squeezes global economy",
            "link": "https://www.nbcnews.com/world/iran/one-month-iran-squeezes-global-economy-rcna265279",
            "snippet": "“The U.S. and Israel are fighting a war aimed at weakening Iran, while Iran is fighting a war to survive. ... war with Iran began to more than 50%. Since ...",
            "date": "8 hours ago",
            "position": 2,
        },
        {
            "title": "Inside Iran's military: missiles, militias and a force built for survival",
            "link": "https://www.foxnews.com/world/inside-irans-military-missiles-militias-force-built-survival",
            "snippet": "Iran's military is designed to survive a war, not win one, experts say. After weeks of U.S. and Israeli strikes, analysts say the force remains capable of ...",
            "date": "6 hours ago",
            "position": 3,
        },
        {
            "title": "Iran War Live Updates: Houthis Enter War With Missile Attack on Israel",
            "link": "https://www.nytimes.com/live/2026/03/28/world/iran-war-trump-israel-oil",
            "snippet": "The Houthis, an Iranian-backed militant group in Yemen, announced on Saturday that they had launched a ballistic missile attack on Israel, appearing to open ...",
            "date": "7 minutes ago",
            "position": 4,
        },
        {
            "title": "Iran war divides older and younger Trump voters at CPAC",
            "link": "https://www.bbc.com/news/articles/cjd8e4px12ro",
            "snippet": "A majority of the American public, polls suggest, have been against the ongoing US-Israeli military campaign in Iran from the day it started. Republicans, ...",
            "date": "16 hours ago",
            "position": 5,
        },
    ],
    "credits": 1,
}
