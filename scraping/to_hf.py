import datasets
import json

TOKEN = ""


# Load in the data (list of json) from tweets.txt
with open('scraping/tweets.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()
for i in range(len(data)):
    data[i] = json.loads(data[i])
    
# Create a dataset from the data
dataset = datasets.Dataset.from_list(data)

# Save the dataset
dataset.push_to_hub('gmongaras/Yann_LeCun_Tweets', token=TOKEN)