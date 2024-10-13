import pymongo
import json

YEAR = "2016"
STATES = ["NSW", "VIC", "QLD", "NZ", "NT", "SA", "TAS", "WA"]
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "gh"
OUTPUT_FILE = f"history_unique_runners_{YEAR}.json"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]


pipeline = [
    {"$unwind": "$runs"},
    {"$group": {"_id": "$runs.dogName"}},
    {"$sort": {"_id": 1}},
    {"$project": {"_id": 0, "name": "$_id"}},
]
names = []

for state in STATES:
    COLLECTION_NAME = f"history_{state.lower()}_{YEAR}"
    collection = db[COLLECTION_NAME]
    result = list(collection.aggregate(pipeline))
    names.extend([doc["name"].replace("'", "") for doc in result])

with open(OUTPUT_FILE, "w") as file:
    json.dump(names, file, indent=2)

print(f"Results have been written to {OUTPUT_FILE}")

client.close()
