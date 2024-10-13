import pymongo
import json

YEAR = "2016"
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "gh"
COLLECTION_NAME = f"markets_{YEAR}"
OUTPUT_FILE = f"market_unique_runners_{YEAR}.json"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

pipeline = [
    {"$unwind": "$runners"},
    {
        "$project": {
            "name": {
                "$trim": {
                    "input": {
                        "$substr": [
                            "$runners.name",
                            {"$add": [{"$indexOfCP": ["$runners.name", "."]}, 1]},
                            {"$strLenCP": "$runners.name"},
                        ]
                    }
                }
            }
        }
    },
    {"$group": {"_id": "$name"}},
    {"$sort": {"_id": 1}},
    {"$project": {"_id": 0, "name": "$_id"}},
]

result = list(collection.aggregate(pipeline))
names = [doc["name"].upper() for doc in result]

with open(OUTPUT_FILE, "w") as file:
    json.dump(names, file, indent=2)

print(f"Results have been written to {OUTPUT_FILE}")

# Close the MongoDB connection
client.close()
