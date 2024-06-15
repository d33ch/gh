from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['test']
collection = db['gh']
documents = collection.find()
count = collection.count_documents({})

print(count)
for document in documents: print(f'{document.get("_id")} {document.get("key")}')