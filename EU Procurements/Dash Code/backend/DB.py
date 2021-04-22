from pymongo import MongoClient

host="rhea.isegi.unl.pt"
port="28005"
user="GROUP_5"
password="MzA0MTk3MTEwNTM2Mzg3NTY4MzMxODIzODUzNzQzNzcwOTAwNjkz"
protocol="mongodb"

client = MongoClient(f"{protocol}://{user}:{password}@{host}:{port}")
db = client.contracts
eu = db.eu
