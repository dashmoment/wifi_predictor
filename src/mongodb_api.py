from pymongo import MongoClient
import pymongo

class mongodb_api:
    
    def __init__(self, ip = '10.144.25.112', user=None, pwd=None, database='test', collection='testinfo'):
        
        

        uri = "mongodb://"+user+":"+ pwd + "@" + ip
        try:
            self.client = MongoClient(uri,serverSelectionTimeoutMS=1)
            self.client.server_info()
        except pymongo.errors.ServerSelectionTimeoutError as err: 
            print(err)
            
        self.database = self.client[database]
        self.collection = self.database[collection]
        
    def insert(self, data):
        
        collection = self.collection
            
        inserted_id = collection.insert_one(data).inserted_id
        
        print("Get data with id ", inserted_id)
        
        
        
    def find(self,  key_value = None, visible = None,ftype = 'one'):
        
        
        collection = self.collection   
        found_data = []
        
        if ftype == 'one':    
            found_data = collection.find_one(key_value,visible)
            return found_data
        elif ftype == 'many':
            for d in collection.find({}, {'_id':0}):
                found_data.append(d)
            return found_data    
        else:
            print("Please dfine type: 'one' or 'many'")
            return 
       
    def remove(self,  key_value = None, justone = True):
        
        collection = self.collection
        
        if justone == True:
            collection.delete_one(key_value)
        else:
            collection.delete_many(key_value)
        
        
    
def api_demo():
    m = mongodb_api(user='ubuntu', pwd='ubuntu')
    post1 = {"author": "Mike2",
             "text": "My first blog post!",
             "tags": ["mongodb", "python", "pymongo"]}
    post1 = {"author": "Mike3",
             "text": "My first blog post2!",
             "tags": ["mongodb", "python", "pymongo"]}
    
    m.insert(post1)
    #found_data = m.find(ftype='many')
    #print("Found_all:", found_data)
    find_reg = {"author": "*"}
#    m.remove(find_reg, True)
    #found_data = m.find(ftype='many')
    for i in range(10):
        one = m.find(key_value = find_reg)
        print("Found one: ",one)
    
m = mongodb_api(user='ubuntu', pwd='ubuntu', collection="TestData1061121")
one = m.find(key_value = {})
print(one)
#api_demo()