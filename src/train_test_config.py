BigRun = { "traffic":["-Two5M","-Two10M","-Two15M","-Two20M"], 
            "ID":["-1", "-2","-3","-4","-5"]}    
BigBigRun = { "traffic":["-Two5M","-Two10M","-Two15M","-Two20M","-Two25M","-Two30M"], 
            "ID":["-1", "-2","-3","-4","-5"]} 
NoneTraf = { "traffic":["-None"], "ID":["-1", "-2","-3","-4","-5","-6","-7","-8","-9","-10"]}    
NoneTraf_big = { "traffic":["-None"], "ID":["-1", "-2","-3","-4","-5","-6","-7","-8","-9","-10"
                                            "-11", "-12","-13","-14","-15","-16","-17","-18","-19","-20"]}  
testrun = { "traffic":["-Two5M"], 
            "ID":["-1"]} 


class train_test_config:

    def __init__(self, conf_train, conf_test):

        self.train = self.get_config(conf_train)
        self.test = self.get_config(conf_test)


    def get_config(self, conf_name):

        try:
            conf =  getattr(self, conf_name)
            return conf()
        except: 
            print("Can not find configuration")
            raise

    def Read_Collection_train_c1(self):

        coll_prefix={}
        coll_prefix["Exception"] = []     
        
        #coll_prefix["1070323-C2-L1is20"] = testrun
        
        
        coll_prefix["1070323-C2-L1is20"] = BigRun  
        
        coll_prefix["1070322-C2-L1is30"] = BigRun
     
        coll_prefix["1070320-C2-L1is20"] = BigRun       
        
        coll_prefix["1070320-C2-L1is30"] = BigRun
        coll_prefix["1070319-C-L1is10"] = BigRun
       
        coll_prefix["1070317-C-L1is0"] = BigRun
        coll_prefix["Exception"].append("1070317-C-L1is0-Two15M-5")
        coll_prefix["Exception"].append("1070317-C-L1is0-Two20M-5")    
        coll_prefix["Exception"].append("1070317-C-L1is0-Two5M-1")  
        
        
        coll_prefix["1070316-L3is40-L4is25"] = BigRun    
        
        coll_prefix["1070315-L3is40-L4is40"] = BigRun
        coll_prefix["Exception"].append("1070315-L3is40-L4is40-Two10M-3")
        
        
        coll_prefix["1070314-L3is10-L4is40"] = BigRun
        coll_prefix["Exception"].append("1070314-L3is10-L4is40-Two20M-1")
        coll_prefix["Exception"].append("1070314-L3is10-L4is40-Two10M-5")
            
        coll_prefix["1070314-L3is10-L4is25"] = BigRun
        coll_prefix["Exception"].append("1070314-L3is10-L4is25-Two5M-1")
        
        coll_prefix["1070313-L3is25-L4is25"] = BigRun
        
        coll_prefix["1070312-L3is25-L4is40"] = BigBigRun    
        coll_prefix["1070307-bigrun-L3is10"] = BigBigRun      
        coll_prefix["1070308-bigrun-L3is25"] = BigBigRun    
        coll_prefix["1070309-bigrun-L3is40"] = BigBigRun  

        
        
        coll_prefix["1070222-clear"]={
                "ID":["","-2","-3","-4","-5"]
                }
            
        coll_prefix["1070223"]={
                "traffic":["-one10M","-one20M"], 
                "ID":["","-2","-3"]
                }    
        coll_prefix["1070227"]={
                "traffic":["-two10M","-two10Mt2"], 
                "ID":["","-2","-3"]
                }    
        coll_prefix["1070301"]={
                "traffic":["-two20M","-two20Mt2"], 
                "ID":["","-2","-3"]
                }    
        coll_prefix["1070302"]={
                "traffic":["-two20M","-two20M-L3is30"], 
                "ID":["","-2","-3"]
                }    
        coll_prefix["1070305"]={
                "traffic":["-two10M-L3is40","-two20M-L3is40"], 
                "ID":["","-2","-3"]
                }        
        
        coll_prefix["1070306"]={
                "traffic":["-two20M-L3is50","-two20M-L3isInf"], 
                "ID":["","-2","-3"]
                }
        
        return coll_prefix


    def Read_Collection_train_c2(self):
        coll_prefix={}
        coll_prefix["Exception"] = []
   
        #coll_prefix["1070323-C2-L1is20"] = testrun
        
        coll_prefix["1070323-C2-L1is20"] = BigRun
        
        coll_prefix["1070322-C2-L1is30"] = BigRun

        
        coll_prefix["1070320-C2-L1is20"] = BigRun
        
        
        coll_prefix["1070320-C2-L1is30"] = BigRun
        coll_prefix["1070319-C-L1is10"] = BigRun
        
        coll_prefix["1070317-C-L1is0"] = BigRun
        coll_prefix["Exception"].append("1070317-C-L1is0-Two15M-5")
        coll_prefix["Exception"].append("1070317-C-L1is0-Two20M-5")    
        coll_prefix["Exception"].append("1070317-C-L1is0-Two5M-1")  
        
        coll_prefix["1070316-L3is40-L4is25"] = BigRun    
        
        coll_prefix["1070315-L3is40-L4is40"] = BigRun
        coll_prefix["Exception"].append("1070315-L3is40-L4is40-Two10M-3")
        
        
        coll_prefix["1070314-L3is10-L4is40"] = BigRun
        coll_prefix["Exception"].append("1070314-L3is10-L4is40-Two20M-1")
        coll_prefix["Exception"].append("1070314-L3is10-L4is40-Two10M-5")
            
        coll_prefix["1070314-L3is10-L4is25"] = BigRun
        coll_prefix["Exception"].append("1070314-L3is10-L4is25-Two5M-1")
        
        coll_prefix["1070313-L3is25-L4is25"] = BigRun
        
        coll_prefix["1070312-L3is25-L4is40"] = BigBigRun    
        coll_prefix["1070307-bigrun-L3is10"] = BigBigRun      
        coll_prefix["1070308-bigrun-L3is25"] = BigBigRun    
        coll_prefix["1070309-bigrun-L3is40"] = BigBigRun
        
      
       
        coll_prefix["1070222-clear"]={
                "ID":["","-2","-3","-4","-5"]
                }
            
        coll_prefix["1070223"]={
                "traffic":["-one10M","-one20M"], 
                "ID":["","-2","-3"]
                }    
        coll_prefix["1070227"]={
                "traffic":["-two10M","-two10Mt2"], 
                "ID":["","-2","-3"]
                }    
        coll_prefix["1070301"]={
                "traffic":["-two20M","-two20Mt2"], 
                "ID":["","-2","-3"]
                }    
        coll_prefix["1070302"]={
                "traffic":["-two20M","-two20M-L3is30"], 
                "ID":["","-2","-3"]
                }    
        coll_prefix["1070305"]={
                "traffic":["-two10M-L3is40","-two20M-L3is40"], 
                "ID":["","-2","-3"]
                }        
        
        coll_prefix["1070306"]={
                "traffic":["-two20M-L3is50","-two20M-L3isInf"], 
                "ID":["","-2","-3"]
                }
        
        
        #---------Differ from C1----------
        coll_prefix["1070328-Clear"] = NoneTraf
        
        return coll_prefix


    def Read_Collection_test_c1(self):
        coll_prefix={}
        coll_prefix["Exception"] = []
        #coll_prefix["1070326-Free-NoInterfere"] = NoneTraf
        coll_prefix["1070328-Office"] = BigRun
        
        return coll_prefix

    def Read_Collection_test_c2(self):
        coll_prefix={}
        coll_prefix["Exception"] = []
        #coll_prefix["1070326-Free-NoInterfere"] = NoneTraf
        coll_prefix["1070328-Office"] = BigRun
        
        #---------Differ from C1----------
        coll_prefix["1070329-OfficeNight3"] = BigRun
        coll_prefix["1070329-OfficeNight2"] = BigRun
        
        return coll_prefix


    def Read_Collection_train_ct(self):
        coll_prefix={}
        coll_prefix["Exception"] = []
        coll_prefix["1070323-C2-L1is20"] = testrun
        
        return coll_prefix


    def Read_Collection_test_ct(self):
        coll_prefix={}
        coll_prefix["Exception"] = []
        coll_prefix["1070322-C2-L1is30"] = BigRun 
        
        return coll_prefix