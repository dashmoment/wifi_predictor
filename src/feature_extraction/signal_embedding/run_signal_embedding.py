import keras as k
from keras import backend as kb
import pandas as pd

class signal_embeding:
    def __init__(self, model_path):
        self.model = k.models.load_model(model_path)
        
    def predict(self, MLdf):
        
         spike_cols = [col for col in MLdf.columns if 'SS_Subval' in col]
         df_embeded = MLdf[spike_cols]
        
         get_embeded_layer_output = kb.function([self.model.layers[0].input], [self.model.layers[-2].output])
         embeded = get_embeded_layer_output([df_embeded])[0]
         
         embeded_name = []
         
         for i in range(8):
             
             name = 'SS_Subval_emb_' + str(i)
             embeded_name.append(name)
             
            
         embeded = pd.DataFrame(data=embeded, columns=embeded_name)
         
         return embeded
        
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
