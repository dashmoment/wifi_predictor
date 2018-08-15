from keras.models import model_from_json

def save_model(
                model,
                graph_path="../models/focal_loss/model.json",
                weight_path="../models/focal_loss/model.h5"
               ):
    model_json = model.to_json()
    with open(graph_path, "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(weight_path)
        print("Saved model to disk")



def load_model(
                graph_path="../models/focal_loss/model.json",
                weight_path="../models/focal_loss/model.h5"
                ):

    json_file = open(graph_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_path)
    print("Loaded model from disk")
    
    return loaded_model