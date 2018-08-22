
def calculate_classWeight(label):    
    
    class_num = label.sum()
    _weight = len(label)/(class_num + 1e-4)
    classWeight = {}
    for idx, value in enumerate(_weight):
        classWeight[idx] = value
    
    return classWeight