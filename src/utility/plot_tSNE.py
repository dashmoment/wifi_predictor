import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import numpy as np

class plot_tSNE:
    def __init__(
                    self, dataLength, 
                    data, label,  
                    color_platte = ['blue','red','orange', 'green'],
                    plot2d = True
                ):
        
        self.dataLength = dataLength
        self.data = data
        self.label = label
        self.color_platte = color_platte
        self.plot2d = plot2d
       

    def plot(self, plot_class=[0,1,2,3]):
        
        assert len(plot_class) <= len(self.color_platte), 'len(plot_class) <= len(color_platte)'
        
        if self.plot2d: n_components = 2
        else: n_components = 3
        X_tsne = TSNE(n_components=n_components,learning_rate=100).fit_transform(self.data[:self.dataLength])
        l = self.label[:self.dataLength]

        fig = plt.figure()
        if self.plot2d: ax = fig.add_subplot(111)
        else: ax = fig.add_subplot(111, projection='3d')
        for idx, cls in enumerate(plot_class):
        
            mask = l[l.idxmax(axis=1) == cls]
            res = np.array([X_tsne[i] for i in mask.index])

            if self.plot2d:
                ax.scatter(res[:,0], res[:,1], marker='*', color=self.color_platte[idx], label='class_'+str(cls))
            else:
                ax.scatter(res[:,0], res[:,1],  res[:,2], marker='*', color=self.color_platte[idx], label='class_'+str(cls))
            
        plt.legend(loc=(1, 0))
        plt.show()

        return X_tsne