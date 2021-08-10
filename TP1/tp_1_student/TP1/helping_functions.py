#!/usr/bin/env python3
# License: BSD 3 clause 
# Inspired from https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html

from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

""" -----------------------------------------------------------------------------------------
Vizualize the predicted and true classification
INPUT: 
    - clf: trained classifier
    - X: feature data
    - X_train y_train: training data and objective
    - X_test y_test: testing data and objective
    - score: score of the classifier on X 
    - n_neighbors: value of k in the classifier clf
    - title: title of your graph
    - x_label: name of feature 1
    - y_label: name of feature 2
----------------------------------------------------------------------------------------- """

cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

def vis_clf(clf, X, X_train, y_train, X_test, y_test, score, n_neighbors, title="", x_label="", y_label=""):
    
    x_min = np.amin(X, axis=0)  # minima over the columns
    x_max = np.amax(X, axis=0)  # maxima over the columns    
    
    # For better visu
    x_min = x_min - 0.1 * (x_max - x_min)
    x_max = x_max + 0.1 * (x_max - x_min)
    
    h = 20 if len(x_min) == 3 else 100  # number of points in the grid
    
    grid = np.transpose(np.linspace(x_min, x_max, h))
    xx = np.meshgrid(*grid)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh
    Z = clf.predict(np.transpose(list(map(lambda x: x.ravel(), xx))))
    
    # Put the result into a color plot
    if len(xx) == 2:
        fig, ax = plt.subplots()
        Z = Z.reshape(xx[0].shape)
        plt.pcolormesh(*xx, Z, cmap=cmap_light, alpha=.8, shading='auto')
    else:  # 3D image
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        mesh = list(map(lambda x: x.ravel(),xx))
        ax.scatter(*mesh, c=Z, cmap=cmap_light, alpha=0.2)
        
    # Plot also the training and testing points   
    scatter = ax.scatter(*np.transpose(X_train), c=y_train, cmap=cmap_bold, \
                         edgecolor='k', s=20)
    scatter = ax.scatter(*np.transpose(X_test), c=y_test, cmap=cmap_bold, \
                         edgecolor='k', s=20, marker="s")
    
    ax.set_title("{} (k = {})".format(title, n_neighbors))
    ax.autoscale(enable=True, axis='both', tight=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if len(xx) == 3:
        ax.set_zlabel('Scaled Lot Frontage')

    l_e = (scatter.legend_elements()[0], ['Low rated', 'High rated'])
    legend1 = ax.legend(*l_e,
                    loc="upper left", title="Classes")
    ax.add_artist(legend1)
    legend_elements = [Line2D([0], [0], marker='o', color='None', label='Scatter',
                          markeredgecolor='grey', markersize=6),
                   Line2D([0], [0], marker='s', color='None', label='Scatter',
                          markeredgecolor='grey', markersize=6)]
    legend2 = ax.legend(handles=legend_elements, labels=['Train', 'Test'],loc="upper right")
    ax.add_artist(legend2)
    
    position = [0.9, 0.1] if len(xx) == 2 else [0.9, 0.1, 0.1]
    ax.text(*position, '{:.2f}'.format(score), size=15,
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.show()
