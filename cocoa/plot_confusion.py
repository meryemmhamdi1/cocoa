import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Few-shot on')
    plt.xlabel('Test on') #naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


if __name__ == "__main__":
    plot_confusion_matrix(cm=np.array([[19.1, 30.0, 30.1, 26.1, 14.6, 5.1],
                                       [34.3, 33.3, 30.0, 30.2, 13.5, 8.9],
                                       [19.2, 34.1, 29.9, 29.1, 5.8, 9.0],
                                       [18.3, 32.2, 44.4, 30.2, 6.7, 10.2],
                                       [19.1, 27.7, 30.1, 31.4, 5.8, 9.0],
                                       [24.1, 25.7, 33.2, 26.1, 30.9, 20.7],
                                       [24.4, 25.6, 34.4, 25.0, 20.2, 30.7],
                                       ]),
                          normalize=False,
                          target_names=['English', 'German', 'French', 'Italian', 'Portuguese', 'Japanese', 'Korean'],
                          title="Cross-lingual Jarvis Confusion Matrix")

    plot_confusion_matrix(cm=np.array([[66.3, 39.8],
                                       [86.3, 35.2],
                                       [68.1, 69.8],
                                       ]),
                          normalize=False,
                          target_names=['Spanish', 'Thai'],
                          title="Cross-lingual Intent Confusion Matrix")

    plot_confusion_matrix(cm=np.array([[55.8, 16.4],
                                       [45.6, 66.9],
                                       ]),
                          normalize=False,
                          target_names=['Spanish', 'Thai'],
                          title="Cross-lingual Slot Confusion Matrix")
