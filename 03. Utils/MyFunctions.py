import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, product
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import interp
import seaborn as sns
from imageio import imread

def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def after_test(classes_dict,result_test,test_batches = None,both=1):
    # Tomo la máxima de las probabilidades de pertenencia a una clase para cada imagen.
    preds = result_test.argmax(axis=-1)
    preds_classes = [classes_dict[p] for p in preds]
    print(pd.Series(preds_classes).value_counts())
    if(both==1):
        gts = np.array([classes_dict[p] for p in test_batches.classes])
        print(classification_report(gts,preds_classes))
        return [preds_classes,gts]
    else: 
        return preds_classes

  

def plot_confusion_matrix(gts,preds_classes, classes,path,title=' ',cmap=plt.cm.Reds ):
    cm = confusion_matrix(gts,preds_classes)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center", fontsize=16,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta verdadera', fontsize=14)
    plt.xlabel('Predicción del modelo', fontsize=14)
    plt.savefig(path +'.png',bbox_inches='tight')
    plt.show()
       


def precision_recall_many(result_test,gts,NUM_CLASSES,classes_dict,path):
    Y_test = pd.get_dummies(gts).to_numpy()
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(NUM_CLASSES):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            result_test[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], result_test[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    result_test.ravel())
    average_precision["macro"] = average_precision_score(Y_test, result_test,
                                                     average="macro")
    colors = cycle(sns.color_palette("husl", NUM_CLASSES))

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('Curvas iso-f1')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                ''.format(average_precision["macro"]))

    for i, color in zip(range(NUM_CLASSES), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall de la clase {0:1} (area = {1:0.2f})'
                    ''.format(classes_dict.get(i), average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall',fontsize = 14)
    plt.ylabel('Precision',fontsize = 14)
    plt.title('Precision-Recall para múltiples clases',fontsize = 16)
    plt.legend(lines, labels, loc=(0, -.45), prop=dict(size=12))

    plt.savefig(path +'.png',bbox_inches='tight')
    plt.show()
   

def precision_recall_global(result_test, gts, NUM_CLASSES,path):
    Y_test = pd.get_dummies(gts).to_numpy()
    
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(NUM_CLASSES):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            result_test[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], result_test[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        result_test.ravel())
    average_precision["macro"] = average_precision_score(Y_test, result_test,
                                                        average="macro")
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall',fontsize=14)
    plt.ylabel('Precision',fontsize=14)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Curva Precision Recall Global Macro',fontsize=16)
    plt.savefig(path +'.png',bbox_inches='tight')
    


def plot_ROC(result_test,gts,NUM_CLASSES,classes_dict,path): 
    # y_score = result_test
    y_test = pd.get_dummies(gts).to_numpy()
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], result_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), result_test.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= NUM_CLASSES

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average curva ROC (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average curva ROC (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors= cycle(sns.color_palette("husl", NUM_CLASSES))
    lw=2

    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='Curva ROC de la clase {0:1} (area = {1:0.2f})'
                ''.format(classes_dict.get(i), roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Especificidad',fontsize=14)
    plt.ylabel('Sensibilidad',fontsize=14)
    plt.title('Curvas ROC', fontsize=16)
    plt.legend(loc="lower right")
    plt.savefig(path +'.png',bbox_inches='tight')
    plt.show()
    


    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average curva ROC (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='micro-average curva ROC (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='Curva ROC de la clase {0} (area = {1:0.2f})'
                ''.format(classes_dict.get(i), roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('1 - Especificidad',fontsize=14)
    plt.ylabel('Sensibilidad',fontsize=14)
    plt.title('Curvas ROC Ampliadas',fontsize=16)
    plt.legend(loc="lower right")
    plt.savefig(path +'_zoom.png',bbox_inches='tight')
    plt.show()
   
def save_image_results(df,image_path,save_path,complete_path = False,bad=True,plot=False):
    if complete_path == True:
        for index, row in df.iterrows():
            read_path =  row["ImageID"]
            imagen = imread(read_path)
            plt.imshow(imagen,cmap=plt.cm.Greys_r)
            plt.text(0,0, 'Ground Truth: ' + row["Projection"] +' Prediction: ' + row["Pred"],fontsize= 12,transform=plt.gcf().transFigure)
            plt.axis('off')
            if bad ==True:
                plt.savefig(save_path +'/Mal_Clasificadas/' + row['ImageID'].split('/')[-1],bbox_inches='tight')
            else:
                plt.savefig(save_path +'/Bien_Clasificadas/' + row['ImageID'].split('/')[-1],bbox_inches='tight')
            if plot ==True:
                plt.show()
    else:
        for index, row in df.iterrows():
            read_path =  image_path + row["ImageID"]
            imagen = imread(read_path)
            plt.imshow(imagen,cmap=plt.cm.Greys_r)
            plt.text(0,0, 'Ground Truth: ' + row["Projection"] +' Prediction: ' + row["Pred"],fontsize= 12,transform=plt.gcf().transFigure)
            plt.axis('off')
            if bad ==True:
                plt.savefig(save_path +'/Mal_Clasificadas/' + row['ImageID'].split('/',1)[1],bbox_inches='tight')
            else:
                plt.savefig(save_path +'/Bien_Clasificadas/' + row['ImageID'].split('/',1)[1],bbox_inches='tight')
            if plot ==True:
                plt.show()
#plt.savefig(savingpath, bbox_inches='tight', pad_inches=0)
    
