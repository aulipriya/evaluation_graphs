import os
import matplotlib.pyplot as plt
import mplcyberpunk
from utills import (load_config_from_yaml, load_labels_from_df, load_label_csv, load_model, predict, postprocess_output,
                    build_eval_transformation)
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import PIL
import pandas as pd
import numpy as np
plt.style.use("cyberpunk")


def create_training_vs_validation_loss_curve(training_data, validation_data, title, xlabel, ylabel, output_image_name):
    graph_lines = {'Training Loss': training_data, 'Validation Loss': validation_data}
    for graph_subject, graph_data in graph_lines.items():
        x_items = [x for x in graph_data.keys()]
        y_items = [y for y in graph_data.values()]
        plt.plot(x_items, y_items, linewidth=1.5, linestyle='--', label=f'{graph_subject}')
        plt.locator_params(integer=True)

    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    mplcyberpunk.add_glow_effects()
    plt.savefig(output_image_name)
    plt.show()


def generate_confusion_matrix(gt, pred, classes, checkpoint, result_folder, i=""):
    cm = confusion_matrix(gt, pred)
    commat = pd.DataFrame(np.mat(cm), index=classes,
                          columns=classes)
    plt.figure()
    sns.heatmap(commat, annot=True, cmap="Blues", fmt='d')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'{checkpoint} Confusion Matrix')
    plt.savefig(f'{result_folder}/{checkpoint}_{i}_cm.jpg')
    plt.show()
    return cm


def generate_precision_recall_graph(cm, classes, checkpoint, result_folder, i=None):
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    # Prepare precision recall graph for each category
    plt.figure()
    pre_rec = pd.DataFrame({'classes': classes,
                            'precision': precision,
                            'recall': recall})
    tidy = pre_rec.melt(id_vars='classes').rename(columns=str.title)
    ax = sns.barplot(x='Classes', y='Value', hue='Variable', data=tidy)
    for patch in ax.patches:
        ax.text(patch.get_x() + patch.get_width() / 2., 0.2 * patch.get_height(),
                round(patch.get_height()*100, 2),
                ha='center', va='bottom', fontsize=14, fontweight='bold',  rotation=90, color='white')
    total_precision = np.mean(precision)
    total_recall = np.mean(recall)
    total_f1 = (2*total_precision*total_recall)/(total_recall+total_precision)
    plt.ylabel('Precision/Recall')
    plt.xlabel('Classes')
    mplcyberpunk.add_glow_effects()
    plt.title(f'Precison {round(total_precision*100, 2)}%   Recall {round(total_recall*100, 2)}% of {checkpoint}')
    plt.savefig(f'{result_folder}/{checkpoint}_{i}_prec_rec.jpg')
    plt.show()
    return total_precision, total_recall, total_f1


def generate_barplot_for_one_metric(metric_dict, metric_name, experiment, result_folder):
    plt.figure(figsize=(12, 8))
    metric_df = pd.DataFrame.from_dict(metric_dict, orient='index').reset_index()
    metric_df.columns = ['checkpoints', 'value']
    ax = sns.barplot(x='checkpoints', y='value', data=metric_df, hue='checkpoints', palette='mako')
    for patch in ax.patches:
        ax.text(patch.get_x() + patch.get_width() / 2., 0.2 * patch.get_height(),
                str(round(patch.get_height() * 100, 2))+"%",
                ha='center', va='bottom', fontsize=14, fontweight='bold', rotation=90, color='white')
    plt.ylabel(metric_name.capitalize())
    plt.xlabel('Epochs')
    ax.set_xticklabels(labels=[checkpoint.split('.')[0] for checkpoint in metric_df['checkpoints'].tolist()], rotation=90)
    plt.title(f'{metric_name.capitalize()}s of Experiment {experiment}')
    plt.savefig(f'{result_folder}/exp_{experiment}_{metric_name}_bar_chart.jpg')
    plt.show()


def generate_roc(gt, pred_probs, classes, result_folder, checkpoint, counter=None):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        gt_i = [1 if t == i else 0 for t in gt]
        pred_probs_i = [probs[i] for probs in pred_probs]
        fpr[i], tpr[i], _ = roc_curve(gt_i, pred_probs_i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.grid(True)
    mplcyberpunk.add_glow_effects()
    plt.savefig(f'{result_folder}/{checkpoint}_{counter}_auc.jpg')
    plt.show()


def evaluate_models(config_path):
    config = load_config_from_yaml(config_path)
    models_path = config['checkpoints_path']
    test_set_path = config['test_set_path']
    test_set_csv_path = config['test_set_csv_path']
    # Build transformation
    tranformation = build_eval_transformation(config)
    test_set_csv = load_label_csv(test_set_csv_path, config)
    images_path = [image_path for image_path in os.listdir(test_set_path) if image_path.endswith('.jpg')]
    labels = {image_path: load_labels_from_df(test_set_csv, image_path, config['classes'])
              for image_path in images_path}

    result_folder = os.path.join(config['result_folder'], str(config['experiment']))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    classwise_accuracies = {}
    classwise_images = {}
    classwise_accurates = {}

    for value in labels.values():
        if not isinstance(value, tuple):
            classwise_images[value] = classwise_images[value] + 1 if value in list(classwise_images.keys()) else 1
        else:
            for i in range(2):
                classwise_images[value[i]] = classwise_images[value[i]] + 1 \
                    if value[i] in list(classwise_images.keys()) else 1
    accurates = {}
    precisions = {}
    recalls = {}
    f1s = {}
    for model_path in os.listdir(models_path):
        gt = []
        preds = []
        pred_probs = []
        accurates[model_path] = 0
        classwise_accurates[model_path] = {key: 0 for key in classwise_images.keys()}
        model = load_model(os.path.join(models_path, model_path), config)
        for image_name in images_path:
            label = labels[image_name]
            image = PIL.Image.open(os.path.join(test_set_path, image_name))
            probabilities = predict(image, tranformation, model, config['device'])
            pred_probs.append([probabilities[list(config['classes'].keys())[1]].squeeze().tolist(),
                               probabilities[list(config['classes'].keys())[0]].squeeze().tolist()]) if len(
                probabilities.keys()) == 2 else pred_probs.append(
                probabilities[list(config['classes'].keys())[0]].squeeze().tolist())
            final_prediction = postprocess_output(probabilities, config['classes'])
            gt.append([config['classes'][list(config['classes'].keys())[1]].index(label[1]),
                       config['classes'][list(config['classes'].keys())[0]].index(label[0])]) if len(
                list(config['classes'].keys())) == 2 else gt.append(
                config['classes'][list(config['classes'].keys())[0]].index(label))
            preds.append([config['classes'][list(config['classes'].keys())[1]].index(
                final_prediction[list(config['classes'].keys())[1]]),
                config['classes'][list(config['classes'].keys())[0]].index(
                    final_prediction[list(config['classes'].keys())[0]])]) if len(
                list(config['classes'].keys())) == 2 else preds.append(
                config['classes'][list(config['classes'].keys())[0]].index(
                    final_prediction[list(config['classes'].keys())[0]]))
            # Calculate accurates
            i = 0
            for key, value in final_prediction.items():
                if label == final_prediction[key]:
                    classwise_accurates[model_path][label] += 1
                    accurates[model_path] += 1
                    i += 1
        cm = generate_confusion_matrix(gt, preds, config['classes'][list(config['classes'].keys())[0]],
                                       model_path, result_folder, '')
        precision, recall, f1 = generate_precision_recall_graph(cm, config['classes'][
            list(config['classes'].keys())[0]], model_path,
                                                                result_folder, '')
        precisions[model_path] = precision
        recalls[model_path] = recall
        f1s[model_path] = f1
        generate_roc(gt, pred_probs, config['classes'][list(config['classes'].keys())[0]], result_folder,
                     model_path, '')
    for key, value in accurates.items():
        classwise_accuracies[key] = {}
        for label_class, class_value in classwise_accurates[key].items():
            classwise_accuracies[key][label_class] = float(classwise_accurates[key][label_class] / classwise_images[label_class])
    # Generate overall precision, recall and f1 graph
    generate_barplot_for_one_metric(precisions, metric_name='precision', experiment=config['experiment'],
                                    result_folder=result_folder)
    generate_barplot_for_one_metric(recalls, metric_name='recall', experiment=config['experiment'],
                                    result_folder=result_folder)
    generate_barplot_for_one_metric(f1s, metric_name='f1', experiment=config['experiment'],
                                    result_folder=result_folder)



