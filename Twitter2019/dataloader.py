import torch
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader, random_split
import pickle


def calculate_class_weights(labels):
    """
    :param labels:
    :return:
    """
    labels = torch.argmax(labels, dim=1)

    class_counts = Counter(labels.cpu().numpy())
    total_samples = len(labels)

    class_weights = {}
    for label, count in class_counts.items():
        class_frequency = count / total_samples
        class_weights[label] = 1.0 / class_frequency

    class_weights = [class_weights[i] for i in range(len(class_weights))]
    weight_tensor = torch.tensor(class_weights)

    return weight_tensor


def count_samples_per_class(predicted_labels, num_classes):
    """
    :param predicted_labels:
    :param num_classes:
    :return:
    """
    predicted_labels = torch.argmax(predicted_labels, dim=1)
    one_hot_labels = torch.nn.functional.one_hot(predicted_labels, num_classes=num_classes)
    class_counts = torch.sum(one_hot_labels, dim=0)
    total_count = torch.sum(class_counts)
    counts_dict = {i: count.item() for i, count in enumerate(class_counts)}
    counts_dict_por = {i: (count.item(), count.item() / total_count.item()) for i, count in enumerate(class_counts)}

    return counts_dict, counts_dict_por


def dataloader(batch_size, encode):
    if encode == 'bert':
        filename = 'data/data_twitter_bert.pkl'
    else:
        filename = 'data/data_twitter_roberta.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    label_train = data['label_train']
    label_encoder_train = data['label_encoder_train']
    word_train_1 = data['text_train_1']
    word_train_2 = data['text_train_2']
    word_train_3 = data['text_train_3']
    word_train_4 = data['text_train_4']
    word_train_all = data['text_train_all']
    img_train = data['visual_train']
    audio_train = data['audio_train']

    print(label_train)
    counts_dict, counts_dict_por = count_samples_per_class(label_train, 2)
    print(counts_dict)
    print(counts_dict_por)

    print(label_train.shape, label_encoder_train.shape, word_train_1.shape, word_train_2.shape, word_train_3.shape,
          word_train_4.shape, word_train_all.shape, img_train.shape, audio_train.shape)

    label_val = data['label_val']
    word_val_1 = data['text_val_1']
    word_val_2 = data['text_val_2']
    word_val_3 = data['text_val_3']
    word_val_4 = data['text_val_4']
    word_val_all = data['text_val_all']
    img_val = data['visual_val']
    audio_val = data['audio_val']

    print(label_val)
    counts_dict_val, counts_dict_por_val = count_samples_per_class(label_val, 2)
    print(counts_dict_val)
    print(counts_dict_por_val)

    print(label_val.shape, word_val_1.shape, word_val_2.shape, word_val_3.shape, word_val_4.shape, word_val_all.shape,
          img_val.shape, audio_val.shape)

    label_test = data['label_test']
    word_test_1 = data['text_test_1']
    word_test_2 = data['text_test_2']
    word_test_3 = data['text_test_3']
    word_test_4 = data['text_test_4']
    word_test_all = data['text_test_all']
    img_test = data['visual_test']
    audio_test = data['audio_test']

    print(label_test)
    counts_dict_test, counts_dict_por_test = count_samples_per_class(label_test, 2)
    print(counts_dict_test)
    print(counts_dict_por_test)

    print(label_test.shape, word_test_1.shape, word_test_2.shape, word_test_3.shape,
          word_test_4.shape, word_test_all.shape, img_test.shape, audio_test.shape)

    train_dataset = TensorDataset(img_train, word_train_1, word_train_2, word_train_3,
                                  word_train_4, word_train_all, audio_train, label_train, label_encoder_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(img_val, word_val_1, word_val_2, word_val_3, word_val_4, word_val_all, audio_val,
                                label_val)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, img_test, word_test_1, word_test_2, word_test_3, \
           word_test_4, word_test_all, audio_test, label_test

# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dataloader(batch_size=32)
