import json

import numpy as np
import torch
import torch.nn as nn
from nltk.stem import PorterStemmer
from torch.utils.data import Dataset, DataLoader

from Model_Final import NetworkModel
from Utiles_Final import bag_of_words, tokenizer

stemmer = PorterStemmer()


def load_intents_data(file_path):
    with open(file_path, 'r') as json_file:
        return json.load(json_file)


def preprocess_intents(intents_data):
    vocabulary = []
    intent_tags = []
    intent_pairs = []

    for intent in intents_data['intents']:
        tag = intent['tag']
        intent_tags.append(tag)
        for pattern in intent['patterns']:
            tokenized_pattern = tokenizer(pattern)
            vocabulary.extend(tokenized_pattern)
            intent_pairs.append((tokenized_pattern, tag))

    unimportant_words = ['?', '.', '!']
    vocabulary = [stemmer.stem(word) for word in vocabulary if word not in unimportant_words]
    vocabulary = sorted(set(vocabulary))
    intent_tags = sorted(set(intent_tags))

    return intent_pairs, vocabulary, intent_tags


def prepare_training_data(intent_pairs, vocabulary, intent_tags):
    X_training = []
    Y_training = []
    for (sentence, tag) in intent_pairs:
        word_bag = bag_of_words(sentence, vocabulary)
        X_training.append(word_bag)
        label = intent_tags.index(tag)
        Y_training.append(label)

    X_training = np.array(X_training)
    Y_training = np.array(Y_training)

    return X_training, Y_training


class IntentDataset(Dataset):
    def __init__(self, X, Y):
        self.sample_num = len(X)
        self.x_data = X
        self.y_data = Y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.sample_num


def train_neural_model(X_training, Y_training, input_dimension, hidden_dimension, output_dimension, epoch_num,
                       batch_num, learning_rate_val):
    dataset = IntentDataset(X_training, Y_training)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_num, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    neural_model = NetworkModel(input_dimension, hidden_dimension, output_dimension).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(neural_model.parameters(), lr=learning_rate_val)

    for epoch in range(epoch_num):
        correct_predictions = 0
        total_predictions = 0
        total_loss = 0

        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = neural_model(words)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted_labels = torch.max(outputs, dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
            total_loss += loss.item()

        accuracy = correct_predictions / total_predictions
        average_loss = total_loss / len(train_loader)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch: {epoch + 1}/{epoch_num}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')

    print(f'Final loss: {average_loss:.4f}, Final accuracy: {accuracy:.4f}')

    return neural_model



def save_trained_data(model, input_dimension, hidden_dimension, output_dimension, vocabulary, intent_tags, file_path):
    trained_data = {
        "model_state": model.state_dict(),
        "input_size": input_dimension,
        "hidden_size": hidden_dimension,
        "output_size": output_dimension,
        "all_words": vocabulary,
        "tags": intent_tags
    }

    torch.save(trained_data, file_path)
    print(f'Training finished. Data saved to: {file_path}')


def load_and_preprocess_intents(file_path):
    intents_data = load_intents_data(file_path)
    intent_pairs, vocabulary, intent_tags = preprocess_intents(intents_data)
    return intent_pairs, vocabulary, intent_tags


def main():
    file_path = 'intents.json'
    intent_pairs, vocabulary, intent_tags = load_and_preprocess_intents(file_path)
    X_training, Y_training = prepare_training_data(intent_pairs, vocabulary, intent_tags)

    epoch_num = 1000
    batch_num = 8
    learning_rate_val = 0.001
    input_dimension = len(X_training[0])
    hidden_dimension = 8
    output_dimension = len(intent_tags)
    print(f"Input size: {input_dimension}, Output size: {output_dimension}")

    neural_model = train_neural_model(X_training, Y_training, input_dimension, hidden_dimension, output_dimension,
                                      epoch_num, batch_num, learning_rate_val)

    file_path = "data.pth"
    save_trained_data(neural_model, input_dimension, hidden_dimension, output_dimension, vocabulary, intent_tags,
                      file_path)


if __name__ == '__main__':
    main()
