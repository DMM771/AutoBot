import json
import random
import sys

import torch

from model import NetworkModel
from utils import tokenizer, bag_of_words


class ChatBot:
    def __init__(self, bot_name="AutoBot"):
        self.bot_name = bot_name
        self.processing_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.intent_patterns = self.load_intents('intents.json')
        self.data_info = self.load_data_info("data.pth")
        self.neural_model = self.load_trained_model()

    @staticmethod
    def load_intents(file_path):
        with open(file_path, 'r') as json_file:
            intent_data = json.load(json_file)
        return intent_data

    @staticmethod
    def load_data_info(file_path):
        data_parameters = torch.load(file_path)
        return data_parameters

    def load_trained_model(self):
        trained_model = NetworkModel(self.data_info["input_size"], self.data_info["hidden_size"], self.data_info["output_size"]).to(self.processing_device)
        trained_model.load_state_dict(self.data_info["model_state"])
        trained_model.eval()
        return trained_model

    def predict_intent_tag(self, user_input):
        tokenized_sentence = tokenizer(user_input)
        X = bag_of_words(tokenized_sentence, self.data_info['all_words'])
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(self.processing_device)

        model_output = self.neural_model(X)
        _, predicted_idx = torch.max(model_output, dim=1)

        return self.data_info['tags'][predicted_idx.item()], model_output, predicted_idx

    def generate_bot_response(self, intent_tag, model_output, predicted_idx):
        output_probabilities = torch.softmax(model_output, dim=1)
        max_probability = output_probabilities[0][predicted_idx.item()]
        if max_probability.item() > 0.75:
            for intent in self.intent_patterns['intents']:
                if intent_tag == intent["tag"]:
                    return f"{random.choice(intent['responses'])}"
        else:
            return f"I'm sorry, I didn't quite get that..."

    def initiate_chat(self):
            intent_tag, model_output, predicted_idx = self.predict_intent_tag(sys.argv[1])
            bot_reply = self.generate_bot_response(intent_tag, model_output, predicted_idx)
            print(bot_reply)


if __name__ == "__main__":
    chatbot_instance = ChatBot()
    chatbot_instance.initiate_chat()
