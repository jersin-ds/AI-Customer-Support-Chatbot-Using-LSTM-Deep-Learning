import os
import pickle
import random
import tkinter as tk
from tkinter import scrolledtext

import numpy as np
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def patch_keras_compatibility():
    """Allow loading a model saved with a slightly newer Keras config."""
    original_embedding_init = Embedding.__init__
    original_dense_init = Dense.__init__
    original_glorot_init = GlorotUniform.__init__

    def embedding_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        return original_embedding_init(self, *args, **kwargs)

    def dense_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        return original_dense_init(self, *args, **kwargs)

    def glorot_init(self, *args, **kwargs):
        kwargs.pop("input_axes", None)
        kwargs.pop("output_axes", None)
        return original_glorot_init(self, *args, **kwargs)

    Embedding.__init__ = embedding_init
    Dense.__init__ = dense_init
    GlorotUniform.__init__ = glorot_init


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lstm_chatbot.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
RESPONSE_DICT_PATH = os.path.join(BASE_DIR, "response_dict.pkl")

patch_keras_compatibility()
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

with open(RESPONSE_DICT_PATH, "rb") as f:
    response_dict = pickle.load(f)


MAX_LENGTH = 30


def predict_intent(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction)

    intent = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    return intent, confidence


def chatbot(text):

    intent, confidence = predict_intent(text)

    if confidence < 0.50:
        return "Sorry, I don't understand your question."

    response = random.choice(
        response_dict[intent]
    )

    return response


def send_message():

    user_text = entry.get()

    if user_text.strip() == "":
        return

    chat_box.insert(
        tk.END,
        "You: " + user_text + "\n\n"
    )

    response = chatbot(user_text)

    chat_box.insert(
        tk.END,
        "Bot: " + response + "\n\n"
    )

    entry.delete(0, tk.END)

    chat_box.see(tk.END)


root = tk.Tk()

root.title("AI Customer Support Chatbot")

root.geometry("700x600")


title = tk.Label(
    root,
    text="AI Customer Support Chatbot",
    font=("Arial", 20, "bold")
)

title.pack(pady=15)


chat_box = scrolledtext.ScrolledText(
    root,
    width=80,
    height=25,
    font=("Arial", 11)
)

chat_box.pack(
    padx=20,
    pady=10
)


bottom_frame = tk.Frame(root)

bottom_frame.pack(
    fill=tk.X,
    padx=20,
    pady=10
)


entry = tk.Entry(
    bottom_frame,
    font=("Arial", 13)
)

entry.pack(
    side=tk.LEFT,
    fill=tk.X,
    expand=True,
    padx=(0, 10)
)


send_button = tk.Button(
    bottom_frame,
    text="Send",
    font=("Arial", 12, "bold"),
    command=send_message
)

send_button.pack(
    side=tk.RIGHT
)


entry.bind(
    "<Return>",
    lambda event: send_message()
)


if __name__ == "__main__":
    root.mainloop()