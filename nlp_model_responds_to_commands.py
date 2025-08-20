import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1) Tiny labeled dataset (expand later)
data = [
    ("move forward", "MOVE"),
    ("move backward", "MOVE"),
    ("go ahead", "MOVE"),
    ("advance three meters", "MOVE"),
    ("step back two meters", "MOVE"),

    ("turn left", "TURN"),
    ("turn right", "TURN"),
    ("rotate left ninety degrees", "TURN"),
    ("rotate right forty five degrees", "TURN"),
    ("face north", "TURN"),

    ("pick up the red cube", "PICK"),
    ("grab the object", "PICK"),
    ("pick the bottle", "PICK"),
    ("lift the box", "PICK"),
    ("collect the item", "PICK"),

    ("place it on the table", "PLACE"),
    ("put the cube down", "PLACE"),
    ("drop the object here", "PLACE"),
    ("set the bottle on the shelf", "PLACE"),
    ("put it there", "PLACE"),

    ("stop", "STOP"),
    ("halt", "STOP"),
    ("do not move", "STOP"),
    ("freeze", "STOP"),
    ("hold position", "STOP"),
]

texts = [t for t, _ in data]
labels = [l for _, l in data]
label2id = {l:i for i,l in enumerate(sorted(set(labels)))}
id2label = {v:k for k,v in label2id.items()}
y = np.array([label2id[l] for l in labels])

# 2) Tokenize + pad
tokenizer = Tokenizer(oov_token="<OOV>") 
tokenizer.fit_on_texts(texts)
X_seq = tokenizer.texts_to_sequences(texts)
max_len = max(len(s) for s in X_seq)
X = pad_sequences(X_seq, maxlen=max_len, padding='post')

vocab_size = len(tokenizer.word_index) + 1
num_classes = len(label2id)

# 3) Model (Embedding + BiLSTM)
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    Bidirectional(LSTM(64)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4) Early stopping
es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# 5) Train (small data → more epochs but early stop protects us)
history = model.fit(X, y, epochs=500, batch_size=8, callbacks=[es], verbose=0)

# 6) Inference helper
def predict_intent(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    probs = model.predict(pad, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    return id2label[pred_id], float(np.max(probs))

# 7) Quick tests
tests = [
    "move ahead",
    "go back two meters",
    "turn right ninety degrees",
    "pick up the bottle",
    "put it on the table",
    "stop moving now"
]
for t in tests:
    intent, p = predict_intent(t)
    print(f"{t!r} -> {intent} (p={p:.2f})")
