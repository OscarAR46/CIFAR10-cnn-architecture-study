"""
Testing file for continuous changes so that everything works without GPU test
"""


import config
from data_utils import DataPreprocessor
from models import get_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

preprocessor = DataPreprocessor()
x_train, y_train, x_val, y_val, x_test, y_test = preprocessor.load_and_preprocess()

# Small subset for quick testing
x_train_full = x_train
y_train_full = y_train
x_val_full = x_val
y_val_full = y_val

model = get_model("vgg_style")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train_full, y_train_full,
    validation_data=(x_val_full, y_val_full),
    epochs=50,
    batch_size=64,
    verbose=1
)

# Test accuracy
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"test accuracy: {test_acc:.2%}")
