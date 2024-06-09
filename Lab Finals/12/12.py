import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# AND-NOT function training data
and_not_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_not_labels = np.array([0, 0, 1, 0])

# XOR function training data
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_labels = np.array([0, 1, 1, 0])

# Function to create and train a model
def create_and_train_model(inputs, labels, epochs=3000):
    model = Sequential([
        Dense(2, input_dim=2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(inputs, labels, epochs=epochs, verbose=0)
    return model

# Train the AND-NOT model
and_not_model = create_and_train_model(and_not_inputs, and_not_labels)

# Train the XOR model
xor_model = create_and_train_model(xor_inputs, xor_labels)

# Function to test the model with specific inputs
def test_model(model, inputs):
    predictions = model.predict(inputs)
    predictions = [round(pred[0]) for pred in predictions]
    return predictions

# Test the models with training data
print("AND-NOT Model Predictions on training data:")
print(test_model(and_not_model, and_not_inputs))

print("\nXOR Model Predictions on training data:")
print(test_model(xor_model, xor_inputs))

# Manually test specific input values
and_not_test_input = [[0, 1]]
xor_test_input = [[1, 0]]

print("\nAND-NOT Model Prediction for input [0, 1]:")
print(test_model(and_not_model, and_not_test_input))

print("\nXOR Model Prediction for input [1, 0]:")
print(test_model(xor_model, xor_test_input))
