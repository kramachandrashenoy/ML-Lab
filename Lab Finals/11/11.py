import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Define the training data for AND function
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Define the training data for OR function
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

# Function to create and train a perceptron model
def create_and_train_model(inputs, labels, epochs=2000):
    model = Sequential([
        Dense(1, input_dim=2, activation='sigmoid')  # Single-layer perceptron
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(inputs, labels, epochs=epochs, verbose=0)
    return model

# Train the perceptron for AND function
model_and = create_and_train_model(X_and, y_and)

# Train the perceptron for OR function
model_or = create_and_train_model(X_or, y_or)

# Function to test the model with specific inputs
def test_model(model, inputs):
    predictions = model.predict(inputs)
    predictions = [round(pred[0]) for pred in predictions]
    return predictions

# Print training results
print("AND Function Predictions:")
print(test_model(model_and, X_and))

print("\nOR Function Predictions:")
print(test_model(model_or, X_or))

# Manually test specific input values
and_test_input = np.array([[1, 1]])
or_test_input = np.array([[0, 1]])

print("\nAND Function Prediction for input [1, 1]:")
print(test_model(model_and, and_test_input))

print("\nOR Function Prediction for input [0, 1]:")
print(test_model(model_or, or_test_input))
