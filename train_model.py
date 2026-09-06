import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    performance_on_categorical_slice
)

# 1. Load the census.csv data.
data_path = os.path.join("data", "census.csv")
data = pd.read_csv(data_path)

# Clean column names in case there are leading/trailing spaces
data.columns = data.columns.str.strip()

# 2. Split the data into a training dataset and a test dataset.
train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data['salary'])

# Define categorical features based on the standard census dataset structure
categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
target = "salary"

# 3. Use the process_data function to process both the training and test data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=categorical_features,
    label=target,
    training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=categorical_features,
    label=target,
    training=False,
    encoder=encoder,
    lb=lb
)

# 4. Use the train_model function to train the model on the training dataset.
print("Training model...")
model = train_model(X_train, y_train)

# 5. Use the inference function to run model inferences on the test dataset.
preds = inference(model, X_test)

# Compute and print overall test metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Test Metrics -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

# Save the model and the encoder to the model/ directory
os.makedirs("model", exist_ok=True)
save_model(model, os.path.join("model", "model.pkl"))
save_model(encoder, os.path.join("model", "encoder.pkl"))
print("Model saved to model/model.pkl")
print("Encoder saved to model/encoder.pkl")

# 6. Compute performance on data slices and save output to slice_output.txt
slice_output_path = "slice_output.txt"
print(f"Computing slice performance and saving to {slice_output_path}...")

with open(slice_output_path, "w") as f:
    for col in categorical_features:
        for val in test[col].unique():
            # Filter the test set for the specific slice value
            df_slice = test[test[col] == val]
            if len(df_slice) == 0:
                continue
            
            # Compute slice metrics using your performance_on_categorical_slice function
            prec, rec, f1 = performance_on_categorical_slice(
                data=test,
                column_name=col,
                slice_value=val,
                categorical_features=categorical_features,
                label=target,
                encoder=encoder,
                lb=lb,
                model=model
            )
            
            # Format output line matching the required text log layout
            output_line = f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}\n{col}: {val}, Count: {len(df_slice)}\n"
            f.write(output_line)

print("Slice evaluation complete!")