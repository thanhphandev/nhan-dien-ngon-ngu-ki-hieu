import os
import numpy as np
import argparse
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training model")

    parser.add_argument("--model_name", help="Name of the model",
                        type=str, default="model")
    parser.add_argument("--dir", help="Location of the model",
                        type=str, default="models")
    args = parser.parse_args()

    print("=" * 80)
    print(f"🧠 Starting Model Training: {args.model_name}")
    print("=" * 80)

    start_time = datetime.now()
    X, y, mapping = [], [], dict()

    data_dir = "data"
    pose_files = list(os.scandir(data_dir))

    if not pose_files:
        print(f"❌ No data found in '{data_dir}' folder.")
        exit(1)

    print(f"📂 Found {len(pose_files)} pose files in '{data_dir}'.")
    print("Loading data...")

    for current_class_index, pose_file in enumerate(pose_files):
        file_path = os.path.join(data_dir, pose_file.name)
        pose_data = np.load(file_path)

        X.append(pose_data)
        y += [current_class_index] * pose_data.shape[0]
        mapping[current_class_index] = pose_file.name.split(".")[0]

    X, y = np.vstack(X), np.array(y)
    print("✅ Data loaded successfully.")
    print(f"→ Total samples: {X.shape[0]}")
    print(f"→ Number of classes: {len(mapping)}\n")

    print("🚀 Training SVM model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # Enable probability=True để tính độ tin cậy
    model = SVC(decision_function_shape='ovo', kernel='rbf', probability=True)
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print("\n--- Training Summary ---")
    print(f"Training samples: {X.shape[0]}")
    print(f"Classes: {len(mapping)}")
    print(f"Train Accuracy: {round(train_accuracy * 100, 2)}%")
    print(f"Test Accuracy:  {round(test_accuracy * 100, 2)}%")
    
    from sklearn.metrics import classification_report
    print("\n--- Detailed Report ---")
    y_pred = model.predict(X_test)
    target_names = [mapping[i] for i in sorted(mapping.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names))

    os.makedirs(args.dir, exist_ok=True)
    model_path = os.path.join(args.dir, f"{args.model_name}.pkl")
    with open(model_path, "wb") as file:
        pickle.dump((model, mapping), file)

    duration = (datetime.now() - start_time).seconds
    print(f"\n💾 Model saved to: {model_path}")
    print(f"⏱️ Training completed in {duration} seconds.")
    print("=" * 80)
    print("✅ Done.")
