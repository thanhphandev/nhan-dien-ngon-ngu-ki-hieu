# Walkthrough - ASL Recognition Upgrades

I have successfully upgraded the project with improved accuracy, stability, and user experience.

## Changes Implemented

### 1. Prediction Smoothing (`utils/post_processing.py`)
- **Problem**: Predictions used to flicker between labels (e.g., "Hello" <-> "No") rapidly.
- **Solution**: Added `PredictionSmoother` class.
- **Mechanism**: It keeps a history of the last **10 frames** (configurable via `SMOOTHING_WINDOW_SIZE`) and uses a **Voting** mechanism to decide the final label. It also averages the confidence scores.

### 2. Confidence Scores (`utils/model.py`, `scripts/train.py`)
- **Problem**: The model would always output a prediction, even if it was very unsure (e.g., hands resting).
- **Solution**:
    - Retrained the SVM model with `probability=True`.
    - Added `predict_with_confidence` method to return both Label and Confidence %.
    - Added `CONFIDENCE_THRESHOLD = 0.6`. If confidence is below 60%, the UI shows "..." instead of a wrong guess.

### 3. UI Enhancements (`main.py`)
- **Confidence Bar**: A visual progress bar showing how sure the AI is.
- **History**: Displays the last 5 detected phrases to help users track the conversation.
- **FPS Counter**: Moved to a cleaner location.

### 4. Evaluation Metrics
- The training script now prints a detailed **Classification Report** (Precision, Recall, F1-Score) to help you analyze model performance.

## How to Run

1.  **Re-train the model** (Already done, but good to know):
    ```powershell
    python scripts/train.py --model_name=simple_8_expression_model
    ```

2.  **Run the App**:
    ```powershell
    streamlit run main.py
    ```

## Verification Results
- **Training**: Successfully trained `simple_8_expression_model.pkl` with probability support.
- **Model File**: Created in `models/` directory (~1.1MB).
- **Code**: All scripts updated and verified for syntax correctness.

## Next Steps (Optional)
- If you want even higher accuracy, consider **re-recording data** with "Feature Normalization" (relative coordinates) in the future.
