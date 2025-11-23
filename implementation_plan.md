# Implementation Plan - ASL Recognition Upgrade

This plan aims to improve the accuracy, stability, and user experience of the Sign Language Recognition project.

## User Review Required
> [!IMPORTANT]
> **Data Re-recording might be needed for "Feature Normalization"**
> To achieve the *highest* accuracy, we should normalize the hand coordinates (relative to wrist) to make the model independent of user position. However, this changes the feature format, requiring you to **re-record all data**.
> **Decision:** For this iteration, I will focus on **Model & Inference Optimizations** (Smoothing, Confidence Scores) which work with your *existing data*. I will NOT change the feature extraction logic yet to avoid breaking your current dataset.

## Proposed Changes

### 1. Core Logic & Inference
#### [NEW] [utils/post_processing.py](file:///e:/AI/DEMO/DEMO/utils/post_processing.py)
- Create a `PredictionSmoother` class.
- Uses a buffer (e.g., last 10 frames) to smooth predictions.
- Implements "Voting" or "Average" strategy to remove flickering.

#### [MODIFY] [utils/model.py](file:///e:/AI/DEMO/DEMO/utils/model.py)
- Update `ASLClassificationModel` to support `predict_proba` (probability scores).
- Add method to return (label, confidence).

### 2. Model Training
#### [MODIFY] [scripts/train.py](file:///e:/AI/DEMO/DEMO/scripts/train.py)
- Enable `probability=True` in `SVC`.
- Add `classification_report` and `confusion_matrix` to evaluate model performance detailedly.
- Save the model with probability support.

### 3. User Interface (Streamlit)
#### [MODIFY] [main.py](file:///e:/AI/DEMO/DEMO/main.py)
- Integrate `PredictionSmoother`.
- Add **Confidence Threshold**: Only display prediction if confidence > `0.6` (configurable).
- Add **Confidence Bar**: Visual indicator of how sure the model is.
- Add **History**: Show last 5 detected phrases.
- Improve layout and styling.

### 4. Configuration
#### [MODIFY] [config.py](file:///e:/AI/DEMO/DEMO/config.py)
- Add `SMOOTHING_WINDOW_SIZE = 10`.
- Add `CONFIDENCE_THRESHOLD = 0.6`.

## Verification Plan

### Automated Tests
- Run `scripts/train.py` to verify model training works with `probability=True`.
- Check if `models/*.pkl` is generated correctly.

### Manual Verification
- Run `streamlit run main.py`.
- Test with webcam:
    - Verify predictions are stable (no flickering).
    - Verify "Unknown" or no prediction when hands are not performing a known gesture (low confidence).
    - Check UI elements (Confidence bar, History).
