# Queue Wait Category Predictor

A Streamlit web app that predicts a queue wait category (**Short**, **Medium**, or **Long**) and estimates a token's current position using the selected day, time slot, and token number.

## Features
- Generates a reproducible simulated queue dataset.
- Trains and compares Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors models.
- Selects the model with the best weighted F1-score.
- Displays a wait-category prediction, estimated queue position, and advice.
- Shows model metrics and a small dataset preview.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload the contents of this folder to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create a new app.
3. Select your repository and branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

Make sure `app.py` and `requirements.txt` are in the repository root (or set the correct path if you put them in a subfolder).

## Important note
This project uses simulated data, not live queue information. Predictions and queue positions are estimates for an academic project.
