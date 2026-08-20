# Customer Churn Retention Support

A Streamlit prototype that converts a trained churn prediction model into a customer retention decision-support tool.

## Main Functions

- Single-customer churn risk assessment
- Low / Medium / High operational risk bands
- Recommended retention actions
- Batch CSV scoring
- Prioritised customer retention list
- Downloadable batch results

## Required Files

Keep these files in the same GitHub repository:

```text
app.py
final_churn_model.pkl
requirements.txt
README.md
batch_customer_template.csv
```

`final_churn_model.pkl` must be generated from the final notebook after model comparison.

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push all required files to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Create a new app.
4. Select the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy.

## Important Note

The risk score is used for customer prioritisation. The prototype uses fixed operational risk bands (0.40 and 0.70). These bands should be validated against actual business costs and retention outcomes before real-world use.
