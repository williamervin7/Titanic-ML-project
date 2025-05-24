# Titanic - Machine Learning from Disaster

This project tackles the classic Titanic dataset to predict passenger survival using supervised learning models. The focus was on **feature engineering, model tuning, and ensembling**, with clean documentation and reproducible code.

## 🔍 Project Overview

The goal is to build a classification model that predicts whether a passenger survived the Titanic disaster based on features like age, class, fare, and more.

- **Source**: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/)
- **Target**: `Survived` (0 = No, 1 = Yes)
- **Metric**: Accuracy

## 🛠️ Features Created

- `FamilySize` = `SibSp` + `Parch` + 1  
- `FarePerPerson` = `Fare` / `FamilySize`  
- `Title` extracted from `Name`  
- `Deck` extracted from `Cabin`  
- One-hot encoded categorical variables: `Sex`, `Embarked`, `Pclass`, `Title`, and `Deck`

Missing values were filled using grouped medians and modes. Train and test features were aligned to avoid column mismatch.

## 🤖 Models Used

Each model was evaluated using 5-fold cross-validation with accuracy:

- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `XGBClassifier`
- `LGBMClassifier` (tuned with `GridSearchCV`)
- `LogisticRegression`
- `VotingClassifier` (ensemble of tuned XGBoost and LightGBM)

## 🧪 Results

| Model                  | CV Accuracy (approx) |
|------------------------|----------------------|
| Random Forest          | ~0.80                |
| Gradient Boosting      | ~0.81                |
| LightGBM (tuned)       | ~0.83                |
| XGBoost (tuned)        | ~0.83                |
| **Ensemble (LGBM + XGB)** | **~0.837**         |

- **Best Kaggle Submission Accuracy:** `0.77751`

## 🗂️ Project Structure

- `notebook.ipynb` — Full workflow: cleaning, feature engineering, modeling, and submission
- `ensemble_submission.csv` — Final submission created using ensemble predictions
- `README.md` — Project summary and documentation

## 📌 Key Takeaways

- High cross-validation accuracy doesn't guarantee leaderboard gains — generalization matters
- Feature engineering is powerful, but must be validated with care
- Ensembling well-tuned models can improve robustness and performance
- Proper train/test feature alignment is essential to avoid prediction errors

## ✅ Next Steps

Possible extensions:

- Try a neural network (`MLPClassifier`)
- Explore more advanced ensembling (e.g., stacking)
- Apply this workflow to a larger or messier real-world dataset
