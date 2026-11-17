# LoanCheck — Model Card

Current PSI warnings for ApplicantIncome and CoapplicantIncome are expected for the static Kaggle train/test split because income is skewed and checked with a small bin count; this is not a bug to fix.

## Model comparison

Logistic regression provides an interpretable linear baseline. A decision tree can learn simple non-linear rules but is prone to overfitting, while a random forest reduces that variance through an ensemble at the cost of interpretability and complexity.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.8618 | 0.8400 | 0.9882 | 0.9081 | 0.8601 |
| Decision tree | 0.7236 | 0.8493 | 0.7294 | 0.7848 | 0.7200 |
| Random forest | 0.8537 | 0.8681 | 0.9294 | 0.8977 | 0.8605 |

Logistic regression was selected as the current champion because it has the highest accuracy (0.8618) while exceeding both development quality bars. Its recall and F1 score are also the strongest of the candidates, and its simpler, interpretable form makes it a suitable baseline for this project.

Champion promoted to version 2 manually — tied version 1 on all metrics but was trained from a clean (non-dirty) git commit, which the automated promotion rule doesn't account for.
