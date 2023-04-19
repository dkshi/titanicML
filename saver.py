import pandas as pd


def save(path_name, test, predicts):
    gender_submission = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": predicts})
    gender_submission.to_csv(path_name, index=False)
    return path_name
