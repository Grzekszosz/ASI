import pickle

with open("/home/grzegorz/Workspace/ASI/Final/final/data/08_reporting/evaluation_report.pkl", "rb") as f:
    report = pickle.load(f)

print(report)
