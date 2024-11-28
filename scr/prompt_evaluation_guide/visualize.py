import json

import matplotlib.pyplot as plt
import seaborn as sns

# Example metric scores
# metrics = {"BLUE": 0.85, "ROGUE-L": 0.87, "Perplexity": 15.2}

with open("data/metrics.json", "r") as file:
    metrics = json.load(file)

#Plot

plt.figure(figsize=(8,5))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
plt.title("Evaluation metrics")
plt.ylabel("Scores")
plt.xlabel("Metric")
plt.show()