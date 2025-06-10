import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error

# --- Load Output Files ---
users = pd.read_csv("users.csv")
users["infected"] = users["infected"].str.lower().str.strip() == "yes"
users.rename(columns={"user_id": "Id"}, inplace=True)

contacts = pd.read_csv("contacts.csv", parse_dates=["contact_start", "contact_end"])
contacts["duration"] = (contacts["contact_end"] - contacts["contact_start"]).dt.total_seconds() / 60

initial_infected = pd.read_csv("initially_infected_users.csv")
super_spreaders = pd.read_csv("super_spreaders.csv")
secondary_infected = pd.read_csv("secondary_infected_users.csv")

# --- Rebuild Graph ---
G = nx.DiGraph()
for _, row in users.iterrows():
    G.add_node(row["Id"], infected=row["infected"])
for _, row in contacts.iterrows():
    G.add_edge(row["reporting_user"], row["contact_user"])

# --- Evaluation Metrics ---
metrics = {}

# 1. Network Structure Analysis
metrics["Network Density"] = nx.density(G)
try:
    metrics["Average Path Length"] = nx.average_shortest_path_length(G)
except:
    metrics["Average Path Length"] = np.nan
metrics["Clustering Coefficient"] = nx.average_clustering(G.to_undirected())

# 2. Super Spreader Evaluation
true_labels = users.set_index("Id").loc[super_spreaders["Id"]]["infected"].astype(int)
pred_labels = pd.Series([1] * len(super_spreaders), index=super_spreaders["Id"])
metrics["Precision"] = precision_score(true_labels, pred_labels)
metrics["Recall"] = recall_score(true_labels, pred_labels)
metrics["F1 Score"] = f1_score(true_labels, pred_labels)

# 3. Containment Effectiveness
r0 = len(secondary_infected) / len(initial_infected) if len(initial_infected) > 0 else np.nan
metrics["Reproduction Number R0"] = r0

# 4. Model Prediction Accuracy
actual = users.set_index("Id")["infected"].astype(int)
predicted = users["Id"].isin(super_spreaders["Id"]).astype(int)
metrics["Mean Squared Error"] = mean_squared_error(actual, predicted)
metrics["Correlation with Real Data"] = np.corrcoef(actual, predicted)[0, 1]

# --- Save Metrics ---
pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).to_csv("evaluation_metrics.csv", index=False)

print("Evaluation complete. Metrics saved to 'evaluation_metrics.csv'")
