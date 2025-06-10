import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
users = pd.read_csv("users.csv", parse_dates=["date_diagnosis"])
users["infected"] = users["infected"].str.lower().str.strip() == "yes"
users.rename(columns={"user_id": "Id"}, inplace=True)

contacts = pd.read_csv("contacts.csv", parse_dates=["contact_start", "contact_end"])
contacts["duration"] = (contacts["contact_end"] - contacts["contact_start"]).dt.total_seconds() / 60

# Build full directed graph
G = nx.DiGraph()
for _, row in contacts.iterrows():
    G.add_edge(row["reporting_user"], row["contact_user"], start=row["contact_start"], end=row["contact_end"])

# Assign infection iteration labels
infection_iteration = {}

# Iteration 0: initially infected
initially_infected = set(users[users["infected"]]["Id"])
for uid in initially_infected:
    infection_iteration[uid] = 0

# Iteration 1: directly infected by initially infected
iteration_1 = set()
for uid in initially_infected:
    iteration_1.update(G.successors(uid))
for uid in iteration_1:
    if uid not in infection_iteration:
        infection_iteration[uid] = 1

# Iteration 2: infected by iteration 1
iteration_2 = set()
for uid in iteration_1:
    iteration_2.update(G.successors(uid))
for uid in iteration_2:
    if uid not in infection_iteration:
        infection_iteration[uid] = 2

# Create subgraph of all users with an infection iteration
infected_subgraph_nodes = list(infection_iteration.keys())
H = G.subgraph(infected_subgraph_nodes).copy()

# Set node colors based on iteration
palette = sns.color_palette("Set2", 3)
color_map = {0: palette[0], 1: palette[1], 2: palette[2]}
node_colors = [color_map[infection_iteration[n]] for n in H.nodes]

# Draw the graph
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(H, seed=42)
nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=600, alpha=0.9)
nx.draw_networkx_edges(H, pos, edge_color='gray', arrows=True, arrowstyle='->')
nx.draw_networkx_labels(H, pos, labels={n: str(n) for n in H.nodes}, font_size=10)

plt.title("COVID Transmission Graph by Infection Iteration (User IDs)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()
