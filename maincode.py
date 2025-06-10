import pandas as pd
import networkx as nx

# --- Load data ---
users = pd.read_csv("users.csv", parse_dates=["date_diagnosis"])
users["infected"] = users["infected"].str.lower().str.strip() == "yes"
users.rename(columns={"user_id": "Id"}, inplace=True)

contacts = pd.read_csv("contacts.csv", parse_dates=["contact_start", "contact_end"])
contacts["duration"] = (contacts["contact_end"] - contacts["contact_start"]).dt.total_seconds() / 60

# --- Build the complete directed graph ---
G = nx.DiGraph()

# Add nodes  attributes
for _, row in withusers.iterrows():
    G.add_node(row["Id"], infected=row["infected"], label=f"{row['first_name']} {row['last_name']}")

# Add edges from contacts
for _, row in contacts.iterrows():
    G.add_edge(row["reporting_user"], row["contact_user"], duration=row["duration"],
               start=row["contact_start"], end=row["contact_end"])

# --- Identify Initially Infected Users ---
initial_infected_users = users[users["infected"]][["Id", "first_name", "last_name", "date_diagnosis"]]
initial_infected_users.to_csv("initially_infected_users.csv", index=False)

# --- Centrality Measures ---
out_deg = nx.out_degree_centrality(G)
btw = nx.betweenness_centrality(G)

nx.set_node_attributes(G, out_deg, "out_degree")
nx.set_node_attributes(G, btw, "betweenness")

# Combine centrality into one DataFrame
centrality_df = pd.DataFrame({
    "Id": list(G.nodes),
    "Label": [G.nodes[n].get("label", "") for n in G.nodes],
    "OutDegree": [out_deg.get(n, 0) for n in G.nodes],
    "Betweenness": [btw.get(n, 0) for n in G.nodes]
})

# Sort by Betweenness and OutDegree for super spreaders
centrality_df = centrality_df.sort_values(by=["Betweenness", "OutDegree"], ascending=False)
centrality_df.head(20).to_csv("super_spreaders.csv", index=False)

# --- Identify Secondary Infected Users ---
# A secondary infected user is someone contacted by an initially infected user
initial_infected_ids = set(initial_infected_users["Id"])
secondary_infected_ids = set()

for _, row in contacts.iterrows():
    if row["reporting_user"] in initial_infected_ids:
        secondary_infected_ids.add(row["contact_user"])

# Remove already initially infected
secondary_infected_ids = secondary_infected_ids - initial_infected_ids

secondary_infected_users = users[users["Id"].isin(secondary_infected_ids)][["Id", "first_name", "last_name", "date_diagnosis"]]
secondary_infected_users.to_csv("secondary_infected_users.csv", index=False)

print("Analysis complete. Files generated:")
print(" - initially_infected_users.csv")
print(" - super_spreaders.csv")
print(" - secondary_infected_users.csv")
