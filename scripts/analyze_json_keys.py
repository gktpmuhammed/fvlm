import os
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------
# STEP 1: Load the Data
# ---------------------------------------------------------
input_filename = str(PROJECT_ROOT / 'data/combined_desc_conc.json')

try:
    with open(input_filename, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: {input_filename} not found.")
    exit()

# ---------------------------------------------------------
# STEP 2: Count Key Occurrences
# ---------------------------------------------------------
key_counts = {}

# Iterate through each patient
for patient_id, content in data.items():
    # Iterate through keys in that patient's record
    for key in content.keys():
        key_counts[key] = key_counts.get(key, 0) + 1

# Convert dictionary to Pandas DataFrame
df = pd.DataFrame(list(key_counts.items()), columns=['Key', 'Count'])

# Sort by count descending for better visualization
df = df.sort_values(by='Count', ascending=False)

# ---------------------------------------------------------
# STEP 3: Save to CSV
# ---------------------------------------------------------
csv_filename = 'key_occurrences.csv'
df.to_csv(csv_filename, index=False)
print(f"Successfully saved CSV to '{csv_filename}'")

# ---------------------------------------------------------
# STEP 4: Generate Plot
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# Create bar chart
bars = plt.bar(df['Key'], df['Count'], color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Keys', fontsize=12)
plt.ylabel('Frequency (Number of Patients)', fontsize=12)
plt.title('Occurrence of Keys in Patient JSON Data', fontsize=14)
plt.xticks(rotation=45, ha='right')  # Rotate labels for readability

# Add count numbers on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), 
             va='bottom', ha='center', fontweight='bold')

plt.tight_layout()

# Save and Show
plot_filename = 'key_occurrences_plot.png'
plt.savefig(plot_filename)
print(f"Successfully saved plot to '{plot_filename}'")
plt.show()