import pandas as pd
import matplotlib.pyplot as plt

# 1) Load the data (adjust path if needed)
df1 = pd.read_csv('LS1.csv')
df2 = pd.read_csv('LS2.csv')

# 2) If there’s an extra unnamed first column, drop it:
# df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# 3) Extract the two columns we need
x1 = df1['XDistanceToStartpoint']
y1 = df1['Hardness']

x2 = df2['XDistanceToStartpoint']
x2 = x2 + 26.658
y2 = df2['Hardness']

x = pd.concat([x1, x2], ignore_index=True)
y = pd.concat([y1, y2], ignore_index=True)

# 4) Make the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel('X Distance to Startpoint')
plt.ylabel('Hardness')
plt.title('Hardness vs. X Distance (LS1 and LS2)')
plt.grid(True)
plt.tight_layout()

# 5) Save to disk (will create 'hardness_vs_xdistance.png' in cwd)
plt.savefig('hardness_vs_xdistance_combined.png', dpi=300)
