import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Define the data ---
data = [
    ['base', '224-resize', 2.47e8, 8.28e10, 0.148, 55.28, 1686.48],
    ['base', '224-crop',   2.47e8, 8.28e10, 0.101, 56.8,  1838],
    ['base', '300-resize', 2.47e8, 9.55e10, 0.1729, 56.8,  1692.7],
    ['base', '300-crop',   2.47e8, 9.55e10, 0.163, 54.16, 1509.7],
    ['base', '348-resize', 2.47e8, 1.21e11, 0.2121, 54.43, 1649.78],
    ['base', '348-crop',   2.47e8, 1.21e11, 0.213,  61.03, 1815.61],
    ['large', '224-resize', 4.70e8, 1.27e11, 0.1257, 215.5,  3907.2],
    ['large', '224-crop',   4.70e8, 1.27e11, 0.2227, 190.09, 3651.37],
    ['large', '300-resize', 4.70e8, 1.68e11, 0.153,  150.1,  3640.39],
    ['large', '300-crop',   4.70e8, 1.68e11, 0.1428, 265.12, 4386.51],
    ['large', '348-resize', 4.70e8, 2.05e11, 0.2072, 280.52, 6133.04],
    ['large', '348-crop',   4.70e8, 2.05e11, 0.1331, 145.37, 4322.46],
]

columns = ['Model', 'Input Size', 'Params', 'FLOPs', 'BLEU', 'Latency (ms)', 'Load (ms)']
df = pd.DataFrame(data, columns=columns)

# --- 2. Extract 'Input Dim' and 'Transform' from 'Input Size' ---
df[['Input Dim', 'Transform']] = df['Input Size'].str.extract(r'(\d+)-(resize|crop)')
df['Input Dim'] = df['Input Dim'].astype(int)

# --- 3. Create subplots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Colors for consistency
colors = {'base': 'blue', 'large': 'orange'}

# --- 4. Plot for Resize ---
ax = axes[0]
resize_df = df[df['Transform'] == 'resize']
for model in ['base', 'large']:
    model_df = resize_df[resize_df['Model'] == model]
    ax.plot(model_df['Input Dim'], model_df['BLEU'], label=model.capitalize(), 
            marker='o', color=colors[model])
ax.set_title('BLEU vs Input Size (Resize)')
ax.set_xlabel('Input Size (px)')
ax.set_ylabel('BLEU Score')
ax.grid(True)
ax.legend()

# --- 5. Plot for Crop ---
ax = axes[1]
crop_df = df[df['Transform'] == 'crop']
for model in ['base', 'large']:
    model_df = crop_df[crop_df['Model'] == model]
    ax.plot(model_df['Input Dim'], model_df['BLEU'], label=model.capitalize(), 
            marker='o', color=colors[model])
ax.set_title('BLEU vs Input Size (Crop)')
ax.set_xlabel('Input Size (px)')
ax.grid(True)
ax.legend()

# --- 6. Final layout ---
# plt.suptitle('Model BLEU Scores by Input Size', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
# plt.show()

# --- 7. Save the figure ---
fig.savefig('bleu_vs_input_size.png', dpi=300, bbox_inches='tight')

# --- 8. Repeat for Latency vs Input Size ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
# Plot for Resize
ax = axes[0]
resize_df = df[df['Transform'] == 'resize']
for model in ['base', 'large']:
    model_df = resize_df[resize_df['Model'] == model]
    ax.plot(model_df['Input Dim'], model_df['Latency (ms)'], label=model.capitalize(), 
            marker='o', color=colors[model])
ax.set_title('Latency vs Input Size (Resize)')
ax.set_xlabel('Input Size (px)')
ax.set_ylabel('Latency (ms)')
ax.grid(True)
ax.legend()

# Plot for Crop
ax = axes[1]
crop_df = df[df['Transform'] == 'crop']
for model in ['base', 'large']:
    model_df = crop_df[crop_df['Model'] == model]
    ax.plot(model_df['Input Dim'], model_df['Latency (ms)'], label=model.capitalize(), 
            marker='o', color=colors[model])
ax.set_title('Latency vs Input Size (Crop)')
ax.set_xlabel('Input Size (px)')
ax.grid(True)
ax.legend()

# Final layout
# plt.suptitle('Model Latency by Input Size', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
# plt.show()

# Save the figure
fig.savefig('latency_vs_input_size.png', dpi=300, bbox_inches='tight')
