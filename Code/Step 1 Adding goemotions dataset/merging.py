import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from datasets import load_dataset
import warnings

warnings.filterwarnings('ignore')

print("1. Loading original training data...")
train_df = pd.read_csv("merged/merged_training.tsv", sep='\t')

value_columns = [
    'Self-direction: thought attained', 'Self-direction: thought constrained',
    'Self-direction: action attained', 'Self-direction: action constrained',
    'Stimulation attained', 'Stimulation constrained',
    'Hedonism attained', 'Hedonism constrained',
    'Achievement attained', 'Achievement constrained',
    'Power: dominance attained', 'Power: dominance constrained',
    'Power: resources attained', 'Power: resources constrained',
    'Face attained', 'Face constrained',
    'Security: personal attained', 'Security: personal constrained',
    'Security: societal attained', 'Security: societal constrained',
    'Tradition attained', 'Tradition constrained',
    'Conformity: rules attained', 'Conformity: rules constrained',
    'Conformity: interpersonal attained', 'Conformity: interpersonal constrained',
    'Humility attained', 'Humility constrained',
    'Benevolence: caring attained', 'Benevolence: caring constrained',
    'Benevolence: dependability attained', 'Benevolence: dependability constrained',
    'Universalism: concern attained', 'Universalism: concern constrained',
    'Universalism: nature attained', 'Universalism: nature constrained',
    'Universalism: tolerance attained', 'Universalism: tolerance constrained'
]

emotion_to_value_map = {
    "admiration": "Achievement attained",
    "anger": "Power: dominance constrained",
    "annoyance": "Power: resources constrained",
    "confusion": "Face constrained",
    "fear": "Security: personal constrained",
    "nervousness": "Security: societal constrained",
    "neutral": "Tradition attained",
    "disapproval": "Conformity: rules constrained",
    "disgust": "Conformity: interpersonal constrained",
    "embarrassment": "Humility constrained",
    "love": "Benevolence: caring attained",
    "gratitude": "Benevolence: dependability attained",
    "caring": "Universalism: concern attained",
    "optimism": "Universalism: nature attained",
    "approval": "Universalism: tolerance attained",
    "curiosity": "Self-direction: thought attained",
    "surprise": "Self-direction: action attained",
    "excitement": "Stimulation attained",
    "joy": "Hedonism attained"
}

TARGET_SAMPLES = 2000

print("2. Calculating 'Before' distributions...")
before_counts_raw = train_df[value_columns].sum()

print("3. Loading GoEmotions dataset...")
goemotions = load_dataset("go_emotions", "simplified")
goemo_df = goemotions['train'].to_pandas()
labels_list = goemotions['train'].features['labels'].feature.names

new_rows = []
goemo_id_counter = 1

print("4. Mapping emotions to values and balancing...")
for emotion, target_col in emotion_to_value_map.items():
    current_count = before_counts_raw.get(target_col, 0)
    deficit = int(TARGET_SAMPLES - current_count)
    
    if deficit > 0:
        emo_idx = labels_list.index(emotion)
        matching_texts = goemo_df[goemo_df['labels'].apply(lambda x: emo_idx in x)]['text'].tolist()
        samples_to_take = matching_texts[:deficit]
        
        for text in samples_to_take:
            new_row = {col: 0.0 for col in value_columns}
            new_row[target_col] = 1.0
            new_row['Text-ID'] = f"GOEMO_{goemo_id_counter}"
            new_row['Sentence-ID'] = 1
            new_row['Text'] = text
            new_rows.append(new_row)
            goemo_id_counter += 1

goemo_translated_df = pd.DataFrame(new_rows)
augmented_train_df = pd.concat([train_df, goemo_translated_df], ignore_index=True).fillna(0.0)

# Save the dataset
augmented_train_df.to_csv("augmented_training.tsv", sep='\t', index=False)
print(f"--> Added {len(new_rows)} new translated GoEmotion sentences.")
print("--> Saved as 'augmented_training.tsv'")

print("\n5. Generating Before & After Graphs...")

after_counts_raw = augmented_train_df[value_columns].sum()

# Aggregate 38 columns into 19 Schwartz Base Values for cleaner visualization
base_values = list(set([col.replace(' attained', '').replace(' constrained', '') for col in value_columns]))
base_values.sort()

before_counts = []
after_counts = []

for val in base_values:
    # Sum both 'attained' and 'constrained' for the base value
    b_sum = before_counts_raw[f'{val} attained'] + before_counts_raw[f'{val} constrained']
    a_sum = after_counts_raw[f'{val} attained'] + after_counts_raw[f'{val} constrained']
    before_counts.append(b_sum)
    after_counts.append(a_sum)

# Create DataFrame for plotting
df_viz = pd.DataFrame({
    'Value': base_values * 2,
    'Count': before_counts + after_counts,
    'State': ['Before Augmentation']*len(base_values) + ['After Augmentation']*len(base_values)
})

sns.set_theme(style="whitegrid")
custom_palette = ['#E74C3C', '#2ECC71']

# --- Graph 1: Grouped Bar Chart ---
plt.figure(figsize=(16, 8))
sns.barplot(data=df_viz, x='Value', y='Count', hue='State', palette=custom_palette)
plt.xticks(rotation=45, ha='right')
plt.title('1. Distribution of 19 Schwartz Values (Before vs After)', fontsize=16, pad=15)
plt.ylabel('Total Sentences')
plt.xlabel('')
plt.legend(title='')
plt.tight_layout()
plt.savefig('1_grouped_bar_chart.png', dpi=150)
plt.close()

# --- Graph 2: Stacked Bar Chart ---
added_counts = [a - b for a, b in zip(after_counts, before_counts)]
plt.figure(figsize=(16, 8))
plt.bar(base_values, before_counts, label='Original Data (ValuesML)', color='#3498DB')
plt.bar(base_values, added_counts, bottom=before_counts, label='Augmented Data (GoEmotions)', color='#F1C40F')
plt.xticks(rotation=45, ha='right')
plt.title('2. Composition of Final Balanced Dataset', fontsize=16, pad=15)
plt.ylabel('Total Sentences')
plt.legend()
plt.tight_layout()
plt.savefig('2_stacked_bar_chart.png', dpi=150)
plt.close()

# --- Graph 3: Radar Chart (Schwartz Continuum Mapping) ---
N = len(base_values)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1] 

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
# Before
b_vals = before_counts + before_counts[:1]
ax.plot(angles, b_vals, linewidth=2, linestyle='solid', label='Before Augmentation', color='#E74C3C')
ax.fill(angles, b_vals, alpha=0.25, color='#E74C3C')
# After
a_vals = after_counts + after_counts[:1]
ax.plot(angles, a_vals, linewidth=2, linestyle='solid', label='After Augmentation', color='#2ECC71')
ax.fill(angles, a_vals, alpha=0.25, color='#2ECC71')

plt.xticks(angles[:-1], base_values, size=9)
plt.title('3. Psychological Dataset Symmetry (Radar Chart)', y=1.08, fontsize=16)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))
plt.tight_layout()
plt.savefig('3_radar_chart.png', dpi=150)
plt.close()

# --- Graph 4: Slopegraph ---
plt.figure(figsize=(12, 10))
for i in range(len(base_values)):
    color = '#2ECC71' if (after_counts[i] - before_counts[i]) > 0 else '#3498DB'
    if before_counts[i] < 500: color = '#E74C3C'
    
    plt.plot([0, 1], [before_counts[i], after_counts[i]], marker='o', color=color, linewidth=2)
    plt.text(-0.05, before_counts[i], f"{base_values[i]} ({int(before_counts[i])})", ha='right', va='center', fontsize=10)

plt.xticks([0, 1], ['Before Augmentation', 'After Augmentation'], fontsize=14, fontweight='bold')
plt.title('4. Trajectory of Minority Class Rescue', fontsize=16, pad=15)
plt.xlim(-0.5, 1.5)
plt.ylim(0, max(after_counts) + 500)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().get_yaxis().set_visible(False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig('4_slope_graph.png', dpi=150)
plt.close()

# --- Graph 5: Box Plot ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_viz, x='State', y='Count', palette=custom_palette)
sns.stripplot(data=df_viz, x='State', y='Count', color='black', alpha=0.6, jitter=True, size=7)
plt.title('5. Eradication of Statistical Variance', fontsize=16, pad=15)
plt.ylabel('Distribution of Samples Across 19 Classes')
plt.xlabel('')
plt.tight_layout()
plt.savefig('5_box_plot.png', dpi=150)
plt.close()

print("--> Successfully generated 5 High-Resolution Graphs!")
print("--> Look in your Kaggle 'Output' directory for the .png files.")