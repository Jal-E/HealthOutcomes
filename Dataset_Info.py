import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
import os
from io import StringIO

# Load the dataset
df = pd.read_csv('../HealthOutcome/ObesityDataSet_raw_and_data_sinthetic.csv')

# Function to save plots
def save_plot(plt, filename):
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust spacing between plots
    plt.savefig(filename)
    plt.close()
    return f'<img src="{filename}" width="800">'

# Capture the output of df.info()
buffer = StringIO()
df.info(buf=buffer)
data_info = buffer.getvalue()

# Data Descriptions
description = df.describe(include='all').to_html()
head = df.head().to_html()

# Histograms for numeric features
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(20, 15))  # Increased figure size
df[numeric_columns].hist(bins=20, color='skyblue', edgecolor='black')
plt.suptitle('Numeric Features Distribution', fontsize=16)  # Optional: add a suptitle for the figure
histogram = save_plot(plt, '../HealthOutcome/histograms.png')

# Box plots for numeric features
plt.figure(figsize=(20, 15))  # Increased figure size
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df, y=col, color='lightgreen')
    plt.title(f'Box Plot of {col}', fontsize=14)
boxplot = save_plot(plt, '../HealthOutcome/boxplots.png')

# Bar plots for categorical features
categorical_columns = df.select_dtypes(include=['object', 'bool']).columns
plt.figure(figsize=(20, 15))  # Increased figure size
for i, col in enumerate(categorical_columns):
    plt.subplot(3, 3, i+1)
    sns.countplot(data=df, x=col, color='lightblue', order=df[col].value_counts().index)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)  # Adjust font size for readability
barplot = save_plot(plt, '../HealthOutcome/barplots.png')

# Correlation heatmap
plt.figure(figsize=(12, 10))  # Adjusted size for better visibility
sns.heatmap(df[numeric_columns].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap', fontsize=16)
heatmap = save_plot(plt, '../HealthOutcome/heatmap.png')

# Ensure the current directory is correct
current_directory = os.getcwd()
env = Environment(loader=FileSystemLoader(current_directory))

# Load the template and render HTML
try:
    template = env.get_template('report_template.html')
    html_content = template.render(data_info=data_info, description=description, head=head,
                                   histogram=histogram, boxplot=boxplot, barplot=barplot, heatmap=heatmap)

    # Save to HTML file
    with open('../HealthOutcome/report.html', 'w') as f:
        f.write(html_content)
except Exception as e:
    print("Error during HTML generation:", e)
