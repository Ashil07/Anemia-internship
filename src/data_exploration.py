"""
data_exploration.py - Exploratory Data Analysis for anemia prediction

This script performs exploratory data analysis on the anemia dataset,
creating visualizations and statistical summaries to understand the data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Add parent directory to path to allow importing from config/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.feature_config import NORMAL_RANGES

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def load_data(filepath):
    """Load the anemia dataset and perform basic preprocessing"""
    try:
        df = pd.read_excel(filepath)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(df):
    """Preprocess data for visualization and analysis"""
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert Gender to binary for modeling purposes
    le = LabelEncoder()
    df_processed['Gender_Encoded'] = le.fit_transform(df_processed['Gender'])
    # Map 'f' -> 1 and 'm' -> 0 for easier interpretation
    gender_mapping = {'f': 1, 'm': 0}
    df_processed['Gender_Encoded'] = df_processed['Gender'].map(gender_mapping)
    
    # Create a copy for visualization
    df_viz = df_processed.copy()
    df_viz['Gender'] = df_viz['Gender'].replace({'f': 'Female', 'm': 'Male'})
    df_viz['Decision_Class'] = df_viz['Decision_Class'].replace({1: 'Anemic', 0: 'Non-Anemic'})
    
    return df_processed, df_viz

def print_dataset_summary(df):
    """Print summary statistics about the dataset"""
    print("Dataset Summary:")
    print(f"Total observations: {df.shape[0]}")
    print(f"Anemic cases: {df[df['Decision_Class'] == 1].shape[0]} ({df[df['Decision_Class'] == 1].shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Non-anemic cases: {df[df['Decision_Class'] == 0].shape[0]} ({df[df['Decision_Class'] == 0].shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Female patients: {df[df['Gender'] == 'f'].shape[0]} ({df[df['Gender'] == 'f'].shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Male patients: {df[df['Gender'] == 'm'].shape[0]} ({df[df['Gender'] == 'm'].shape[0]/df.shape[0]*100:.1f}%)")
    
    # Distribution of anemia by gender
    gender_anemia = pd.crosstab(df['Gender'], df['Decision_Class'], normalize='index') * 100
    print("\nPercentage of Anemia by Gender:")
    print(gender_anemia)
    
    # Descriptive statistics for numerical features
    print("\nDescriptive Statistics for Each Feature:")
    print(df.describe().round(2))

def plot_hematological_parameters(df_viz, output_dir=None):
    """Create histograms for all hematological parameters"""
    features = ['Hb', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.histplot(data=df_viz, x=feature, hue='Decision_Class', kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature} by Anemia Status')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'hematological_parameters_distribution.png'), dpi=300)
    plt.show()

def plot_violin_by_anemia(df_viz, output_dir=None):
    """Create violin plots comparing distributions by anemia status"""
    features = ['Hb', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.violinplot(x='Decision_Class', y=feature, data=df_viz, ax=axes[i])
        axes[i].set_title(f'{feature} by Anemia Status')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'violin_plots_by_anemia.png'), dpi=300)
    plt.show()

def plot_boxplots_by_gender(df_viz, output_dir=None):
    """Create comparison of parameters by gender"""
    features = ['Hb', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.boxplot(x='Gender', y=feature, hue='Decision_Class', data=df_viz, ax=axes[i])
        axes[i].set_title(f'{feature} by Gender and Anemia Status')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'boxplots_by_gender.png'), dpi=300)
    plt.show()

def analyze_age_groups(df_viz, output_dir=None):
    """Analyze anemia by age groups"""
    # Create age groups
    bins = [0, 18, 35, 50, 65, 100]
    labels = ['<18', '18-35', '36-50', '51-65', '>65']
    df_viz['Age_Group'] = pd.cut(df_viz['Age'], bins=bins, labels=labels, right=False)
    
    # Plot anemia prevalence by age group
    plt.figure(figsize=(12, 6))
    anemia_by_age = pd.crosstab(df_viz['Age_Group'], df_viz['Decision_Class'], normalize='index') * 100
    anemia_by_age['Anemic'].plot(kind='bar', color='coral')
    plt.title('Anemia Prevalence by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Percentage (%)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'anemia_by_age_group.png'), dpi=300)
    plt.show()
    
    # Heatmap of age group vs gender for anemia
    plt.figure(figsize=(12, 8))
    heatmap_data = pd.crosstab([df_viz['Age_Group'], df_viz['Gender']], df_viz['Decision_Class'], 
                              normalize='index')['Anemic'] * 100
    heatmap_data = heatmap_data.unstack(level=1)
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.1f')
    plt.title('Anemia Prevalence (%) by Age Group and Gender')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'anemia_heatmap_age_gender.png'), dpi=300)
    plt.show()

def plot_correlation_matrix(df, output_dir=None):
    """Create and plot correlation matrix"""
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.drop(['Gender'], axis=1).corr()
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
    plt.title('Correlation Matrix of Anemia Indicators')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
    plt.show()
    
    # Calculate the correlation of each feature with the target
    correlation_with_target = correlation_matrix['Decision_Class'].sort_values(ascending=False)
    print("\nCorrelation of Features with Anemia Status:")
    print(correlation_with_target)

def analyze_normal_ranges(df):
    """Analyze parameters relative to normal clinical ranges"""
    print("\nAnalysis of Parameters Relative to Normal Clinical Ranges:")
    results = {}
    
    for param, ranges in NORMAL_RANGES.items():
        if 'all' in ranges:
            min_val, max_val = ranges['all']
            below_normal = df[df[param] < min_val].shape[0]
            above_normal = df[df[param] > max_val].shape[0]
            normal = df[(df[param] >= min_val) & (df[param] <= max_val)].shape[0]
            
            results[param] = {
                'Below Normal': below_normal,
                'Normal': normal,
                'Above Normal': above_normal,
                'Below Normal %': below_normal / len(df) * 100,
                'Normal %': normal / len(df) * 100,
                'Above Normal %': above_normal / len(df) * 100
            }
        else:
            results[param] = {'Gender': [], 'Below Normal': [], 'Normal': [], 'Above Normal': []}
            
            for gender, gender_range in ranges.items():
                min_val, max_val = gender_range
                gender_df = df[df['Gender'] == gender]
                
                below_normal = gender_df[gender_df[param] < min_val].shape[0]
                above_normal = gender_df[gender_df[param] > max_val].shape[0]
                normal = gender_df[(gender_df[param] >= min_val) & (gender_df[param] <= max_val)].shape[0]
                
                results[param]['Gender'].append(gender)
                results[param]['Below Normal'].append(below_normal)
                results[param]['Normal'].append(normal)
                results[param]['Above Normal'].append(above_normal)
    
    for param, data in results.items():
        if 'Gender' in data:
            print(f"\n{param} Analysis by Gender:")
            for i, gender in enumerate(data['Gender']):
                total = data['Below Normal'][i] + data['Normal'][i] + data['Above Normal'][i]
                print(f"  {gender.upper()}: Below Normal: {data['Below Normal'][i]} ({data['Below Normal'][i]/total*100:.1f}%), "
                      f"Normal: {data['Normal'][i]} ({data['Normal'][i]/total*100:.1f}%), "
                      f"Above Normal: {data['Above Normal'][i]} ({data['Above Normal'][i]/total*100:.1f}%)")
        else:
            total = data['Below Normal'] + data['Normal'] + data['Above Normal']
            print(f"\n{param} Analysis (All Genders):")
            print(f"  Below Normal: {data['Below Normal']} ({data['Below Normal %']:.1f}%), "
                  f"Normal: {data['Normal']} ({data['Normal %']:.1f}%), "
                  f"Above Normal: {data['Above Normal']} ({data['Above Normal %']:.1f}%)")

def plot_scatter_matrix(df_viz, output_dir=None):
    """Create scatter plots to identify relationships"""
    features = ['Hb', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC']
    
    # Create pairplot
    plt.figure(figsize=(15, 15))
    pairplot = sns.pairplot(df_viz, vars=features, hue='Decision_Class', 
                           palette={'Anemic': 'coral', 'Non-Anemic': 'teal'},
                           diag_kind='kde', plot_kws={'alpha': 0.6})
    pairplot.fig.suptitle('Relationships Between Hematological Parameters', y=1.02, fontsize=16)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'scatter_matrix.png'), dpi=300)
    plt.show()

def run_exploratory_analysis(data_path, output_dir=None):
    """Run the full exploratory data analysis workflow"""
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load and preprocess data
    df = load_data(data_path)
    if df is None:
        return
    
    df_processed, df_viz = preprocess_data(df)
    
    # Print dataset summary
    print_dataset_summary(df)
    
    # Generate visualizations
    print("\nGenerating visualizations for hematological parameters...")
    plot_hematological_parameters(df_viz, output_dir)
    plot_violin_by_anemia(df_viz, output_dir)
    plot_boxplots_by_gender(df_viz, output_dir)
    analyze_age_groups(df_viz, output_dir)
    plot_correlation_matrix(df_processed, output_dir)
    analyze_normal_ranges(df)
    plot_scatter_matrix(df_viz, output_dir)
    
    print("\nExploratory Data Analysis Complete!")
    return df_processed

if __name__ == "__main__":
    # Example usage
    data_path = "./data/Anemia Dataset.xlsx"
    output_dir = "../output/figures"
    run_exploratory_analysis(data_path, output_dir)