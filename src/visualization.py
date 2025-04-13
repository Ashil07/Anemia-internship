"""
visualization.py - Functions for data visualization

This file contains functions for creating visualizations of the anemia dataset,
which can be used both in the exploratory analysis and in the web application.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os

def create_anemia_distribution_pie(df, output_dir=None):
    """
    Create a pie chart showing the distribution of anemia cases
    
    Args:
        df: DataFrame with anemia data
        output_dir: Directory to save the plot
        
    Returns:
        fig: Plotly figure object
    """
    # Create a copy for visualization
    df_viz = df.copy()
    if 'Decision_Class' in df_viz.columns:
        df_viz['Diagnosis'] = df_viz['Decision_Class'].replace({1: 'Anemic', 0: 'Non-Anemic'})
    
    # Create diagnosis pie chart
    diagnosis_counts = df_viz['Diagnosis'].value_counts().reset_index()
    diagnosis_counts.columns = ['Diagnosis', 'Count']
    
    fig = px.pie(
        diagnosis_counts, 
        names='Diagnosis', 
        values='Count', 
        title='Anemia Status Distribution', 
        color_discrete_sequence=['#FF9999', '#66B2FF']
    )
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.write_image(os.path.join(output_dir, 'anemia_distribution_pie.png'))
    
    return fig

def create_gender_distribution_plots(df, output_dir=None):
    """
    Create plots showing gender distribution and anemia by gender
    
    Args:
        df: DataFrame with anemia data
        output_dir: Directory to save the plots
        
    Returns:
        figs: Dictionary of Plotly figure objects
    """
    # Create a copy for visualization
    df_viz = df.copy()
    
    # Ensure proper formatting of columns
    if 'Gender' in df_viz.columns:
        if df_viz['Gender'].isin(['f', 'm']).all():
            df_viz['Gender'] = df_viz['Gender'].replace({'f': 'Female', 'm': 'Male'})
    
    if 'Decision_Class' in df_viz.columns:
        df_viz['Diagnosis'] = df_viz['Decision_Class'].replace({1: 'Anemic', 0: 'Non-Anemic'})
    
    # Gender pie chart
    gender_counts = df_viz['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    
    fig_gender_pie = px.pie(
        gender_counts, 
        names='Gender', 
        values='Count', 
        title='Gender Distribution', 
        color_discrete_sequence=['#FFB6C1', '#ADD8E6']
    )
    
    # Gender by diagnosis bar chart
    gender_diagnosis = pd.crosstab(df_viz['Gender'], df_viz['Diagnosis'])
    
    fig_gender_diagnosis = px.bar(
        gender_diagnosis, 
        title="Anemia Status by Gender", 
        color_discrete_sequence=['#FF9999', '#66B2FF'],
        barmode='group'
    )
    
    # Save figures if output directory is provided
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig_gender_pie.write_image(os.path.join(output_dir, 'gender_distribution_pie.png'))
        fig_gender_diagnosis.write_image(os.path.join(output_dir, 'anemia_by_gender_bar.png'))
    
    return {
        'gender_pie': fig_gender_pie,
        'gender_diagnosis_bar': fig_gender_diagnosis
    }

def create_age_distribution_plots(df, output_dir=None):
    """
    Create plots showing age distribution and anemia by age
    
    Args:
        df: DataFrame with anemia data
        output_dir: Directory to save the plots
        
    Returns:
        figs: Dictionary of Plotly figure objects
    """
    # Create a copy for visualization
    df_viz = df.copy()
    
    # Ensure proper formatting of columns
    if 'Decision_Class' in df_viz.columns:
        df_viz['Diagnosis'] = df_viz['Decision_Class'].replace({1: 'Anemic', 0: 'Non-Anemic'})
    
    # Age histogram
    fig_age = px.histogram(
        df_viz, 
        x='Age', 
        color='Diagnosis', 
        nbins=20,
        title='Age Distribution by Anemia Status',
        color_discrete_sequence=['#FF9999', '#66B2FF']
    )
    
    # Create age groups
    bins = [0, 18, 35, 50, 65, 100]
    labels = ['<18', '18-35', '36-50', '51-65', '>65']
    df_viz['Age_Group'] = pd.cut(df_viz['Age'], bins=bins, labels=labels, right=False)
    
    # Anemia by age group
    anemia_by_age = pd.crosstab(df_viz['Age_Group'], df_viz['Diagnosis'], normalize='index') * 100
    anemia_by_age = anemia_by_age.reset_index()
    anemia_by_age = pd.melt(anemia_by_age, id_vars=['Age_Group'], var_name='Diagnosis', value_name='Percentage')
    
    fig_age_anemia = px.bar(
        anemia_by_age, 
        x='Age_Group', 
        y='Percentage', 
        color='Diagnosis',
        title='Anemia Prevalence by Age Group (%)',
        color_discrete_sequence=['#FF9999', '#66B2FF']
    )
    
    # Save figures if output directory is provided
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig_age.write_image(os.path.join(output_dir, 'age_distribution_histogram.png'))
        fig_age_anemia.write_image(os.path.join(output_dir, 'anemia_by_age_bar.png'))
    
    return {
        'age_histogram': fig_age,
        'age_anemia_bar': fig_age_anemia
    }

def create_hematological_parameter_plots(df, parameter, output_dir=None):
    """
    Create plots for a specific hematological parameter
    
    Args:
        df: DataFrame with anemia data
        parameter: Name of the hematological parameter to visualize
        output_dir: Directory to save the plots
        
    Returns:
        figs: Dictionary of Plotly figure objects
    """
    # Create a copy for visualization
    df_viz = df.copy()
    
    # Ensure proper formatting of columns
    if 'Gender' in df_viz.columns:
        if df_viz['Gender'].isin(['f', 'm']).all():
            df_viz['Gender'] = df_viz['Gender'].replace({'f': 'Female', 'm': 'Male'})
    
    if 'Decision_Class' in df_viz.columns:
        df_viz['Diagnosis'] = df_viz['Decision_Class'].replace({1: 'Anemic', 0: 'Non-Anemic'})
    
    # Histogram with box plot
    fig_hist = px.histogram(
        df_viz, 
        x=parameter, 
        color='Diagnosis', 
        title=f'Distribution of {parameter} by Anemia Status',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        marginal='box'
    )
    
    # Box plot by gender and diagnosis
    fig_box = px.box(
        df_viz, 
        x='Gender', 
        y=parameter, 
        color='Diagnosis',
        title=f'{parameter} by Gender and Anemia Status',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Save figures if output directory is provided
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig_hist.write_image(os.path.join(output_dir, f'{parameter}_histogram.png'))
        fig_box.write_image(os.path.join(output_dir, f'{parameter}_boxplot.png'))
    
    return {
        'histogram': fig_hist,
        'boxplot': fig_box
    }

def create_correlation_heatmap(df, output_dir=None):
    """
    Create a correlation heatmap for hematological parameters
    
    Args:
        df: DataFrame with anemia data
        output_dir: Directory to save the plot
        
    Returns:
        fig: Plotly figure object
    """
    # Create a copy and drop non-numeric columns
    df_corr = df.copy()
    if 'Gender' in df_corr.columns:
        df_corr = df_corr.drop('Gender', axis=1)
    
    # Calculate correlation
    corr = df_corr.corr().round(2)
    
    # Create heatmap
    fig = px.imshow(
        corr, 
        text_auto=True, 
        aspect="auto", 
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap of Hematological Parameters"
    )
    
    # Save figure if output directory is provided
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.write_image(os.path.join(output_dir, 'correlation_heatmap.png'))
    
    return fig

def create_prediction_gauge(probability, prediction, title="Probability of Anemia"):
    """
    Create a gauge chart for displaying prediction probability
    
    Args:
        probability: Prediction probability value (0-1)
        prediction: Binary prediction (0 or 1)
        title: Title for the gauge chart
        
    Returns:
        fig: Plotly figure object
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.7], 'color': "lightyellow"},
                {'range': [0.7, 1], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability
            }
        }
    ))
    
    return fig

def create_feature_comparison_radar(user_input, normal_ranges):
    """
    Create a radar chart comparing user input to normal ranges
    
    Args:
        user_input: Dictionary of user input values
        normal_ranges: Dictionary of normal ranges for parameters
        
    Returns:
        fig: Plotly figure object
    """
    # Prepare data for radar chart
    parameters = []
    user_values = []
    lower_bounds = []
    upper_bounds = []
    
    gender = 'f' if user_input.get('Gender_Encoded', user_input.get('Gender', 0)) == 1 else 'm'
    
    for param, ranges in normal_ranges.items():
        if param in user_input:
            parameters.append(param)
            user_values.append(user_input[param])
            
            if 'all' in ranges:
                lower_bounds.append(ranges['all'][0])
                upper_bounds.append(ranges['all'][1])
            else:
                lower_bounds.append(ranges[gender][0])
                upper_bounds.append(ranges[gender][1])
    
    # Create radar chart
    fig = go.Figure()
    
    # Add user values
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=parameters,
        fill='toself',
        name='User Values',
        line_color='red'
    ))
    
    # Add normal range (as a band)
    fig.add_trace(go.Scatterpolar(
        r=upper_bounds,
        theta=parameters,
        fill=None,
        name='Upper Normal',
        line_color='green'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=lower_bounds,
        theta=parameters,
        fill='tonext',  # Fill area between this trace and the next one
        name='Lower Normal',
        line_color='green'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
            ),
        ),
        showlegend=True,
        title="Parameter Comparison with Normal Ranges"
    )
    
    return fig