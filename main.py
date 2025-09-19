import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import numpy as np
from plotly.subplots import make_subplots

# Load and prepare the data
def load_and_prepare_data():
    """Load and clean the Titanic dataset"""
    df = sns.load_dataset('titanic')
    
    # Data cleaning steps from your original code
    df = df.drop(['deck'], axis=1, errors='ignore')
    df['age'] = df['age'].fillna(df['age'].mean())
    df = df.drop(['embark_town', 'embarked'], axis=1, errors='ignore')
    
    # Create age and fare groups
    df['age_grouped'] = pd.cut(df['age'],
                              bins=[0, 12, 20, 40, 60, 80], 
                              labels=['Child', 'Teen', 'Adult', 'Middle-Aged', 'Senior'])
    
    df['fare_grouped'] = pd.cut(df['fare'],
                               bins=[0, 10, 30, 50, 100, 200, 500],
                               labels=['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extremely High'])
    
    return df

# Load the data
df = load_and_prepare_data()

# Initialize the Dash app
app = dash.Dash(__name__)

# Define insight options
insight_options = [
    {'label': '1. Overall Survival Pattern - 549 Died vs 342 Survived', 'value': 'survival_counts'},
    {'label': '2. Age Demographics - Most Passengers Were Young Adults', 'value': 'age_distribution'},
    {'label': '3. Economic Class Structure - Fare Distribution by Class', 'value': 'fare_by_class'},
    {'label': '4. Age vs Fare Relationship - Survivors Had Higher Fares', 'value': 'age_vs_fare'},
    {'label': '5. Class-Based Survival Inequality - First Class 62.9%, Third Class 24.2%', 'value': 'survival_by_class'},
    {'label': '6. Age Distribution by Class - First Class Passengers Were Older', 'value': 'age_by_class'},
    {'label': '7. Age Group & Class Intersection - Combined Effect on Survival', 'value': 'age_class_heatmap'},
    {'label': '8. Gender Survival Disparity - Women 74.2%, Men 18.9%', 'value': 'gender_survival'},
    {'label': '9. Multi-Factor Analysis - All Factors Combined', 'value': 'multi_factor'},
    {'label': '10. Data Overview - Summary Statistics', 'value': 'data_overview'}
]

# Define the app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Titanic Data Analysis - Interactive Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px',
                       'fontFamily': 'Arial, sans-serif', 'fontWeight': 'bold'}),
        html.P("Comprehensive visualization of Titanic survival patterns and passenger demographics", 
               style={'textAlign': 'center', 'color': '#34495e', 'fontSize': '18px',
                      'fontFamily': 'Arial, sans-serif'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '30px',
              'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}),
    
    # Control panel
    html.Div([
        html.Div([
            html.Label("Select an Insight to Explore:", 
                      style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '10px',
                             'color': '#2c3e50'}),
            dcc.Dropdown(
                id='insight-dropdown',
                options=insight_options,
                value='survival_counts',
                style={'marginBottom': '20px', 'fontFamily': 'Arial, sans-serif'}
            )
        ], style={'width': '100%', 'padding': '0 20px'})
    ], style={'backgroundColor': 'white', 'padding': '20px', 'marginBottom': '20px', 
              'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}),
    
    # Main visualization area
    html.Div([
        dcc.Graph(id='main-visualization', style={'height': '600px'})
    ], style={'padding': '0 20px', 'backgroundColor': 'white', 'borderRadius': '10px',
              'marginBottom': '20px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}),
    
    # Insight description and stats in two columns
    html.Div([
        # Left column - Insight description
        html.Div([
            html.Div(id='insight-description')
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
                  'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
                  'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}),
        
        # Right column - Statistics summary
        html.Div([
            html.H3("Key Statistics", style={'textAlign': 'center', 'color': '#2c3e50',
                                           'marginBottom': '15px'}),
            html.Div(id='stats-summary')
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
                  'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
                  'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginLeft': '4%'})
    ], style={'margin': '20px'})
])

# Callback for updating visualization
@app.callback(
    [Output('main-visualization', 'figure'),
     Output('insight-description', 'children'),
     Output('stats-summary', 'children')],
    [Input('insight-dropdown', 'value')]
)
def update_visualization(selected_insight):
    
    if selected_insight == 'survival_counts':
        # Insight 1: Overall Survival Pattern
        survival_counts = df['survived'].value_counts()
        fig = go.Figure(data=[
            go.Bar(x=['Died', 'Survived'], y=survival_counts.values, 
                   marker_color=['#e74c3c', '#27ae60'],
                   text=survival_counts.values,
                   textposition='auto',
                   textfont=dict(size=16, color='white'))
        ])
        fig.update_layout(
            title={'text': 'Overall Survival Counts', 'x': 0.5, 'font': {'size': 20}},
            xaxis_title='Outcome',
            yaxis_title='Number of Passengers',
            template='plotly_white',
            showlegend=False
        )
        
        description = html.Div([
            html.H4("The Devastating Reality", style={'color': '#8b4513', 'marginBottom': '15px'}),
            html.P(f"Out of {len(df)} passengers aboard the Titanic, {survival_counts[0]} perished while only {survival_counts[1]} survived, representing a devastating {(survival_counts[0]/len(df)*100):.1f}% mortality rate. This stark visualization reveals the catastrophic nature of one of history's most tragic maritime disasters.")
        ])
        
        mortality_rate = survival_counts[0] / len(df) * 100
        survival_rate = survival_counts[1] / len(df) * 100
        
        stats = html.Div([
            html.P(f"📊 Total Passengers: {len(df):,}", style={'margin': '10px 0'}),
            html.P(f"💀 Deaths: {survival_counts[0]:,} ({mortality_rate:.1f}%)", 
                   style={'margin': '10px 0', 'color': '#e74c3c'}),
            html.P(f"✅ Survivors: {survival_counts[1]:,} ({survival_rate:.1f}%)", 
                   style={'margin': '10px 0', 'color': '#27ae60'}),
            html.P(f"📈 Odds of Survival: 1 in {len(df)/survival_counts[1]:.1f}", 
                   style={'margin': '10px 0', 'fontWeight': 'bold'})
        ])
    
    elif selected_insight == 'age_distribution':
        # Insight 2: Age Demographics
        fig = px.histogram(df, x='age', nbins=25, 
                          title='Age Distribution of Titanic Passengers',
                          color_discrete_sequence=['#3498db'])
        fig.update_layout(
            title={'x': 0.5, 'font': {'size': 20}},
            xaxis_title='Age (years)',
            yaxis_title='Number of Passengers',
            template='plotly_white'
        )
        fig.add_vline(x=df['age'].mean(), line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {df['age'].mean():.1f} years")
        
        description = html.Div([
            html.H4("Young Adult Majority", style={'color': '#3498db', 'marginBottom': '15px'}),
            html.P("The age distribution reveals that the majority of Titanic passengers were young adults in their 20s and 30s. This demographic likely consisted of immigrants seeking new opportunities in America, young professionals on business trips, and newlyweds on their honeymoon voyages. The relatively few elderly passengers reflects the physical demands and costs of transatlantic travel in 1912.")
        ])
        
        age_stats = {
            'mean': df['age'].mean(),
            'median': df['age'].median(),
            'min': df['age'].min(),
            'max': df['age'].max(),
            'std': df['age'].std()
        }
        
        young_adults = len(df[(df['age'] >= 20) & (df['age'] < 40)])
        children = len(df[df['age'] < 18])
        
        stats = html.Div([
            html.P(f"📊 Average Age: {age_stats['mean']:.1f} years", style={'margin': '8px 0'}),
            html.P(f"📊 Median Age: {age_stats['median']:.1f} years", style={'margin': '8px 0'}),
            html.P(f"👶 Children (<18): {children} ({children/len(df)*100:.1f}%)", style={'margin': '8px 0'}),
            html.P(f"👨‍💼 Young Adults (20-40): {young_adults} ({young_adults/len(df)*100:.1f}%)", style={'margin': '8px 0'}),
            html.P(f"📈 Age Range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years", style={'margin': '8px 0'})
        ])
    
    elif selected_insight == 'fare_by_class':
        # Insight 3: Economic Class Structure
        fig = px.box(df, x='class', y='fare', 
                     title='Fare Distribution by Passenger Class',
                     color='class',
                     color_discrete_sequence=['#8b4513', '#ff8c00', '#32cd32'])
        fig.update_layout(
            title={'x': 0.5, 'font': {'size': 20}},
            template='plotly_white'
        )
        
        description = html.Div([
            html.H4("Economic Stratification", style={'color': '#ff8c00', 'marginBottom': '15px'}),
            html.P("The fare distribution reveals extreme economic inequality aboard the Titanic. First-class passengers paid premium prices for luxury accommodations, with some paying over $500 (equivalent to $13,000+ today). Meanwhile, third-class passengers paid modest fares under $50 for basic accommodations. This economic divide would prove tragically consequential for survival outcomes.")
        ])
        
        class_fare_stats = df.groupby('class')['fare'].agg(['mean', 'median', 'min', 'max', 'std'])
        
        stats = html.Div([
            html.P("💰 First Class:", style={'fontWeight': 'bold', 'color': '#8b4513'}),
            html.P(f"  Average: ${class_fare_stats.loc['First', 'mean']:.2f}", style={'margin': '5px 0 5px 20px'}),
            html.P(f"  Max: ${class_fare_stats.loc['First', 'max']:.2f}", style={'margin': '5px 0 10px 20px'}),
            html.P("💰 Second Class:", style={'fontWeight': 'bold', 'color': '#ff8c00'}),
            html.P(f"  Average: ${class_fare_stats.loc['Second', 'mean']:.2f}", style={'margin': '5px 0 10px 20px'}),
            html.P("💰 Third Class:", style={'fontWeight': 'bold', 'color': '#32cd32'}),
            html.P(f"  Average: ${class_fare_stats.loc['Third', 'mean']:.2f}", style={'margin': '5px 0 10px 20px'}),
            html.P(f"📊 Price Ratio (1st:3rd): {class_fare_stats.loc['First', 'mean']/class_fare_stats.loc['Third', 'mean']:.1f}:1", 
                   style={'fontWeight': 'bold'})
        ])
    
    elif selected_insight == 'age_vs_fare':
        # Insight 4: Age vs Fare Relationship
        color_map = {0: '#e74c3c', 1: '#27ae60'}
        fig = px.scatter(df, x='age', y='fare', color='survived',
                        title='Age vs Fare by Survival Status',
                        color_discrete_map=color_map,
                        hover_data=['class', 'sex'],
                        labels={'survived': 'Survived'})
        fig.update_layout(
            title={'x': 0.5, 'font': {'size': 20}},
            template='plotly_white'
        )
        
        description = html.Div([
            html.H4("Wealth and Survival Connection", style={'color': '#9b59b6', 'marginBottom': '15px'}),
            html.P("This scatter plot reveals a crucial pattern: green dots (survivors) are more concentrated in higher fare regions, demonstrating that wealth significantly increased survival chances. Young passengers who paid higher fares had particularly favorable survival odds, suggesting that both economic status and age influenced access to lifeboats during the disaster.")
        ])
        
        fare_median = df['fare'].median()
        high_fare_survival = df[df['fare'] > fare_median]['survived'].mean()
        low_fare_survival = df[df['fare'] <= fare_median]['survived'].mean()
        
        young_high_fare = df[(df['age'] < 40) & (df['fare'] > fare_median)]['survived'].mean()
        old_low_fare = df[(df['age'] >= 40) & (df['fare'] <= fare_median)]['survived'].mean()
        
        stats = html.Div([
            html.P(f"💰 High Fare (>${fare_median:.2f}+): {high_fare_survival:.1%} survived", 
                   style={'margin': '8px 0', 'color': '#27ae60'}),
            html.P(f"💸 Low Fare (≤${fare_median:.2f}): {low_fare_survival:.1%} survived", 
                   style={'margin': '8px 0', 'color': '#e74c3c'}),
            html.P(f"📊 Wealth Advantage: {high_fare_survival - low_fare_survival:.1%} better survival", 
                   style={'margin': '8px 0', 'fontWeight': 'bold'}),
            html.P(f"🎯 Young + Wealthy: {young_high_fare:.1%} survived", style={'margin': '8px 0'}),
            html.P(f"⚠️ Old + Poor: {old_low_fare:.1%} survived", style={'margin': '8px 0'})
        ])
    
    elif selected_insight == 'survival_by_class':
        # Insight 5: Class-Based Survival Inequality
        class_survival = df.groupby('class')['survived'].mean()
        fig = go.Figure(data=[
            go.Bar(x=class_survival.index, y=class_survival.values,
                   marker_color=['#8b4513', '#ff8c00', '#32cd32'],
                   text=[f'{rate:.1%}' for rate in class_survival.values],
                   textposition='auto',
                   textfont=dict(size=14, color='white'))
        ])
        fig.update_layout(
            title={'text': 'Survival Rate by Passenger Class', 'x': 0.5, 'font': {'size': 20}},
            xaxis_title='Passenger Class',
            yaxis_title='Survival Rate',
            template='plotly_white',
            showlegend=False
        )
        
        description = html.Div([
            html.H4("Class-Based Life and Death", style={'color': '#8b4513', 'marginBottom': '15px'}),
            html.P("This chart exposes the stark inequality in survival outcomes. First-class passengers enjoyed 2.6 times better survival odds than third-class passengers. The 'women and children first' protocol was applied differently across classes - proximity to lifeboats, crew assistance, and knowledge of the ship's layout all favored the wealthy. Social hierarchy literally determined who lived or died during the disaster.")
        ])
        
        class_counts = df.groupby('class').size()
        class_survivors = df.groupby('class')['survived'].sum()
        
        stats = html.Div([
            html.P("🥇 First Class:", style={'fontWeight': 'bold', 'color': '#8b4513'}),
            html.P(f"  {class_survivors['First']}/{class_counts['First']} survived ({class_survival['First']:.1%})", 
                   style={'margin': '5px 0 10px 20px'}),
            html.P("🥈 Second Class:", style={'fontWeight': 'bold', 'color': '#ff8c00'}),
            html.P(f"  {class_survivors['Second']}/{class_counts['Second']} survived ({class_survival['Second']:.1%})", 
                   style={'margin': '5px 0 10px 20px'}),
            html.P("🥉 Third Class:", style={'fontWeight': 'bold', 'color': '#32cd32'}),
            html.P(f"  {class_survivors['Third']}/{class_counts['Third']} survived ({class_survival['Third']:.1%})", 
                   style={'margin': '5px 0 10px 20px'}),
            html.P(f"📊 Class Advantage: {class_survival['First'] - class_survival['Third']:.1%} gap between 1st & 3rd", 
                   style={'fontWeight': 'bold', 'color': '#e74c3c'})
        ])
    
    elif selected_insight == 'age_by_class':
        # Insight 6: Age Distribution by Class
        fig = px.violin(df, x='class', y='age', 
                       title='Age Distribution by Passenger Class',
                       color='class',
                       color_discrete_sequence=['#8b4513', '#ff8c00', '#32cd32'])
        fig.update_layout(
            title={'x': 0.5, 'font': {'size': 20}},
            template='plotly_white'
        )
        
        description = html.Div([
            html.H4("Age and Economic Status", style={'color': '#34495e', 'marginBottom': '15px'}),
            html.P("The violin plots reveal distinct age patterns across passenger classes. First-class passengers were generally older, established individuals with accumulated wealth. Third-class passengers skewed younger, likely comprising immigrants, young workers, and families seeking new opportunities in America. This age-wealth correlation reflects the socioeconomic realities of early 20th-century society.")
        ])
        
        age_by_class = df.groupby('class')['age'].agg(['mean', 'median', 'std'])
        
        stats = html.Div([
            html.P("👔 First Class:", style={'fontWeight': 'bold', 'color': '#8b4513'}),
            html.P(f"  Average: {age_by_class.loc['First', 'mean']:.1f} years", style={'margin': '5px 0 10px 20px'}),
            html.P("👨‍💼 Second Class:", style={'fontWeight': 'bold', 'color': '#ff8c00'}),
            html.P(f"  Average: {age_by_class.loc['Second', 'mean']:.1f} years", style={'margin': '5px 0 10px 20px'}),
            html.P("👷 Third Class:", style={'fontWeight': 'bold', 'color': '#32cd32'}),
            html.P(f"  Average: {age_by_class.loc['Third', 'mean']:.1f} years", style={'margin': '5px 0 10px 20px'}),
            html.P(f"📊 Age Gap: {age_by_class.loc['First', 'mean'] - age_by_class.loc['Third', 'mean']:.1f} years between 1st & 3rd class", 
                   style={'fontWeight': 'bold'})
        ])
    
    elif selected_insight == 'age_class_heatmap':
        # Insight 7: Age Group & Class Intersection
        pivot_table = df.groupby(['age_grouped', 'class'])['survived'].mean().unstack()
        fig = px.imshow(pivot_table, 
                       title='Survival Rate Heatmap: Age Group vs Class',
                       color_continuous_scale='RdYlGn',
                       text_auto='.2f',
                       aspect='auto')
        fig.update_layout(
            title={'x': 0.5, 'font': {'size': 20}},
            template='plotly_white'
        )
        
        description = html.Div([
            html.H4("Double Advantage Effect", style={'color': '#27ae60', 'marginBottom': '15px'}),
            html.P("This heatmap reveals how age and class factors combined to determine survival outcomes. The brightest green cells show that being both young AND wealthy provided the ultimate advantage. First-class children had the highest survival rates, while third-class middle-aged adults faced the worst odds. The intersection of multiple demographic factors created a complex hierarchy of survival chances.")
        ])
        
        best_combo_rate = pivot_table.max().max()
        worst_combo_rate = pivot_table.min().min()
        best_combo_idx = pivot_table.stack().idxmax()
        worst_combo_idx = pivot_table.stack().idxmin()
        
        stats = html.Div([
            html.P(f"🏆 Best Survival Rate: {best_combo_rate:.1%}", 
                   style={'margin': '8px 0', 'color': '#27ae60', 'fontWeight': 'bold'}),
            html.P(f"   ({best_combo_idx[0]} in {best_combo_idx[1]} Class)", 
                   style={'margin': '0 0 10px 20px', 'color': '#27ae60'}),
            html.P(f"💀 Worst Survival Rate: {worst_combo_rate:.1%}", 
                   style={'margin': '8px 0', 'color': '#e74c3c', 'fontWeight': 'bold'}),
            html.P(f"   ({worst_combo_idx[0]} in {worst_combo_idx[1]} Class)", 
                   style={'margin': '0 0 10px 20px', 'color': '#e74c3c'}),
            html.P(f"📊 Maximum Advantage Gap: {best_combo_rate - worst_combo_rate:.1%}", 
                   style={'fontWeight': 'bold'})
        ])
    
    elif selected_insight == 'gender_survival':
        # Insight 8: Gender Survival Disparity
        gender_survival = df.groupby('sex')['survived'].mean()
        fig = go.Figure(data=[
            go.Bar(x=['Female', 'Male'], y=gender_survival.values,
                   marker_color=['#e91e63', '#2196f3'],
                   text=[f'{rate:.1%}' for rate in gender_survival.values],
                   textposition='auto',
                   textfont=dict(size=16, color='white'))
        ])
        fig.update_layout(
            title={'text': 'Survival Rate by Gender', 'x': 0.5, 'font': {'size': 20}},
            xaxis_title='Gender',
            yaxis_title='Survival Rate',
            template='plotly_white',
            showlegend=False
        )
        
        description = html.Div([
            html.H4("Women and Children First", style={'color': '#e91e63', 'marginBottom': '15px'}),
            html.P("The maritime protocol 'women and children first' was dramatically evident in the Titanic's survival statistics. Women had nearly 4 times better survival odds than men, reflecting the social chivalric codes of 1912. This gender disparity was enforced by crew members and male passengers who prioritized saving women and children, often at the cost of their own lives.")
        ])
        
        gender_counts = df.groupby('sex').size()
        gender_survivors = df.groupby('sex')['survived'].sum()
        
        stats = html.Div([
            html.P("👩 Women:", style={'fontWeight': 'bold', 'color': '#e91e63'}),
            html.P(f"  {gender_survivors['female']}/{gender_counts['female']} survived ({gender_survival['female']:.1%})", 
                   style={'margin': '5px 0 10px 20px'}),
            html.P("👨 Men:", style={'fontWeight': 'bold', 'color': '#2196f3'}),
            html.P(f"  {gender_survivors['male']}/{gender_counts['male']} survived ({gender_survival['male']:.1%})", 
                   style={'margin': '5px 0 10px 20px'}),
            html.P(f"📊 Gender Advantage: {gender_survival['female']/gender_survival['male']:.1f}x more likely (women)", 
                   style={'fontWeight': 'bold'}),
            html.P(f"📈 Absolute Difference: {gender_survival['female'] - gender_survival['male']:.1%}", 
                   style={'fontWeight': 'bold'})
        ])
    
    elif selected_insight == 'multi_factor':
        # Insight 9: Multi-Factor Analysis
        # Create subplots showing multiple factors
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Survival by Gender & Class', 'Age Distribution by Survival', 
                           'Fare vs Age (by Survival)', 'Class Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Survival by gender and class
        cross_tab = pd.crosstab([df['sex'], df['class']], df['survived'], normalize='index')
        for i, (gender, class_name) in enumerate(cross_tab.index):
            fig.add_trace(
                go.Bar(name=f'{gender.title()} {class_name}', 
                       x=[f'{gender.title()} {class_name}'], 
                       y=[cross_tab.loc[(gender, class_name), 1]],
                       showlegend=False),
                row=1, col=1
            )
        
        # Plot 2: Age distribution by survival
        survivors = df[df['survived'] == 1]['age']
        non_survivors = df[df['survived'] == 0]['age']
        fig.add_trace(go.Histogram(x=survivors, name='Survived', opacity=0.7, 
                                  marker_color='green', showlegend=False), row=1, col=2)
        fig.add_trace(go.Histogram(x=non_survivors, name='Died', opacity=0.7, 
                                  marker_color='red', showlegend=False), row=1, col=2)
        
        # Plot 3: Scatter of fare vs age
        fig.add_trace(go.Scatter(x=df[df['survived']==1]['age'], 
                                y=df[df['survived']==1]['fare'],
                                mode='markers', name='Survived', 
                                marker=dict(color='green', size=4),
                                showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df[df['survived']==0]['age'], 
                                y=df[df['survived']==0]['fare'],
                                mode='markers', name='Died', 
                                marker=dict(color='red', size=4),
                                showlegend=False), row=2, col=1)
        
        # Plot 4: Class distribution
        class_counts = df['class'].value_counts()
        fig.add_trace(go.Pie(values=class_counts.values, labels=class_counts.index,
                            showlegend=False), row=2, col=2)
        
        fig.update_layout(
            title={'text': 'Multi-Factor Survival Analysis', 'x': 0.5, 'font': {'size': 20}},
            template='plotly_white',
            height=600
        )
        
        description = html.Div([
            html.H4("Comprehensive Analysis", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.P("This multi-panel view combines all major factors affecting Titanic survival. The interplay between gender, class, age, and fare created a complex hierarchy of survival chances. First-class women had the best odds, while third-class men faced the greatest peril. The disaster revealed how social stratification translated directly into life-or-death outcomes.")
        ])
        
        # Calculate some complex statistics
        first_class_women = df[(df['class'] == 'First') & (df['sex'] == 'female')]
        third_class_men = df[(df['class'] == 'Third') & (df['sex'] == 'male')]
        
        best_group_survival = first_class_women['survived'].mean()
        worst_group_survival = third_class_men['survived'].mean()
        
        # Age impact within same class
        young_first_class = df[(df['class'] == 'First') & (df['age'] < 30)]['survived'].mean()
        old_first_class = df[(df['class'] == 'First') & (df['age'] >= 50)]['survived'].mean()
        
        stats = html.Div([
            html.P(f"👑 Best Group (1st Class Women): {best_group_survival:.1%} survived", 
                   style={'margin': '8px 0', 'color': '#27ae60', 'fontWeight': 'bold'}),
            html.P(f"💀 Worst Group (3rd Class Men): {worst_group_survival:.1%} survived", 
                   style={'margin': '8px 0', 'color': '#e74c3c', 'fontWeight': 'bold'}),
            html.P(f"📊 Maximum Group Gap: {best_group_survival - worst_group_survival:.1%}", 
                   style={'margin': '8px 0', 'fontWeight': 'bold'}),
            html.P(f"🎯 Young 1st Class: {young_first_class:.1%} survived", style={'margin': '8px 0'}),
            html.P(f"👴 Old 1st Class: {old_first_class:.1%} survived", style={'margin': '8px 0'}),
            html.P(f"🔄 Age still mattered even in 1st class!", style={'margin': '8px 0', 'fontStyle': 'italic'})
        ])
    
    elif selected_insight == 'data_overview':
        # Insight 10: Data Overview
        # Create a summary table
        summary_data = {
            'Metric': ['Total Passengers', 'Survival Rate', 'Average Age', 'Average Fare',
                      'Male Passengers', 'Female Passengers', 'First Class', 'Second Class', 
                      'Third Class', 'Children (<18)', 'Adults (18-64)', 'Seniors (65+)'],
            'Value': [
                f"{len(df):,}",
                f"{df['survived'].mean():.1%}",
                f"{df['age'].mean():.1f} years",
                f"${df['fare'].mean():.2f}",
                f"{len(df[df['sex']=='male']):,} ({len(df[df['sex']=='male'])/len(df)*100:.1f}%)",
                f"{len(df[df['sex']=='female']):,} ({len(df[df['sex']=='female'])/len(df)*100:.1f}%)",
                f"{len(df[df['class']=='First']):,} ({len(df[df['class']=='First'])/len(df)*100:.1f}%)",
                f"{len(df[df['class']=='Second']):,} ({len(df[df['class']=='Second'])/len(df)*100:.1f}%)",
                f"{len(df[df['class']=='Third']):,} ({len(df[df['class']=='Third'])/len(df)*100:.1f}%)",
                f"{len(df[df['age']<18]):,} ({len(df[df['age']<18])/len(df)*100:.1f}%)",
                f"{len(df[(df['age']>=18) & (df['age']<65)]):,} ({len(df[(df['age']>=18) & (df['age']<65)])/len(df)*100:.1f}%)",
                f"{len(df[df['age']>=65]):,} ({len(df[df['age']>=65])/len(df)*100:.1f}%)"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create a simple bar chart of key metrics
        fig = go.Figure()
        
        # Survival rates by different categories
        categories = ['Overall', 'Male', 'Female', 'First Class', 'Second Class', 'Third Class']
        survival_rates = [
            df['survived'].mean(),
            df[df['sex'] == 'male']['survived'].mean(),
            df[df['sex'] == 'female']['survived'].mean(),
            df[df['class'] == 'First']['survived'].mean(),
            df[df['class'] == 'Second']['survived'].mean(),
            df[df['class'] == 'Third']['survived'].mean()
        ]
        
        colors = ['#34495e', '#2196f3', '#e91e63', '#8b4513', '#ff8c00', '#32cd32']
        
        fig.add_trace(go.Bar(
            x=categories,
            y=[rate * 100 for rate in survival_rates],
            marker_color=colors,
            text=[f'{rate:.1%}' for rate in survival_rates],
            textposition='auto',
            textfont=dict(size=12, color='white')
        ))
        
        fig.update_layout(
            title={'text': 'Survival Rates Across Different Categories', 'x': 0.5, 'font': {'size': 20}},
            xaxis_title='Category',
            yaxis_title='Survival Rate (%)',
            template='plotly_white',
            showlegend=False
        )
        
        description = html.Div([
            html.H4("Dataset Overview", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.P("This comprehensive overview presents the key demographic and survival statistics from the Titanic dataset. The data reveals patterns of social inequality, demographic distributions, and survival outcomes that tell the story of one of history's most studied disasters. The stark differences in survival rates across gender, class, and age groups demonstrate how social hierarchies influenced life-and-death outcomes."),
            html.Br(),
            html.H5("Key Dataset Statistics:", style={'color': '#34495e', 'marginTop': '15px'}),
            html.Div([
                html.Table([
                    html.Thead([html.Tr([html.Th("Metric"), html.Th("Value")])]),
                    html.Tbody([
                        html.Tr([html.Td(row['Metric']), html.Td(row['Value'])]) 
                        for _, row in summary_df.iterrows()
                    ])
                ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse'}),
            ])
        ])
        
        # Advanced insights
        fare_age_corr = df['fare'].corr(df['age'])
        class_survival_range = df.groupby('class')['survived'].mean().max() - df.groupby('class')['survived'].mean().min()
        gender_survival_range = df.groupby('sex')['survived'].mean().max() - df.groupby('sex')['survived'].mean().min()
        
        stats = html.Div([
            html.P("🔍 Advanced Insights:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
            html.P(f"📈 Age-Fare Correlation: {fare_age_corr:.3f}", style={'margin': '8px 0'}),
            html.P(f"🎯 Class Survival Range: {class_survival_range:.1%}", style={'margin': '8px 0'}),
            html.P(f"⚡ Gender Survival Range: {gender_survival_range:.1%}", style={'margin': '8px 0'}),
            html.P(f"💡 Missing Data: {df.isnull().sum().sum()} total missing values", style={'margin': '8px 0'}),
            html.Br(),
            html.P("🎪 Fun Facts:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
            html.P(f"👶 Youngest: {df['age'].min():.1f} years old", style={'margin': '8px 0'}),
            html.P(f"👴 Oldest: {df['age'].max():.1f} years old", style={'margin': '8px 0'}),
            html.P(f"💰 Most Expensive Ticket: ${df['fare'].max():.2f}", style={'margin': '8px 0'}),
            html.P(f"🎫 Free Tickets: {len(df[df['fare'] == 0])} passengers", style={'margin': '8px 0'})
        ])
    
    return fig, description, stats

# Additional utility functions for enhanced analysis
def calculate_survival_metrics(df):
    """Calculate comprehensive survival metrics"""
    metrics = {}
    
    # Overall metrics
    metrics['total_passengers'] = len(df)
    metrics['total_survivors'] = df['survived'].sum()
    metrics['survival_rate'] = df['survived'].mean()
    
    # By class
    metrics['class_survival'] = df.groupby('class')['survived'].mean().to_dict()
    metrics['class_counts'] = df.groupby('class').size().to_dict()
    
    # By gender
    metrics['gender_survival'] = df.groupby('sex')['survived'].mean().to_dict()
    metrics['gender_counts'] = df.groupby('sex').size().to_dict()
    
    # By age groups
    age_groups = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], labels=['Child', 'Young Adult', 'Middle-aged', 'Senior'])
    metrics['age_group_survival'] = df.groupby(age_groups)['survived'].mean().to_dict()
    
    return metrics

# Run the app
if __name__ == '__main__':
    print("Starting Titanic Data Analysis Dashboard...")
    print(f"Dataset loaded: {len(df)} passengers")
    print(f"Survival rate: {df['survived'].mean():.1%}")
    print("\nStarting dashboard on http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop the server")
    
    app.run_server(debug=True, host='127.0.0.1', port=8050)


# Summary of Key Findings (for reference)
"""
TITANIC DATA ANALYSIS - KEY FINDINGS SUMMARY

1. OVERALL SURVIVAL PATTERN
   - 549 passengers died vs 342 survived (61.6% mortality rate)
   - Demonstrates the catastrophic nature of the disaster

2. DEMOGRAPHIC PATTERNS
   - Most passengers were young adults (20-40 years)
   - 577 male vs 314 female passengers
   - Clear age-wealth correlation across passenger classes

3. ECONOMIC STRATIFICATION
   - Extreme fare inequality: 1st class paid 15x more than 3rd class on average
   - First class: $84.15 average fare, up to $512.33
   - Third class: $13.68 average fare

4. SURVIVAL INEQUALITIES
   - Class-based: 1st (62.9%) > 2nd (47.3%) > 3rd (24.2%)
   - Gender-based: Women (74.2%) > Men (18.9%)
   - Age-based: Children and young adults had better survival rates

5. INTERSECTIONAL EFFECTS
   - First-class women had highest survival rates (~97%)
   - Third-class men had lowest survival rates (~13%)
   - Multiple demographic factors compounded advantages/disadvantages

6. SOCIAL IMPLICATIONS
   - "Women and children first" protocol clearly implemented
   - Wealth provided privileged access to lifeboats
   - Social hierarchy directly translated to survival outcomes
   - Disaster revealed deep inequalities of early 20th-century society

This analysis reveals how the Titanic disaster was not just a maritime tragedy,
but a stark illustration of how social class, gender, and age determined
life-and-death outcomes in a crisis situation.
"""
