import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # Initialize gates
        self.Wxi = np.random.randn(input_size, hidden_size) * 0.01
        self.Whi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bi = np.zeros((1, hidden_size))
        
        self.Wxf = np.random.randn(input_size, hidden_size) * 0.01
        self.Whf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bf = np.zeros((1, hidden_size))
        
        self.Wxo = np.random.randn(input_size, hidden_size) * 0.01
        self.Who = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bo = np.zeros((1, hidden_size))
        
        self.Wxc = np.random.randn(input_size, hidden_size) * 0.01
        self.Whc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bc = np.zeros((1, hidden_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x, prev_h, prev_c):
        self.x, self.prev_h, self.prev_c = x, prev_h, prev_c
        
        self.i = self.sigmoid(np.dot(x, self.Wxi) + np.dot(prev_h, self.Whi) + self.bi)
        self.f = self.sigmoid(np.dot(x, self.Wxf) + np.dot(prev_h, self.Whf) + self.bf)
        self.o = self.sigmoid(np.dot(x, self.Wxo) + np.dot(prev_h, self.Who) + self.bo)
        self.c_tilde = self.tanh(np.dot(x, self.Wxc) + np.dot(prev_h, self.Whc) + self.bc)
        
        self.c = self.f * prev_c + self.i * self.c_tilde
        self.h = self.o * self.tanh(self.c)
        
        return self.h, self.c

class DeepLearningModel:
    def __init__(self, input_size=20, hidden_size=32, output_size=2):
        self.lstm1 = LSTMCell(input_size, hidden_size)
        self.lstm2 = LSTMCell(hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_weights = np.random.randn(hidden_size, output_size) * 0.01
        self.output_bias = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def forward(self, x):
        batch_size = x.shape[0]
        h1 = np.zeros((batch_size, self.hidden_size))
        c1 = np.zeros((batch_size, self.hidden_size))
        h2 = np.zeros((batch_size, self.hidden_size))
        c2 = np.zeros((batch_size, self.hidden_size))
        
        h1, c1 = self.lstm1.forward(x, h1, c1)
        h2, c2 = self.lstm2.forward(h1, h2, c2)
        
        self.h2 = h2
        output = self.sigmoid(np.dot(h2, self.output_weights) + self.output_bias)
        
        return output
    
    def backward(self, x, y, learning_rate=0.01):
        d_output = (self.forward(x) - y)
        self.output_weights -= learning_rate * np.dot(self.h2.T, d_output)
        self.output_bias -= learning_rate * np.sum(d_output, axis=0, keepdims=True)

def validate_data(df):
    """
    Validate and clean the dataset
    Returns: cleaned dataframe and validation report
    """
    validation_report = {
        'total_rows': len(df),
        'issues': [],
        'removed_rows': 0,
        'affected_provinces': set()
    }
    
    # Create a copy of the dataframe
    cleaned_df = df.copy()
    
    # Check for NULL or None values
    null_mask = cleaned_df.isnull().any(axis=1) | (cleaned_df == 'None').any(axis=1)
    if null_mask.any():
        null_rows = cleaned_df[null_mask]
        validation_report['issues'].append({
            'type': 'NULL/None values',
            'count': len(null_rows),
            'provinces': list(null_rows['Province'].unique()),
            'years': list(null_rows['Year'].unique())
        })
        validation_report['removed_rows'] += len(null_rows)
        validation_report['affected_provinces'].update(null_rows['Province'].unique())
        cleaned_df = cleaned_df[~null_mask]
    
    # Check for invalid numeric values in Unemployment_Rate
    numeric_mask = pd.to_numeric(cleaned_df['Unemployment_Rate'], errors='coerce').isnull()
    if numeric_mask.any():
        invalid_rows = cleaned_df[numeric_mask]
        validation_report['issues'].append({
            'type': 'Invalid unemployment rate',
            'count': len(invalid_rows),
            'provinces': list(invalid_rows['Province'].unique()),
            'years': list(invalid_rows['Year'].unique())
        })
        validation_report['removed_rows'] += len(invalid_rows)
        validation_report['affected_provinces'].update(invalid_rows['Province'].unique())
        cleaned_df = cleaned_df[~numeric_mask]
    
    # Convert Unemployment_Rate to float
    cleaned_df['Unemployment_Rate'] = pd.to_numeric(cleaned_df['Unemployment_Rate'])
    
    # Check for out of range values (e.g., negative or unreasonably high)
    invalid_rate_mask = (cleaned_df['Unemployment_Rate'] < 0) | (cleaned_df['Unemployment_Rate'] > 100)
    if invalid_rate_mask.any():
        invalid_rate_rows = cleaned_df[invalid_rate_mask]
        validation_report['issues'].append({
            'type': 'Out of range unemployment rate',
            'count': len(invalid_rate_rows),
            'provinces': list(invalid_rate_rows['Province'].unique()),
            'years': list(invalid_rate_rows['Year'].unique())
        })
        validation_report['removed_rows'] += len(invalid_rate_rows)
        validation_report['affected_provinces'].update(invalid_rate_rows['Province'].unique())
        cleaned_df = cleaned_df[~invalid_rate_mask]
    
    # Check for duplicate entries
    duplicates = cleaned_df.duplicated(subset=['Year', 'Province'], keep=False)
    if duplicates.any():
        duplicate_rows = cleaned_df[duplicates]
        validation_report['issues'].append({
            'type': 'Duplicate entries',
            'count': len(duplicate_rows) // 2,  # Divide by 2 to get unique duplicate pairs
            'provinces': list(duplicate_rows['Province'].unique()),
            'years': list(duplicate_rows['Year'].unique())
        })
        # Keep the first occurrence of duplicates
        cleaned_df = cleaned_df.drop_duplicates(subset=['Year', 'Province'], keep='first')
        validation_report['removed_rows'] += len(duplicate_rows) // 2
    
    validation_report['affected_provinces'] = list(validation_report['affected_provinces'])
    validation_report['rows_remaining'] = len(cleaned_df)
    validation_report['provinces_remaining'] = len(cleaned_df['Province'].unique())
    
    return cleaned_df, validation_report

def display_validation_report(report):
    """
    Display the validation report in a user-friendly format
    """
    st.subheader("🔍 Data Validation Report")
    
    # Overall statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Rows Processed",
            report['total_rows']
        )
    with col2:
        st.metric(
            "Rows Removed",
            report['removed_rows'],
            delta=-report['removed_rows'],
            delta_color="inverse"
        )
    with col3:
        st.metric(
            "Rows Remaining",
            report['rows_remaining']
        )
    
    # Issues found
    if report['issues']:
        st.markdown("### Issues Found")
        for issue in report['issues']:
            with st.expander(f"🚫 {issue['type']} ({issue['count']} instances)"):
                st.markdown(f"**Affected Provinces:** {', '.join(issue['provinces'])}")
                st.markdown(f"**Affected Years:** {', '.join(map(str, issue['years']))}")
                st.markdown(f"**Number of Instances:** {issue['count']}")
    else:
        st.success("✅ No data quality issues found!")
    
    # Affected provinces summary
    if report['affected_provinces']:
        st.markdown("### Affected Provinces Summary")
        st.warning(
            f"Data quality issues were found in {len(report['affected_provinces'])} provinces: "
            f"{', '.join(report['affected_provinces'])}"
        )
    
    # Final data summary
    st.markdown("### Final Data Summary")
    st.info(
        f"After validation, the dataset contains {report['rows_remaining']} rows "
        f"from {report['provinces_remaining']} provinces."
    )

def normalize(data):
    """
    Normalize the data to range [0, 1]
    """
    if len(data) == 0:
        raise ValueError("Cannot normalize empty data")
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def denormalize(normalized_data, original_data):
    return normalized_data * (np.max(original_data) - np.min(original_data)) + np.min(original_data)

def prepare_sequences(data, seq_length, future_steps):
    if len(data) < seq_length + future_steps:
        raise ValueError(f"Not enough data. Need at least {seq_length + future_steps} points.")
    
    X, y = [], []
    for i in range(len(data) - seq_length - future_steps + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[(i + seq_length):(i + seq_length + future_steps)])
    
    return np.array(X), np.array(y)

def train_provincial_models(df, sequence_length, future_steps, epochs, learning_rate):
    results = []
    provinces = df['Province'].unique()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, province in enumerate(provinces):
        status_text.text(f"Training model for {province}...")
        
        province_data = df[df['Province'] == province]['Unemployment_Rate'].values
        
        try:
            model = DeepLearningModel(sequence_length, 32, future_steps)
            normalized_rates = normalize(province_data)
            X, y = prepare_sequences(normalized_rates, sequence_length, future_steps)
            
            losses = []
            for epoch in range(epochs):
                predictions = model.forward(X)
                loss = np.mean((predictions - y) ** 2)
                losses.append(loss)
                model.backward(X, y, learning_rate)
            
            final_predictions = model.forward(X)
            denormalized_predictions = denormalize(final_predictions, province_data)
            
            y_actual = province_data[sequence_length:sequence_length+future_steps]
            y_pred = denormalized_predictions[-1]
            
            mse = np.mean((y_pred - y_actual) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_pred - y_actual) / y_actual)) * 100
            success_rate = 100 - mape
            
            results.append({
                'Province': province,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape,
                'Success Rate': success_rate,
                'Last Prediction': y_pred.mean(),
                'Training Loss': losses[-1],
                'Min Loss': min(losses),
                'Loss Improvement': losses[0] - losses[-1]
            })
            
        except Exception as e:
            results.append({
                'Province': province,
                'MSE': None,
                'RMSE': None,
                'MAPE': None,
                'Success Rate': None,
                'Last Prediction': None,
                'Training Loss': None,
                'Min Loss': None,
                'Loss Improvement': None,
                'Error': str(e)
            })
        
        progress_bar.progress((i + 1) / len(provinces))
    
    status_text.text("Training completed!")
    return pd.DataFrame(results)

def calculate_provincial_statistics(df):
    """
    Calculate detailed statistics for each province
    """
    stats = []
    
    for province in df['Province'].unique():
        province_data = df[df['Province'] == province]['Unemployment_Rate']
        
        # Basic statistics
        basic_stats = province_data.describe()
        
        # Additional statistics
        trend = province_data.iloc[-1] - province_data.iloc[0]
        trend_percent = ((province_data.iloc[-1] - province_data.iloc[0]) / province_data.iloc[0]) * 100
        
        # Calculate year-over-year changes
        yoy_changes = province_data.pct_change() * 100
        avg_yoy_change = yoy_changes.mean()
        
        # Calculate volatility (standard deviation of year-over-year changes)
        volatility = yoy_changes.std()
        
        # Find years of max and min rates
        year_max = df[df['Province'] == province].loc[province_data.idxmax(), 'Year']
        year_min = df[df['Province'] == province].loc[province_data.idxmin(), 'Year']
        
        # Calculate 5-year average (if enough data)
        if len(province_data) >= 5:
            five_year_avg = province_data.tail(5).mean()
        else:
            five_year_avg = None
        
        stats.append({
            'Province': province,
            'Current Rate': province_data.iloc[-1],
            'Historical Average': basic_stats['mean'],
            'Minimum Rate': basic_stats['min'],
            'Maximum Rate': basic_stats['max'],
            'Standard Deviation': basic_stats['std'],
            'Year of Maximum': year_max,
            'Year of Minimum': year_min,
            'Overall Trend': trend,
            'Trend (%)': trend_percent,
            'Avg Annual Change (%)': avg_yoy_change,
            'Volatility': volatility,
            '5-Year Average': five_year_avg,
            'Data Points': len(province_data)
        })
    
    return pd.DataFrame(stats)

def plot_provincial_statistics(stats_df):
    """
    Create visualizations for provincial statistics
    """
    # Create tabs for different statistical views
    stat_tabs = st.tabs([
        "Basic Statistics",
        "Trend Analysis",
        "Volatility Analysis",
        "Historical Extremes"
    ])
    
    with stat_tabs[0]:
        # Basic statistics visualization
        fig = px.scatter(
            stats_df,
            x='Historical Average',
            y='Current Rate',
            size='Standard Deviation',
            color='Trend (%)',
            hover_data=['Province', '5-Year Average'],
            labels={
                'Historical Average': 'Historical Average Rate (%)',
                'Current Rate': 'Current Rate (%)',
                'Trend (%)': 'Overall Trend (%)'
            },
            title='Current vs Historical Unemployment Rates'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with stat_tabs[1]:
        # Trend analysis
        fig = px.bar(
            stats_df.sort_values('Trend (%)'),
            x='Province',
            y=['Trend (%)', 'Avg Annual Change (%)'],
            barmode='group',
            title='Unemployment Rate Trends by Province',
            labels={
                'value': 'Change (%)',
                'variable': 'Metric'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with stat_tabs[2]:
        # Volatility analysis
        fig = px.scatter(
            stats_df,
            x='Volatility',
            y='Avg Annual Change (%)',
            size='Standard Deviation',
            color='Current Rate',
            hover_data=['Province', 'Data Points'],
            title='Volatility vs Average Annual Change',
            labels={
                'Volatility': 'Rate Volatility (%)',
                'Avg Annual Change (%)': 'Average Annual Change (%)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with stat_tabs[3]:
        # Historical extremes
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Maximum Rate',
            x=stats_df['Province'],
            y=stats_df['Maximum Rate'],
            marker_color='red'
        ))
        
        fig.add_trace(go.Bar(
            name='Minimum Rate',
            x=stats_df['Province'],
            y=stats_df['Minimum Rate'],
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            name='Current Rate',
            x=stats_df['Province'],
            y=stats_df['Current Rate'],
            marker_color='blue'
        ))
        
        fig.update_layout(
            barmode='group',
            title='Historical Extremes vs Current Rate',
            yaxis_title='Unemployment Rate (%)'
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_provincial_comparison(results_df):
    # Create tabs for different visualizations
    viz_tabs = st.tabs([
        "Success Rates", 
        "Error Metrics", 
        "Predictions", 
        "Training Performance"
    ])
    
    with viz_tabs[0]:
        # Success Rate Visualization
        fig = px.bar(
            results_df.sort_values('Success Rate', ascending=True),
            x='Success Rate',
            y='Province',
            orientation='h',
            title='Model Success Rate by Province',
            labels={'Success Rate': 'Success Rate (%)'},
            color='Success Rate',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[1]:
        # Error Metrics Comparison
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('MSE by Province', 'RMSE by Province')
        )
        
        fig.add_trace(
            go.Bar(
                x=results_df['Province'],
                y=results_df['MSE'],
                name='MSE'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=results_df['Province'],
                y=results_df['RMSE'],
                name='RMSE'
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[2]:
        # Predictions Visualization
        fig = px.scatter(
            results_df,
            x='Province',
            y='Last Prediction',
            size='Success Rate',
            color='MAPE',
            title='Latest Predictions with Model Performance',
            labels={'Last Prediction': 'Predicted Unemployment Rate (%)'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[3]:
        # Training Performance
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training Loss', 'Loss Improvement')
        )
        
        fig.add_trace(
            go.Bar(
                x=results_df['Province'],
                y=results_df['Training Loss'],
                name='Final Training Loss'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=results_df['Province'],
                y=results_df['Loss Improvement'],
                name='Loss Improvement'
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Indonesia Provincial Unemployment Rate Prediction",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title('🇮🇩 Indonesia Provincial Unemployment Rate Prediction')
    
    # Sidebar
    st.sidebar.header('📊 Data Selection')
    data_source = st.sidebar.radio(
        "Choose Data Source",
        ["Use Example Data", "Upload Own Data"]
    )
    
    if data_source == "Upload Own Data":
        uploaded_file = st.sidebar.file_uploader(
            "Upload provincial data (CSV)",
            type=['csv'],
            help="CSV file with columns: Year, Province, Unemployment_Rate"
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Validate required columns
                required_columns = ['Year', 'Province', 'Unemployment_Rate']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    return
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
        else:
            st.sidebar.warning('Please upload a file or select "Use Example Data"')
            return
    else:
        if os.path.exists('provincial_unemployment_data.csv'):
            df = pd.read_csv('provincial_unemployment_data.csv')
        else:
            st.error("Example provincial data file not found!")
            return
    
    # Validate and clean data
    with st.spinner('Validating data...'):
        cleaned_df, validation_report = validate_data(df)
        
        # Display validation report
        display_validation_report(validation_report)
        
        # If there are no rows remaining after validation, stop processing
        if validation_report['rows_remaining'] == 0:
            st.error("No valid data remaining after validation. Please check your data and try again.")
            return
        
        # If more than 20% of data was removed, show warning
        if validation_report['removed_rows'] / validation_report['total_rows'] > 0.2:
            st.warning(
                f"⚠️ More than 20% of the data was removed during validation "
                f"({validation_report['removed_rows']} out of {validation_report['total_rows']} rows). "
                "Please review the validation report carefully."
            )
        
        # Update df to use cleaned data
        df = cleaned_df
    
    # Model Parameters
    st.sidebar.header('🔧 Model Parameters')
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        sequence_length = st.number_input(
            'Sequence Length',
            min_value=5,
            max_value=50,
            value=20,
            help="Years of historical data to use"
        )
    
    with col2:
        future_steps = st.number_input(
            'Prediction Years',
            min_value=1,
            max_value=5,
            value=2,
            help="Years to predict into future"
        )
    
    epochs = st.sidebar.number_input(
        'Training Epochs',
        min_value=10,
        max_value=1000,
        value=100,
        help="Number of training iterations"
    )
    
    learning_rate = st.sidebar.slider(
        'Learning Rate',
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        format='%.3f',
        help="Model learning rate"
    )
    
    # Data Overview
    st.header('📊 Data Overview')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Provinces", len(df['Province'].unique()))
    with col2:
        st.metric("Year Range", f"{df['Year'].min()} - {df['Year'].max()}")
    with col3:
        st.metric("Total Data Points", len(df))
    
    # Data viewer with filters
    with st.expander("🔍 View Data"):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Province filter
            selected_provinces = st.multiselect(
                "Select Provinces",
                options=sorted(df['Province'].unique()),
                default=None,
                placeholder="Choose provinces..."
            )
            
            # Year range filter
            year_range = st.slider(
                "Year Range",
                min_value=int(df['Year'].min()),
                max_value=int(df['Year'].max()),
                value=(int(df['Year'].min()), int(df['Year'].max()))
            )
            
            # Sort options
            sort_by = st.selectbox(
                "Sort by",
                options=['Year', 'Province', 'Unemployment_Rate'],
                index=0
            )
            
            sort_order = st.radio(
                "Sort order",
                options=['Ascending', 'Descending'],
                horizontal=True
            )
            
            # Download button for filtered data
            st.markdown("### Download Data")
            
            # Filter data based on selection
            filtered_df = df.copy()
            if selected_provinces:
                filtered_df = filtered_df[filtered_df['Province'].isin(selected_provinces)]
            filtered_df = filtered_df[
                (filtered_df['Year'] >= year_range[0]) & 
                (filtered_df['Year'] <= year_range[1])
            ]
            
            # Sort data
            ascending = sort_order == 'Ascending'
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
            
            # Download buttons
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=f"unemployment_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download the currently filtered data"
            )
            
            # Download full dataset
            full_csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Dataset (CSV)",
                data=full_csv,
                file_name=f"unemployment_data_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download the complete dataset"
            )
        
        with col2:
            # Display filtered and formatted data
            st.markdown("### Data Preview")
            
            # Format the display data
            display_df = filtered_df.copy()
            display_df['Unemployment_Rate'] = display_df['Unemployment_Rate'].apply(lambda x: f'{x:.2f}%')
            
            # Add row numbers
            display_df = display_df.reset_index(drop=True)
            display_df.index = display_df.index + 1
            
            # Show data with formatting
            st.dataframe(
                display_df,
                column_config={
                    "Year": st.column_config.NumberColumn(
                        "Year",
                        help="Year of measurement",
                        format="%d"
                    ),
                    "Province": st.column_config.TextColumn(
                        "Province",
                        help="Indonesian Province",
                        width="medium"
                    ),
                    "Unemployment_Rate": st.column_config.TextColumn(
                        "Unemployment Rate",
                        help="Unemployment rate in percentage"
                    )
                },
                use_container_width=True
            )
            
            # Show data summary
            st.markdown("### Data Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(filtered_df))
            with col2:
                st.metric("Provinces Shown", len(filtered_df['Province'].unique()))
            with col3:
                st.metric("Year Range", f"{filtered_df['Year'].min()} - {filtered_df['Year'].max()}")
            
            # Quick statistics for selected data
            if len(filtered_df) > 0:
                st.markdown("### Quick Statistics")
                stats_cols = st.columns(3)
                with stats_cols[0]:
                    st.metric(
                        "Average Rate",
                        f"{filtered_df['Unemployment_Rate'].mean():.2f}%"
                    )
                with stats_cols[1]:
                    st.metric(
                        "Minimum Rate",
                        f"{filtered_df['Unemployment_Rate'].min():.2f}%"
                    )
                with stats_cols[2]:
                    st.metric(
                        "Maximum Rate",
                        f"{filtered_df['Unemployment_Rate'].max():.2f}%"
                    )
    
    # Calculate and display provincial statistics
    st.header('📊 Provincial Statistics')
    
    # Calculate statistics
    stats_df = calculate_provincial_statistics(df)
    
    # Create tabs for different views of statistics
    stat_view_tabs = st.tabs(["Table View", "Visual Analysis"])
    
    with stat_view_tabs[0]:
        # Display statistics table with formatting
        st.subheader('Provincial Statistics Table')
        
        # Format numeric columns
        formatted_stats = stats_df.copy()
        
        # Format percentages
        pct_columns = ['Current Rate', 'Historical Average', 'Minimum Rate', 
                      'Maximum Rate', 'Trend (%)', 'Avg Annual Change (%)', 
                      'Volatility', '5-Year Average']
        
        for col in pct_columns:
            formatted_stats[col] = formatted_stats[col].apply(lambda x: f'{x:.2f}%' if pd.notnull(x) else 'N/A')
        
        # Format standard deviation
        formatted_stats['Standard Deviation'] = formatted_stats['Standard Deviation'].apply(
            lambda x: f'{x:.3f}' if pd.notnull(x) else 'N/A'
        )
        
        # Add sorting and filtering
        st.data_editor(
            formatted_stats,
            column_config={
                "Province": st.column_config.TextColumn(
                    "Province",
                    help="Indonesian Province",
                    width="medium",
                ),
                "Current Rate": st.column_config.TextColumn(
                    "Current Rate",
                    help="Most recent unemployment rate",
                ),
                "Year of Maximum": st.column_config.NumberColumn(
                    "Peak Year",
                    help="Year with highest unemployment rate",
                ),
                "Year of Minimum": st.column_config.NumberColumn(
                    "Trough Year",
                    help="Year with lowest unemployment rate",
                ),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Download button for statistics
        csv = stats_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Statistics (CSV)",
            data=csv,
            file_name=f"provincial_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with stat_view_tabs[1]:
        # Display statistical visualizations
        plot_provincial_statistics(stats_df)
        
        # Additional insights
        st.subheader('📈 Key Insights')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            highest_current = stats_df.loc[stats_df['Current Rate'].idxmax()]
            st.metric(
                "Highest Current Rate",
                f"{highest_current['Current Rate']:.2f}%",
                f"{highest_current['Province']}"
            )
        
        with col2:
            most_volatile = stats_df.loc[stats_df['Volatility'].idxmax()]
            st.metric(
                "Most Volatile Province",
                most_volatile['Province'],
                f"{most_volatile['Volatility']:.2f}% σ"
            )
        
        with col3:
            best_trend = stats_df.loc[stats_df['Trend (%)'].idxmin()]
            st.metric(
                "Best Improving Trend",
                best_trend['Province'],
                f"{best_trend['Trend (%)']:.2f}%"
            )
    
    # Train Models
    if st.button('🚀 Train Models for All Provinces'):
        try:
            # Train and get results
            results_df = train_provincial_models(
                df,
                sequence_length,
                future_steps,
                epochs,
                learning_rate
            )
            
            # Display results table
            st.header('📊 Model Performance by Province')
            
            # Format results
            display_df = results_df.copy()
            numeric_cols = ['MSE', 'RMSE', 'MAPE', 'Success Rate', 'Last Prediction']
            for col in numeric_cols:
                display_df[col] = display_df[col].apply(
                    lambda x: f'{x:.2f}' if x is not None else 'N/A'
                )
            
            # Show table with conditional formatting
            st.dataframe(
                display_df.style
                .highlight_max(subset=['Success Rate'], color='lightgreen')
                .highlight_min(subset=['MSE'], color='lightblue'),
                use_container_width=True
            )
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"provincial_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Visualizations
            st.header('📈 Performance Visualization')
            plot_provincial_comparison(results_df)
            
            # Summary Statistics
            st.header('📊 Summary Statistics')
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Average Success Rate",
                    f"{results_df['Success Rate'].mean():.2f}%",
                    f"{results_df['Success Rate'].std():.2f}% σ"
                )
            
            with col2:
                st.metric(
                    "Best Performing Province",
                    results_df.loc[results_df['Success Rate'].idxmax(), 'Province'],
                    f"{results_df['Success Rate'].max():.2f}%"
                )
            
            with col3:
                st.metric(
                    "Average RMSE",
                    f"{results_df['RMSE'].mean():.4f}",
                    f"{results_df['RMSE'].std():.4f} σ"
                )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Debug information:")
            st.write(f"Data shape: {df.shape}")
            st.write(f"Provinces: {df['Province'].unique()}")
            st.write(f"Years: {df['Year'].unique()}")

if __name__ == '__main__':
    main()
