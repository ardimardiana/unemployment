import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import os

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # Input gate
        self.Wxi = np.random.randn(input_size, hidden_size) * 0.01
        self.Whi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bi = np.zeros((1, hidden_size))
        
        # Forget gate
        self.Wxf = np.random.randn(input_size, hidden_size) * 0.01
        self.Whf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bf = np.zeros((1, hidden_size))
        
        # Output gate
        self.Wxo = np.random.randn(input_size, hidden_size) * 0.01
        self.Who = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bo = np.zeros((1, hidden_size))
        
        # Cell state
        self.Wxc = np.random.randn(input_size, hidden_size) * 0.01
        self.Whc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bc = np.zeros((1, hidden_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x, prev_h, prev_c):
        # Input gate
        i = self.sigmoid(np.dot(x, self.Wxi) + np.dot(prev_h, self.Whi) + self.bi)
        
        # Forget gate
        f = self.sigmoid(np.dot(x, self.Wxf) + np.dot(prev_h, self.Whf) + self.bf)
        
        # Output gate
        o = self.sigmoid(np.dot(x, self.Wxo) + np.dot(prev_h, self.Who) + self.bo)
        
        # Cell state candidate
        c_candidate = self.tanh(np.dot(x, self.Wxc) + np.dot(prev_h, self.Whc) + self.bc)
        
        # New cell state
        c = f * prev_c + i * c_candidate
        
        # New hidden state
        h = o * self.tanh(c)
        
        return h, c

class DeepLearningModel:
    def __init__(self, input_size=20, hidden_size=32, output_size=2):
        self.lstm1 = LSTMCell(input_size, hidden_size)
        self.lstm2 = LSTMCell(hidden_size, hidden_size)
        self.hidden_size = hidden_size
        
        # Output layer weights
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
        
        # First LSTM layer
        h1, c1 = self.lstm1.forward(x, h1, c1)
        
        # Second LSTM layer
        h2, c2 = self.lstm2.forward(h1, h2, c2)
        
        # Output layer
        self.h2 = h2  # Save for backward pass
        output = self.sigmoid(np.dot(h2, self.output_weights) + self.output_bias)
        
        return output
    
    def backward(self, x, y, learning_rate=0.01):
        # Output layer gradients
        d_output = (self.forward(x) - y)
        
        # Update output layer
        self.output_weights -= learning_rate * np.dot(self.h2.T, d_output)
        self.output_bias -= learning_rate * np.sum(d_output, axis=0, keepdims=True)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def denormalize(normalized_data, original_data):
    return normalized_data * (np.max(original_data) - np.min(original_data)) + np.min(original_data)

def prepare_sequences(data, seq_length, future_steps):
    """
    Prepare sequences for training and prediction.
    Args:
        data: Input time series data
        seq_length: Number of time steps to use as input
        future_steps: Number of time steps to predict
    Returns:
        X: Input sequences of shape (n_samples, seq_length)
        y: Target sequences of shape (n_samples, future_steps)
    """
    if len(data) < seq_length + future_steps:
        raise ValueError(f"Not enough data. Need at least {seq_length + future_steps} points.")
        
    X, y = [], []
    # Create sequences up to the last point where we can make a full sequence
    for i in range(len(data) - seq_length - future_steps + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[(i + seq_length):(i + seq_length + future_steps)])
    
    X = np.array(X)
    y = np.array(y)
    
    # Print shapes for debugging
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y

def train_model(data, sequence_length=20, future_steps=2, epochs=100, learning_rate=0.01):
    # Normalize data
    normalized_rates = normalize(data)
    
    # Prepare sequences
    X, y = prepare_sequences(normalized_rates, sequence_length, future_steps)
    
    # Create model
    model = DeepLearningModel(sequence_length, 32, future_steps)
    
    # Training
    losses = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(X)
        
        # Calculate loss (MSE)
        loss = np.mean((predictions - y) ** 2)
        losses.append(loss)
        
        # Backward pass
        model.backward(X, y, learning_rate)
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Training Progress: {progress*100:.1f}% (Loss: {loss:.6f})")
    
    return model, losses

def plot_results(years, actual_data, predictions, pred_years, losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Training Loss
    ax1.plot(losses)
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot 2: Predictions
    ax2.plot(years, actual_data, 'b-', label='Actual Data')
    ax2.plot(pred_years, predictions, 'r--', label='Predictions')
    ax2.set_title('Indonesia Open Unemployment Rate Prediction')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Unemployment Rate (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(
        page_title="Indonesia Unemployment Rate Prediction",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title('🇮🇩 Indonesia Open Unemployment Rate Prediction')
    st.write("""
    This application predicts Indonesia's Open Unemployment Rate using Deep Learning with LSTM layers.
    Upload your historical data or use the example data provided.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your unemployment rate data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Use example data
        df = pd.read_csv('unemployment_data.csv')
    
    # Sidebar for data selection and parameters
    st.sidebar.header('📊 Data Selection')
    data_source = st.sidebar.radio(
        "Choose Data Source",
        ["Use Example Data", "Upload Own Data"]
    )
    
    if data_source == "Upload Own Data":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.sidebar.warning('Please upload a file or select "Use Example Data"')
            return
    else:
        df = pd.read_csv('unemployment_data.csv')
    
    # Display the data in main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('📈 Historical Data')
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader('📊 Data Statistics')
        st.write(df.describe())
    
    # Training parameters in sidebar
    st.sidebar.header('🔧 Model Parameters')
    
    sequence_length = st.sidebar.number_input(
        'Sequence Length (Years of Historical Data)',
        min_value=5,
        max_value=50,
        value=20,
        help="Number of years of historical data to use for prediction"
    )
    
    future_steps = st.sidebar.number_input(
        'Future Steps to Predict (Years)',
        min_value=1,
        max_value=5,
        value=2,
        help="Number of years to predict into the future"
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
        help="Model learning rate - lower values are more stable but train slower"
    )
    
    # Add model architecture information
    st.sidebar.header('ℹ️ Model Architecture')
    st.sidebar.markdown("""
    - Input Layer: {} nodes
    - LSTM Layer 1: 32 units
    - LSTM Layer 2: 32 units
    - Output Layer: {} nodes
    """.format(sequence_length, future_steps))
    
    # Validate data before training
    if len(df) < sequence_length + future_steps:
        st.error(f"Not enough data! Need at least {sequence_length + future_steps} years of data. Current data has {len(df)} years.")
        return
        
    # Training button
    if st.button('🚀 Train Model'):
        st.subheader('🔄 Training Progress')
        
        try:
            with st.spinner('Preparing data and initializing model...'):
                # Prepare data
                years = df['Year'].values
                unemployment_rates = df['Unemployment_Rate'].values
                
                # Train model
                model, losses = train_model(
                    unemployment_rates,
                    sequence_length=sequence_length,
                    future_steps=future_steps,
                    epochs=epochs,
                    learning_rate=learning_rate
                )
                
                # Make predictions
                normalized_rates = normalize(unemployment_rates)
                X, _ = prepare_sequences(normalized_rates, sequence_length, future_steps)
                final_predictions = model.forward(X)
                denormalized_predictions = denormalize(final_predictions, unemployment_rates)
                
                # Calculate metrics
                y = unemployment_rates[sequence_length:sequence_length+future_steps]
                y_pred = denormalized_predictions[-1]
                
                mse = np.mean((y_pred - y) ** 2)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((y_pred - y) / y)) * 100
                success_percentage = 100 - mape
                
                # Display metrics
                st.subheader('📊 Model Performance')
                col1, col2, col3, col4 = st.columns(4)
                col1.metric('MSE', f'{mse:.5f}')
                col2.metric('RMSE', f'{rmse:.5f}')
                col3.metric('MAPE', f'{mape:.2f}%')
                col4.metric('Success Rate', f'{success_percentage:.2f}%')
                
                # Plot results
                st.subheader('📈 Results Visualization')
                fig = plot_results(years, unemployment_rates, denormalized_predictions[-1], 
                                years[sequence_length:sequence_length+future_steps], losses)
                st.pyplot(fig)
                
                # Predictions for next years
                st.subheader('🔮 Future Predictions')
                next_years = [years[-1] + i + 1 for i in range(future_steps)]
                predictions_df = pd.DataFrame({
                    'Year': next_years,
                    'Predicted Unemployment Rate (%)': denormalized_predictions[-1]
                })
                st.dataframe(predictions_df)
                
        except Exception as e:
            st.error(f'An error occurred during training: {str(e)}')
            st.write("Debug information:")
            st.write(f"Data shape: {unemployment_rates.shape}")
            st.write(f"Sequence length: {sequence_length}")
            st.write(f"Future steps: {future_steps}")
            
        # Make predictions
        normalized_rates = normalize(unemployment_rates)
        X, _ = prepare_sequences(normalized_rates, sequence_length, future_steps)
        final_predictions = model.forward(X)
        denormalized_predictions = denormalize(final_predictions, unemployment_rates)
        
        # Calculate metrics
        y = unemployment_rates[sequence_length:sequence_length+future_steps]
        y_pred = denormalized_predictions[-1]  # Use only the last prediction
        
        mse = np.mean((y_pred - y) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_pred - y) / y)) * 100
        success_percentage = 100 - mape
        
        # Print debug information
        st.write("Debug Information:")
        st.write(f"Prediction shape: {y_pred.shape}")
        st.write(f"Actual shape: {y.shape}")
        st.write(f"Predictions: {y_pred}")
        st.write(f"Actual values: {y}")
        
        # Display metrics
        st.subheader('Model Performance')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('MSE', f'{mse:.5f}')
        col2.metric('RMSE', f'{rmse:.5f}')
        col3.metric('MAPE', f'{mape:.2f}%')
        col4.metric('Success Rate', f'{success_percentage:.2f}%')
        
        # Plot results
        st.subheader('Results Visualization')
        pred_years = years[sequence_length:sequence_length+future_steps]
        fig = plot_results(years, unemployment_rates, denormalized_predictions[-1], pred_years, losses)
        st.pyplot(fig)
        
        # Predictions for next years
        st.subheader('Predictions for Next Years')
        next_years = [years[-1] + i + 1 for i in range(future_steps)]
        predictions_df = pd.DataFrame({
            'Year': next_years,
            'Predicted Unemployment Rate (%)': denormalized_predictions[-1]
        })
        st.dataframe(predictions_df)

if __name__ == '__main__':
    main()
