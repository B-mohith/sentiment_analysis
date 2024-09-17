import streamlit as st
import requests

# Streamlit UI
st.title("Sentiment Analysis App")

st.write("Upload a transcript file and get the sentiment analysis")

# File uploader widget
uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

# If a file is uploaded
if uploaded_file is not None:
    st.write("File uploaded successfully!")
    
    # Display file content
    st.write(uploaded_file.getvalue().decode("utf-8"))

    # When the user clicks the analyze button
    if st.button('Analyze Sentiment'):
        # Send the file to the FastAPI backend
        files = {'file': uploaded_file.getvalue()}
        response = requests.post("sentiback-production.up.railway.app/analyze", files={"file": uploaded_file})
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            st.write("Analysis Complete!")
            st.write(f"Positive Chunks: {result['positive_chunks']}")
            st.write(f"Negative Chunks: {result['negative_chunks']}")
            st.write(f"Neutral Chunks: {result['neutral_chunks']}")
            st.write(f"Total Chunks: {result['total_chunks']}")
            st.write(f"Overall Sentiment: {result['overall_sentiment']}")
        else:
            st.write("Error in the analysis. Please try again.")
