# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Streamlit
EXPOSE 8501

# Set environment variable to allow duplicate libraries
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
