# Use an official Python runtime as a base image, favoring smaller "alpine" variants for production
FROM python:3.11-alpine

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages, using --no-cache-dir for efficiency
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt && rm -rf ~/.cache/pip

# Copy the rest of the application code into the container
COPY . .

# Define the command to run the application when the container starts
CMD ["python", "app.py"]
 