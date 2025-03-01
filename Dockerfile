# Use an official Python runtime as a parent image
FROM python:latest

# Set the working directory to /app
WORKDIR /app

# Install editor
RUN apt update 
RUN apt install vim -y
RUN apt install git

# Install Jupyter Notebook
RUN pip install notebook
RUN pip install --upgrade pip

# Copy files
COPY jupyter_server_config.* /root/.jupyter/
COPY rag-notebook .

# PIP install requirements
RUN pip install -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run Jupyter Notebook when the container launches
#CMD ["jupyter-lab", "--allow-root", "--config=/app/.jupyter/jupyter_server_config.py"]
CMD ["jupyter-lab", "--allow-root"]
