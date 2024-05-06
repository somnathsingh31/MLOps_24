# Use the official Apache Airflow image as a base image
FROM apache/airflow:latest

# Set the working directory in the container
WORKDIR /root/airflow

# Copy the DAGs directory into the container
COPY ./dags /root/airflow/dags

# Copy any additional files or directories needed for your DAGs
COPY ./data /root/airflow/dags/data
COPY ./models /root/airflow/dags/models

# Expose the ports used by Airflow (8080 by default)
EXPOSE 8080

# Command to start the Airflow webserver
CMD ["webserver"]
