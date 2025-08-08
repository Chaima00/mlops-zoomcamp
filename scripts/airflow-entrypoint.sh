#!/bin/bash
set -e

# Ensure logs directory exists and has correct permissions
mkdir -p /opt/airflow/logs/scheduler
chown -R airflow:root /opt/airflow/logs
chmod -R 775 /opt/airflow/logs

echo "Installing Python dependencies..."
pip install -r /requirements.txt

echo "Initializing Airflow DB..."
airflow db init

if ! airflow users list | grep -q admin; then
  echo "Creating admin user..."
  airflow users create \
    --username admin \
    --password admin \
    --firstname admin \
    --lastname admin \
    --role Admin \
    --email admin@example.com
else
  echo "Admin user already exists, skipping creation."
fi

echo "Starting scheduler in background..."
airflow scheduler &

echo "Starting webserver..."
exec airflow webserver --timeout 300