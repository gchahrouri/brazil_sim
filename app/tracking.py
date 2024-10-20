import logging
from flask import request
import sqlite3
from datetime import datetime, date, timedelta
import uuid
import pytz
import os

# Use an environment variable for the database path, with a default for local development
DB_PATH = os.environ.get('SQLITE_DB_PATH', 'user_actions.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_actions
                 (id TEXT, timestamp TEXT, action TEXT, details TEXT)''')
    conn.commit()
    conn.close()

def log_action(action, details=''):
    conn = get_db_connection()
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    user_id = request.cookies.get('user_id', str(uuid.uuid4()))
    c.execute("INSERT INTO user_actions VALUES (?, ?, ?, ?)",
              (user_id, timestamp, action, details))
    conn.commit()
    conn.close()
    print(f"Debug - Logged action: {action}, Details: {details}, User ID: {user_id}")  # Debug log
    return user_id

def get_unique_users():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT id) FROM user_actions")
    count = c.fetchone()[0]
    conn.close()
    return count

def get_action_count(action):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM user_actions WHERE action = ?", (action,))
    count = c.fetchone()[0]
    conn.close()
    return count

def get_parameter_changes():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT details FROM user_actions WHERE action = 'parameter_change'")
    changes = [row[0].split(':')[0] for row in c.fetchall()]  # Only keep the parameter name, not the value
    change_counts = [(param, changes.count(param)) for param in set(changes)]
    conn.close()
    return sorted(change_counts, key=lambda x: x[1], reverse=True)  # Sort by count in descending order

def get_analytics_data():
    conn = get_db_connection()
    c = conn.cursor()
    
    pacific_tz = pytz.timezone('America/Los_Angeles')
    end_date = datetime.now(pacific_tz).date()
    start_date = end_date - timedelta(days=6)
    
    logging.info(f"Fetching analytics data from {start_date} to {end_date}")

    # Get total counts
    total_users = get_unique_users()
    total_simulation_runs = get_action_count('run_simulation')
    total_sensitivity_runs = get_action_count('run_sensitivity')
    
    logging.info(f"Total users: {total_users}")
    logging.info(f"Total simulation runs: {total_simulation_runs}")
    logging.info(f"Total sensitivity runs: {total_sensitivity_runs}")

    # Get unique users per day data in Pacific Time
    c.execute("""
        SELECT date(datetime(timestamp, 'localtime')) as day, COUNT(DISTINCT id) as user_count
        FROM user_actions
        WHERE date(datetime(timestamp, 'localtime')) >= ?
        GROUP BY date(datetime(timestamp, 'localtime'))
        ORDER BY date(datetime(timestamp, 'localtime'))
    """, (start_date.strftime('%Y-%m-%d'),))
    users_per_day = c.fetchall()
    
    logging.info(f"Users per day: {users_per_day}")

    # Get unique simulation users per day in Pacific Time
    c.execute("""
        SELECT date(datetime(timestamp, 'localtime')) as day, COUNT(DISTINCT id) as user_count
        FROM user_actions
        WHERE action = 'run_simulation' AND date(datetime(timestamp, 'localtime')) >= ?
        GROUP BY date(datetime(timestamp, 'localtime'))
        ORDER BY date(datetime(timestamp, 'localtime'))
    """, (start_date.strftime('%Y-%m-%d'),))
    simulation_users_per_day = c.fetchall()
    
    # Get unique sensitivity analysis users per day in Pacific Time
    c.execute("""
        SELECT date(datetime(timestamp, 'localtime')) as day, COUNT(DISTINCT id) as user_count
        FROM user_actions
        WHERE action = 'run_sensitivity' AND date(datetime(timestamp, 'localtime')) >= ?
        GROUP BY date(datetime(timestamp, 'localtime'))
        ORDER BY date(datetime(timestamp, 'localtime'))
    """, (start_date.strftime('%Y-%m-%d'),))
    sensitivity_users_per_day = c.fetchall()
    
    logging.info(f"Simulation users per day: {simulation_users_per_day}")
    logging.info(f"Sensitivity users per day: {sensitivity_users_per_day}")
    
    conn.close()
    return {
        'total_users': total_users,
        'total_simulation_runs': total_simulation_runs,
        'total_sensitivity_runs': total_sensitivity_runs,
        'users_per_day': users_per_day,
        'simulation_users_per_day': simulation_users_per_day,
        'sensitivity_users_per_day': sensitivity_users_per_day,
    }

def filter_simulations_for_sensitivity(simulations, target_param):
    """
    Filter simulations to keep only those where other parameters are at their base values.
    """
    base_values = {
        'marketing_spend': 160000000.00,
        'marketing_roi': 0.15,
        'scale_growth_factor': 1.0,
        'retention_rate': 0.80,
        'spend_growth': 0.10
    }
    
    return [
        sim for sim in simulations
        if all(
            sim[param] == base_values[param]
            for param in base_values
            if param != target_param
        )
    ]
