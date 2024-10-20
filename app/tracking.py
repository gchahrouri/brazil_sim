import logging
from flask import request
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, date, timedelta
import uuid
import pytz
import os

# Use environment variables for database connection
DATABASE_URL = os.environ.get('DATABASE_URL')

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS user_actions
                   (id TEXT, timestamp TIMESTAMP, action TEXT, details TEXT)''')
    conn.commit()
    cur.close()
    conn.close()

def log_action(action, details=''):
    conn = get_db_connection()
    cur = conn.cursor()
    timestamp = datetime.now(pytz.utc)
    user_id = request.cookies.get('user_id', str(uuid.uuid4()))
    cur.execute("INSERT INTO user_actions (id, timestamp, action, details) VALUES (%s, %s, %s, %s)",
                (user_id, timestamp, action, details))
    conn.commit()
    cur.close()
    conn.close()
    logging.info(f"Logged action: {action}, Details: {details}, User ID: {user_id}")
    return user_id

def get_unique_users():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT id) FROM user_actions")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count

def get_action_count(action):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM user_actions WHERE action = %s", (action,))
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count

def get_parameter_changes():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT details FROM user_actions WHERE action = 'parameter_change'")
    changes = [row[0].split(':')[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return [(param, changes.count(param)) for param in set(changes)]

def get_analytics_data():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    
    pacific_tz = pytz.timezone('America/Los_Angeles')
    end_date = datetime.now(pacific_tz).date()
    start_date = end_date - timedelta(days=6)
    
    logging.info(f"Fetching analytics data from {start_date} to {end_date}")

    # Get total counts
    total_users = get_unique_users()
    total_simulation_runs = get_action_count('run_simulation')
    total_sensitivity_runs = get_action_count('run_sensitivity')

    # Get users per day data in Pacific Time
    cur.execute("""
        SELECT DATE(timestamp AT TIME ZONE 'America/Los_Angeles') as day, COUNT(DISTINCT id) as user_count
        FROM user_actions
        WHERE timestamp AT TIME ZONE 'America/Los_Angeles' >= %s
        GROUP BY DATE(timestamp AT TIME ZONE 'America/Los_Angeles')
        ORDER BY day
    """, (start_date,))
    users_per_day = cur.fetchall()

    # Get simulation and sensitivity users per day in Pacific Time
    cur.execute("""
        SELECT DATE(timestamp AT TIME ZONE 'America/Los_Angeles') as day, COUNT(DISTINCT id) as user_count
        FROM user_actions
        WHERE action = 'run_simulation' AND timestamp AT TIME ZONE 'America/Los_Angeles' >= %s
        GROUP BY DATE(timestamp AT TIME ZONE 'America/Los_Angeles')
        ORDER BY day
    """, (start_date,))
    simulation_users_per_day = cur.fetchall()

    cur.execute("""
        SELECT DATE(timestamp AT TIME ZONE 'America/Los_Angeles') as day, COUNT(DISTINCT id) as user_count
        FROM user_actions
        WHERE action = 'run_sensitivity' AND timestamp AT TIME ZONE 'America/Los_Angeles' >= %s
        GROUP BY DATE(timestamp AT TIME ZONE 'America/Los_Angeles')
        ORDER BY day
    """, (start_date,))
    sensitivity_users_per_day = cur.fetchall()

    cur.close()
    conn.close()

    # Convert database results to list of tuples
    users_per_day = [(row['day'].strftime('%Y-%m-%d'), row['user_count']) for row in users_per_day]
    simulation_users_per_day = [(row['day'].strftime('%Y-%m-%d'), row['user_count']) for row in simulation_users_per_day]
    sensitivity_users_per_day = [(row['day'].strftime('%Y-%m-%d'), row['user_count']) for row in sensitivity_users_per_day]

    logging.info(f"Users per day: {users_per_day}")
    logging.info(f"Simulation users per day: {simulation_users_per_day}")
    logging.info(f"Sensitivity users per day: {sensitivity_users_per_day}")

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
