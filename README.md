# BR Market Simulation

This Flask application simulates the Brazilian iGaming market dynamics from 2025 to 2030.

## Setup

1. Create a virtual environment and activate it:   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`   ```

2. Install the required packages:   ```
   pip install -r requirements.txt   ```

## Running the Application

1. Start the Flask development server:   ```
   python run.py   ```

2. Open a web browser and navigate to `http://localhost:5000`.

3. Enter the simulation parameters and click "Run Simulation" to view the results.

## Deployment

This application is configured for deployment on Heroku. To deploy:

1. Create a Heroku account and install the Heroku CLI.
2. Login to Heroku CLI: `heroku login`
3. Create a new Heroku app: `heroku create your-app-name`
4. Push the code to Heroku: `git push heroku main`
5. Open the app in your browser: `heroku open`
