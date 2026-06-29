"""
Application Entry Point
========================
Creates and runs the Flask application.

Usage:
    python run.py                    # Runs development server
    flask run                        # Alternative (uses FLASK_APP env var)
    flask db init                    # Initialize migrations
    flask db migrate -m "message"    # Generate migration
    flask db upgrade                 # Apply migrations
"""

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
