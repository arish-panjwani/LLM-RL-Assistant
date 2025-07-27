# server/deployment_server.py
from flask import Flask
from server.api_routes import register_routes

class DeploymentServer:
    """Deployment server wrapper for Flask app"""
    
    def __init__(self):
        self.app = Flask(__name__)
        register_routes(self.app)
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask server"""
        self.app.run(host=host, port=port, debug=debug)

# Legacy app instance for backward compatibility
app = Flask(__name__)
register_routes(app)