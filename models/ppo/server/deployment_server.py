# server/deployment_server.py
import sys
import os

# Add the parent directory to the Python path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, environment variables may not load")
except Exception as e:
    print(f"⚠️ Error loading .env file: {e}")

from flask import Flask, jsonify
from flask_cors import CORS
from server.api_routes import register_routes
import logging
import traceback
from werkzeug.exceptions import HTTPException

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentServer:
    """Deployment server wrapper for Flask app"""
    
    def __init__(self):
        self.app = Flask(__name__)
        
        # Enable CORS for webapp
        CORS(self.app, origins=["http://localhost:8080", "http://127.0.0.1:8080"])
        
        # Add error handlers
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            logger.error(f"Unhandled exception: {str(e)}")
            logger.error(traceback.format_exc())
            
            if isinstance(e, HTTPException):
                return jsonify({"error": str(e)}), e.code
                
            return jsonify({
                "error": "Internal server error",
                "details": str(e)
            }), 500
        
        register_routes(self.app)
        logger.info("Server initialized with updated Groq client")
    
    def run(self, host='0.0.0.0', port=8000, debug=False):
        """Run the Flask server"""
        logger.info(f"Starting server on {host}:{port} with debug={debug}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    server = DeploymentServer()
    server.run()