import requests
import logging
from typing import Optional, Dict, Any
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class WolframClient:
    """Client for Wolfram Alpha API"""
    
    def __init__(self, app_id: str):
        self.app_id = app_id
        self.base_url = "http://api.wolframalpha.com/v2/query"
        self.api_available = bool(app_id and app_id.strip())
        
        if self.api_available:
            logger.info("Wolfram Alpha API client initialized successfully")
        else:
            logger.warning("No Wolfram Alpha App ID provided")
    
    def query(self, query: str) -> Optional[Dict[str, Any]]:
        """Query Wolfram Alpha for computational or factual information"""
        if not self.api_available:
            logger.warning("Wolfram Alpha API not available")
            return None
            
        try:
            params = {
                'input': query,
                'appid': self.app_id,
                'format': 'plaintext',
                'output': 'xml'
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                return self._parse_wolfram_response(response.text)
            else:
                logger.error(f"Wolfram API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Wolfram query error: {e}")
            return None
    
    def _parse_wolfram_response(self, xml_response: str) -> Dict[str, Any]:
        """Parse Wolfram Alpha XML response"""
        try:
            root = ET.fromstring(xml_response)
            
            # Extract plaintext results
            results = []
            for pod in root.findall('.//pod'):
                title = pod.get('title', '')
                for plaintext in pod.findall('.//plaintext'):
                    if plaintext.text:
                        results.append({
                            'title': title,
                            'content': plaintext.text.strip()
                        })
            
            return {
                'success': True,
                'results': results,
                'query': root.get('inputstring', '')
            }
            
        except Exception as e:
            logger.error(f"Error parsing Wolfram response: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_computational_answer(self, query: str) -> str:
        """Get computational answer from Wolfram Alpha"""
        if not self.api_available:
            return f"Computational answer for {query} would be available with Wolfram Alpha App ID."
        
        result = self.query(query)
        if result and result.get('success'):
            results = result.get('results', [])
            if results:
                return results[0].get('content', f"Computational result for {query}")
        
        return f"Could not compute answer for {query}"
    
    def get_factual_answer(self, query: str) -> str:
        """Get factual answer from Wolfram Alpha"""
        if not self.api_available:
            return f"Factual information about {query} would be available with Wolfram Alpha App ID."
        
        result = self.query(query)
        if result and result.get('success'):
            results = result.get('results', [])
            if results:
                return results[0].get('content', f"Factual information about {query}")
        
        return f"Could not find factual information for {query}" 