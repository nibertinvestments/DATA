"""
Comprehensive Web Scraping and API Examples in Python
Demonstrates requests, BeautifulSoup, API interactions, and error handling.
"""

import requests
from typing import Dict, List, Optional, Any
import json
import time
import logging
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import hashlib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ApiResponse:
    """Data class to represent API response."""
    status_code: int
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    response_time: float


class BaseApiClient:
    """Base class for API clients with common functionality."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Python-API-Client/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> ApiResponse:
        """Make HTTP request with error handling and timing."""
        url = urljoin(self.base_url + '/', endpoint.lstrip('/'))
        start_time = time.time()
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            response_time = time.time() - start_time
            
            # Try to parse JSON response
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {"raw_content": response.text}
            
            if response.status_code >= 400:
                error_msg = data.get('message', f'HTTP {response.status_code}')
                return ApiResponse(
                    status_code=response.status_code,
                    data=None,
                    error=error_msg,
                    response_time=response_time
                )
            
            return ApiResponse(
                status_code=response.status_code,
                data=data,
                error=None,
                response_time=response_time
            )
            
        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            logger.error(f"Request failed: {e}")
            return ApiResponse(
                status_code=0,
                data=None,
                error=str(e),
                response_time=response_time
            )
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> ApiResponse:
        """Make GET request."""
        return self._make_request('GET', endpoint, params=params)
    
    def post(self, endpoint: str, data: Optional[Dict] = None) -> ApiResponse:
        """Make POST request."""
        return self._make_request('POST', endpoint, json=data)
    
    def put(self, endpoint: str, data: Optional[Dict] = None) -> ApiResponse:
        """Make PUT request."""
        return self._make_request('PUT', endpoint, json=data)
    
    def delete(self, endpoint: str) -> ApiResponse:
        """Make DELETE request."""
        return self._make_request('DELETE', endpoint)


class JSONPlaceholderClient(BaseApiClient):
    """Client for JSONPlaceholder API - a fake REST API for testing."""
    
    def __init__(self):
        super().__init__('https://jsonplaceholder.typicode.com')
    
    def get_posts(self, user_id: Optional[int] = None) -> List[Dict]:
        """Get all posts or posts by user ID."""
        params = {'userId': user_id} if user_id else None
        response = self.get('/posts', params=params)
        
        if response.error:
            logger.error(f"Failed to get posts: {response.error}")
            return []
        
        return response.data if isinstance(response.data, list) else []
    
    def get_post(self, post_id: int) -> Optional[Dict]:
        """Get a specific post by ID."""
        response = self.get(f'/posts/{post_id}')
        
        if response.error:
            logger.error(f"Failed to get post {post_id}: {response.error}")
            return None
        
        return response.data
    
    def create_post(self, title: str, body: str, user_id: int) -> Optional[Dict]:
        """Create a new post."""
        post_data = {
            'title': title,
            'body': body,
            'userId': user_id
        }
        
        response = self.post('/posts', data=post_data)
        
        if response.error:
            logger.error(f"Failed to create post: {response.error}")
            return None
        
        return response.data
    
    def get_users(self) -> List[Dict]:
        """Get all users."""
        response = self.get('/users')
        
        if response.error:
            logger.error(f"Failed to get users: {response.error}")
            return []
        
        return response.data if isinstance(response.data, list) else []


class WebScraper:
    """Web scraper with BeautifulSoup (simulated - would require bs4)."""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Python-Scraper/1.0)'
        })
    
    def scrape_quotes(self) -> List[Dict[str, str]]:
        """
        Simulate scraping quotes (would use BeautifulSoup in real implementation).
        This is a mock implementation for demonstration purposes.
        """
        # Mock data that would normally be scraped
        mock_quotes = [
            {
                'text': 'The way to get started is to quit talking and begin doing.',
                'author': 'Walt Disney',
                'tags': ['inspirational', 'motivational']
            },
            {
                'text': 'Life is what happens to you while you\'re busy making other plans.',
                'author': 'John Lennon',
                'tags': ['life', 'wisdom']
            },
            {
                'text': 'The future belongs to those who believe in the beauty of their dreams.',
                'author': 'Eleanor Roosevelt',
                'tags': ['dreams', 'future', 'inspirational']
            }
        ]
        
        logger.info(f"Scraped {len(mock_quotes)} quotes")
        return mock_quotes
    
    def download_image(self, url: str, filename: str) -> bool:
        """Download an image from URL."""
        try:
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded image: {filename}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image: {e}")
            return False


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window  # seconds
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < self.time_window]
        
        # If we're at the limit, wait
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self.calls.append(now)


class CacheManager:
    """Simple file-based cache for API responses."""
    
    def __init__(self, cache_dir: str = "cache", expiry_hours: int = 24):
        self.cache_dir = cache_dir
        self.expiry_seconds = expiry_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from URL and parameters."""
        cache_input = url
        if params:
            cache_input += json.dumps(params, sort_keys=True)
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def get(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Get cached response if it exists and is not expired."""
        cache_key = self._get_cache_key(url, params)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            # Check if cache is expired
            if time.time() - os.path.getmtime(cache_file) < self.expiry_seconds:
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass
        
        return None
    
    def set(self, url: str, data: Dict, params: Optional[Dict] = None):
        """Cache the response data."""
        cache_key = self._get_cache_key(url, params)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to cache data: {e}")


class AsyncAPIClient:
    """
    Simulated async API client (would use aiohttp in real implementation).
    This shows the pattern without requiring async dependencies.
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.rate_limiter = RateLimiter(max_calls=10, time_window=60)  # 10 calls per minute
        self.cache = CacheManager()
    
    def fetch_multiple_posts(self, post_ids: List[int]) -> List[Optional[Dict]]:
        """
        Simulate fetching multiple posts concurrently.
        In real async implementation, this would use asyncio.gather().
        """
        client = JSONPlaceholderClient()
        results = []
        
        for post_id in post_ids:
            self.rate_limiter.wait_if_needed()
            
            # Check cache first
            cached_data = self.cache.get(f'/posts/{post_id}')
            if cached_data:
                logger.info(f"Using cached data for post {post_id}")
                results.append(cached_data)
                continue
            
            # Fetch from API
            post_data = client.get_post(post_id)
            if post_data:
                self.cache.set(f'/posts/{post_id}', post_data)
            
            results.append(post_data)
        
        return results


def demonstrate_api_client():
    """Demonstrate API client functionality."""
    client = JSONPlaceholderClient()
    
    print("=== API Client Demo ===")
    
    # Get all posts
    posts = client.get_posts()
    print(f"Retrieved {len(posts)} posts")
    
    # Get posts by specific user
    user_posts = client.get_posts(user_id=1)
    print(f"User 1 has {len(user_posts)} posts")
    
    # Get specific post
    post = client.get_post(1)
    if post:
        print(f"Post 1 title: {post.get('title', 'N/A')}")
    
    # Create new post
    new_post = client.create_post(
        title="Test Post",
        body="This is a test post created via API",
        user_id=1
    )
    if new_post:
        print(f"Created post with ID: {new_post.get('id', 'N/A')}")


def demonstrate_web_scraping():
    """Demonstrate web scraping functionality."""
    scraper = WebScraper(delay=1.0)
    
    print("\n=== Web Scraping Demo ===")
    
    # Scrape quotes (mock implementation)
    quotes = scraper.scrape_quotes()
    
    for i, quote in enumerate(quotes, 1):
        print(f"Quote {i}:")
        print(f"  Text: {quote['text']}")
        print(f"  Author: {quote['author']}")
        print(f"  Tags: {', '.join(quote['tags'])}")
        print()


def demonstrate_async_fetching():
    """Demonstrate async-style fetching with caching and rate limiting."""
    async_client = AsyncAPIClient('https://jsonplaceholder.typicode.com')
    
    print("\n=== Async Fetching Demo ===")
    
    # Fetch multiple posts
    post_ids = [1, 2, 3, 4, 5]
    posts = async_client.fetch_multiple_posts(post_ids)
    
    successful_fetches = [p for p in posts if p is not None]
    print(f"Successfully fetched {len(successful_fetches)} out of {len(post_ids)} posts")


def demonstrate_error_handling():
    """Demonstrate error handling in API calls."""
    client = BaseApiClient('https://httpstat.us')  # Service for testing HTTP status codes
    
    print("\n=== Error Handling Demo ===")
    
    # Test various HTTP status codes
    test_cases = [200, 404, 500, 503]
    
    for status_code in test_cases:
        response = client.get(f'/{status_code}')
        print(f"Status {status_code}: ", end="")
        
        if response.error:
            print(f"Error - {response.error}")
        else:
            print(f"Success - Response time: {response.response_time:.3f}s")


if __name__ == "__main__":
    print("=== Comprehensive API and Web Scraping Examples ===\n")
    
    # Run demonstrations
    demonstrate_api_client()
    demonstrate_web_scraping()
    demonstrate_async_fetching()
    demonstrate_error_handling()
    
    print("\n=== Additional Features Demonstrated ===")
    print("- RESTful API client with error handling")
    print("- Web scraping patterns (simulated)")
    print("- Rate limiting and caching")
    print("- Async-style concurrent requests")
    print("- Comprehensive error handling")
    print("- Logging and monitoring")
    print("- Data classes for structured responses")