# reddit_scraper.py
import praw
import pandas as pd
from datetime import datetime
import logging
from typing import List, Optional
from pathlib import Path

class TrialSpecificScraper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str, base_dir: str = "data"):
        """Initialize Reddit scraper with PRAW credentials and directory setup."""
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Setup directories
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        # Create directories 
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)

    def search_trial_specific_posts(self, 
                                  subreddits: List[str], 
                                  condition: str,
                                  keywords: List[str] = None,
                                  limit: int = 100,
                                  save: bool = True,
                                  filename: str = None) -> pd.DataFrame:
        """
        Search for posts relevant to a specific clinical trial condition.
        
        Args:
            subreddits: List of subreddit names to search
            condition: Medical condition for the trial
            keywords: Additional keywords to filter posts
            limit: Maximum posts per subreddit
            save: Whether to save the data to file
            filename: Custom filename to save data (optional)
        """
        all_posts = []
        base_keywords = ["trial", "study", "research", "participant", "clinical", "treatment", "clinical trial", "medical study", "medical trial"]
        if keywords:
            base_keywords.extend(keywords)
            
        search_query = f"{condition} ({' OR '.join(base_keywords)})"
        
        for subreddit_name in subreddits:
            self.logger.info(f"Searching r/{subreddit_name} for: {search_query}")
            subreddit = self.reddit.subreddit(subreddit_name)
            
            try:
                for post in subreddit.search(search_query, limit=limit):
                    post_data = {
                        'subreddit': subreddit_name,
                        'title': post.title,
                        'text': post.selftext,
                        'author': str(post.author),
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'score': post.score,
                        'id': post.id,
                        'url': post.url
                    }
                    all_posts.append(post_data)
                    
            except Exception as e:
                self.logger.error(f"Error searching r/{subreddit_name}: {str(e)}")
                continue
        
        df = pd.DataFrame(all_posts)
        
        # Save data if requested
        if save and not df.empty:
            saved_path = self.save_data(df, filename)
            self.logger.info(f"Saved raw data to {saved_path}")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: Optional[str] = None) -> Path:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"raw_posts_{timestamp}.csv"
        
        file_path = self.raw_dir / filename
        df.to_csv(file_path, index=False)
        return file_path
    
    def load_data(self, filename: str) -> pd.DataFrame:
        file_path = self.raw_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found at {file_path}")
        
        return pd.read_csv(file_path)
    
    def list_available_data(self) -> List[str]:
        return [f.name for f in self.raw_dir.glob("*.csv")]
