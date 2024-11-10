# trial_pipeline.py
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime
from typing import Optional, List, Dict
import pandas as pd
from tqdm.auto import tqdm

from reddit_scraper import TrialSpecificScraper
from sentiment_analyzer import ClinicalTrialSentimentAnalyzer, analyze_sentiment_distribution
from message_generator import MessageGenerator

class TrialPipeline:
    def __init__(self, base_dir: str = "data", trial_details: Dict = None):
        """Initialize the pipeline with scraper, analyzer, and message generator."""
        load_dotenv()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Store trial details
        self.trial_details = trial_details or {
            'condition': 'Type 2 Diabetes',
            'phase': 'Phase 3',
            'location': 'Many locations',
            'duration': '10 weeks',
            'compensation': '$500'
        }
        
        # Initialize components
        self.scraper = TrialSpecificScraper(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT'),
            base_dir=base_dir
        )
        self.analyzer = ClinicalTrialSentimentAnalyzer()
        self.message_generator = MessageGenerator()
    
    def generate_personalized_messages(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Generate personalized messages for each post based on sentiment analysis."""
        self.logger.info("Generating personalized messages...")
        
        # Create a copy to avoid modifying the original
        df = results_df.copy()
        
        # Generate messages for each post
        messages = []
        for _, row in df.iterrows():
            message = self.message_generator.generate_message(
                post_text=row['text'],
                sentiment=row['trial_interest_sentiment'],
                confidence=row['trial_interest_confidence'],
                trial_details=self.trial_details
            )
            
            # Use available columns and add defaults if needed
            message_data = {
                'original_text': row['text'],
                'subreddit': row.get('subreddit', 'unknown'),
                'author': row.get('author', 'anonymous'),
                'sentiment': row['trial_interest_sentiment'],
                'confidence': row['trial_interest_confidence'],
                'personalized_message': message
            }
            
            # Add post_id if available
            if 'post_id' in row:
                message_data['post_id'] = row['post_id']
            elif '_id' in row:
                message_data['post_id'] = row['_id']
            else:
                message_data['post_id'] = f"post_{len(messages)}"
                
            messages.append(message_data)
        
        # Create messages DataFrame
        messages_df = pd.DataFrame(messages)
        
        return messages_df
    
    def run(self, 
            mode: str = 'load',
            input_file: Optional[str] = None,
            subreddits: Optional[List[str]] = None,
            condition: Optional[str] = None,
            keywords: Optional[List[str]] = None) -> None:
        """Run the complete pipeline with message generation."""
        try:
            # Steps 1-2: Get and Analyze Data (same as before)
            if mode == 'scrape':
                if not all([subreddits, condition]):
                    raise ValueError("subreddits and condition are required for scraping mode")
                
                self.logger.info(f"Scraping data for condition: {condition}")
                df = self.scraper.search_trial_specific_posts(
                    subreddits=subreddits,
                    condition=condition,
                    keywords=keywords,
                    save=True
                )
            else:  # load mode
                if not input_file:
                    available_files = self.scraper.list_available_data()
                    if not available_files:
                        raise ValueError("No data files available to load")
                    input_file = available_files[-1]
                    
                self.logger.info(f"Loading data from: {input_file}")
                df = self.scraper.load_data(input_file)
                
            
            
            if df.empty:
                self.logger.warning("No data to analyze")
                return None
            
            df['text'] = df['text'].astype(str).apply(lambda x: x.strip())
            # Remove empty or invalid texts
            df = df[df['text'].str.len() > 0]
            
            # Step 3: Analyze Sentiment
            self.logger.info("Analyzing sentiments...")
            sentiment_results = self.analyzer.analyze_sentiments(
                df,
                text_column='text',
                categories=['trial_interest', 'experience', 'intent']
            )
            
            # Step 4: Generate Messages
            messages_df = self.generate_personalized_messages(sentiment_results)
            
            # Step 5: Save Results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save sentiment analysis results
            sentiment_path = self.scraper.processed_dir / f"sentiment_results_{timestamp}.csv"
            sentiment_results.to_csv(sentiment_path, index=False)
            
            # Save personalized messages
            messages_path = self.scraper.processed_dir / f"personalized_messages_{timestamp}.csv"
            messages_df.to_csv(messages_path, index=False)
            
            # Step 6: Show
            print("\nAnalysis Results:")
            print(f"Total posts analyzed: {len(sentiment_results)}")
            analyze_sentiment_distribution(
                sentiment_results,
                categories=['trial_interest', 'experience', 'intent']
            )
            
            print("\nMessage Generation Summary:")
            print(f"Generated {len(messages_df)} personalized messages")
            print("\nSample message for positive sentiment:")
            positive_sample = messages_df[messages_df['sentiment'] == 'positive'].iloc[0] if not messages_df[messages_df['sentiment'] == 'positive'].empty else None
            if positive_sample is not None:
                print(f"\nOriginal post: {positive_sample['original_text'][:200]}...")
                print(f"\nGenerated message: {positive_sample['personalized_message']}")
            
            return sentiment_results, messages_df
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return None, None

def main():
    """Example usage of the pipeline with message generation."""
    # Define trial details
    trial_details = {
        'condition': 'Type 2 Diabetes',
        'phase': 'Phase 3',
        'location': 'Multiple US locations',
        'duration': '12 weeks',
        'compensation': '$500 for completion',
        'eligibility': 'Adults 18-65 with Type 2 Diabetes',
        'treatment': 'Novel oral medication'
    }
    
    pipeline = TrialPipeline(trial_details=trial_details)
    
    # Run scrape 
    sentiment_results, messages = pipeline.run(
        mode='scrape',
        subreddits=['diabetes', 'type2diabetes'],
        condition='Type 2 Diabetes',
        keywords=['insulin', 'glucose', 'A1C']
    )
    # sentiment_results, messages = pipeline.run(
    #     mode='load',
    #     input_file='posts_20241101_192718.csv'
    # )
    
    if sentiment_results is not None and messages is not None:
        print(f"\nResults saved to processed directory.")
        print("Files generated:")
        print(f"- sentiment_results_*.csv")
        print(f"- personalized_messages_*.csv")

if __name__ == "__main__":
    main()