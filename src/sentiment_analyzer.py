import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm.auto import tqdm
import numpy as np
import logging
from functools import lru_cache
import os
from pathlib import Path

class ClinicalTrialSentimentAnalyzer:
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ClinicalTrialSentimentAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, batch_size=8, cache_dir=None):
        if not hasattr(self, 'initialized'):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_name = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
            self.batch_size = batch_size
            self.max_length = 512
            self.cache_dir = Path(cache_dir) if cache_dir else Path('model_cache')
            
            # Create cache directory if it doesn't exist
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            
            self.initialized = True
            
        if self._model is None or self._tokenizer is None:
            self.load_model()
            
    def prepare_candidate_labels(self):
        return {
            "trial_interest": [
                "interested in clinical trials",
                "neutral about clinical trials",
                "not interested in clinical trials"
            ],
            "experience": [
                "positive experience with clinical trials",
                "neutral experience with clinical trials",
                "negative experience with clinical trials"
            ],
            "intent": [
                "wants to participate in clinical trials",
                "undecided about participating in clinical trials",
                "does not want to participate in clinical trials"
            ]
        }

    def _format_for_zero_shot(self, text: str, candidate_label: str) -> str:
        """Format text and label for zero-shot classification."""
        return f"{text}\nThis text expresses: {candidate_label}"

    @classmethod
    def clear_cache(cls):
        """Clear the cached model and tokenizer."""
        cls._model = None
        cls._tokenizer = None
        torch.cuda.empty_cache()

    def is_model_cached(self) -> bool:
        """Check if model is cached."""
        model_path = self.cache_dir / self.model_name.replace('/', '_')
        return model_path.exists() and (model_path / "config.json").exists()
    
    def load_model(self, force_download=False):
        """Load the DeBERTa zero-shot model and tokenizer with caching."""
        if self._model is None or self._tokenizer is None:
            model_path = self.cache_dir / self.model_name.replace('/', '_')
            
            try:
                if not force_download and self.is_model_cached():
                    self.logger.info(f"Loading model from cache: {model_path}")
                    self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
                else:
                    self.logger.info(f"Downloading model: {self.model_name}")
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                    
                    # Save to cache
                    self.logger.info(f"Saving model to cache: {model_path}")
                    self._tokenizer.save_pretrained(model_path)
                    self._model.save_pretrained(model_path)
                
                self._model = self._model.to(self.device)
                self.logger.info("Model loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
                raise
            
        self.tokenizer = self._tokenizer
        self.model = self._model

    def _format_for_zero_shot(self, text: str, candidate_label: str) -> str:
        """Format text and label for zero-shot classification."""
        return f"{text}\nThis text expresses: {candidate_label}"

    def classify_text(self, text: str, candidate_labels: list) -> tuple:
        """Classify a single text against multiple candidate labels."""
        try:
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()
            
            if not text:
                return "neutral", 0.0

            # Process each candidate label
            max_score = -float('inf')
            best_label = None
            
            # Convert text for zero-shot
            formatted_sequences = [
                self._format_for_zero_shot(text, label)
                for label in candidate_labels
            ]
            
            # Tokenize all sequences at once
            inputs = self.tokenizer(
                formatted_sequences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                probs = softmax(logits, dim=1)
                
                sequence_scores = probs[:, 1]  # Take the positive class probability
                
                best_idx = torch.argmax(sequence_scores).item()
                max_score = sequence_scores[best_idx].item()
                best_label = candidate_labels[best_idx]

            return best_label, max_score
            
        except Exception as e:
            self.logger.error(f"Error in classification: {str(e)}")
            self.logger.error(f"Text sample: {text[:100]}...")
            return "neutral", 0.0

    def analyze_sentiments(self, df: pd.DataFrame, text_column: str, categories: list = None) -> pd.DataFrame:
        """Analyze sentiments for all texts in the DataFrame."""
        if categories is None:
            categories = list(self.prepare_candidate_labels().keys())
            
        all_results = []
        candidate_labels = self.prepare_candidate_labels()
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        df[text_column] = df[text_column].astype(str)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing sentiments"):
            result = {'text': row[text_column]}
            
            try:
                for category in categories:
                    labels = candidate_labels[category]
                    predicted_label, confidence = self.classify_text(row[text_column], labels)
                    
                    # Map to simple sentiment
                    sentiment_map = {
                        'interested': 'positive',
                        'positive': 'positive',
                        'wants': 'positive',
                        'neutral': 'neutral',
                        'undecided': 'neutral',
                        'not interested': 'negative',
                        'negative': 'negative',
                        'does not want': 'negative'
                    }
                    
                    sentiment = 'neutral'
                    for key, value in sentiment_map.items():
                        if key in predicted_label.lower():
                            sentiment = value
                            break
                    
                    result[f'{category}_sentiment'] = sentiment
                    result[f'{category}_confidence'] = confidence
                    result[f'{category}_raw_label'] = predicted_label
                
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {str(e)}")
                for category in categories:
                    result[f'{category}_sentiment'] = 'neutral'
                    result[f'{category}_confidence'] = 0.0
                    result[f'{category}_raw_label'] = 'neutral'
            
            all_results.append(result)
        
        results_df = pd.DataFrame(all_results)
        
        if len(df.columns) > 1:
            results_df = pd.merge(
                df.drop(columns=[text_column]), 
                results_df,
                left_index=True,
                right_index=True
            )
        
        return results_df

def analyze_sentiment_distribution(df, categories):
    """Analyze and print sentiment distribution for each category."""
    print("\nSentiment Distribution Analysis:")
    for category in categories:
        print(f"\n{category.title()}:")
        sentiment_counts = df[f'{category}_sentiment'].value_counts()
        sentiment_percentages = df[f'{category}_sentiment'].value_counts(normalize=True) * 100
        
        for sentiment in sentiment_counts.index:
            count = sentiment_counts[sentiment]
            percentage = sentiment_percentages[sentiment]
            avg_confidence = df[df[f'{category}_sentiment'] == sentiment][f'{category}_confidence'].mean()
            print(f"{sentiment.title():8} : {count:3d} ({percentage:5.1f}%) - Avg Confidence: {avg_confidence:.3f}")

def main():
    # Test the analyzer
    analyzer = ClinicalTrialSentimentAnalyzer()
    
    test_text = """I'm interested in participating in clinical trials but I have some concerns about the safety."""
    labels = analyzer.prepare_candidate_labels()["trial_interest"]
    
    label, conf = analyzer.classify_text(test_text, labels)
    print(f"\nTest Results:")
    print(f"Text: {test_text}")
    print(f"Label: {label}")
    print(f"Confidence: {conf:.3f}")

if __name__ == "__main__":
    main()