import openai
import logging
import os
from typing import Optional, List, Dict

class MessageGenerator:
    def __init__(self):
        """Initialize the message generator with OpenAI."""
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.logger = logging.getLogger(__name__)
        self.client = openai.OpenAI()  # Create client instance for v1.0.0+

    def generate_message(self, 
                        post_text: str, 
                        sentiment: str, 
                        confidence: float,
                        trial_details: Dict) -> str:
        """Generate personalized message based on post content and sentiment."""
        # Create prompt based on sentiment
        if sentiment == 'positive':
            prompt_base = (
                "Given a user's positive interest in clinical trials, create an engaging "
                "and informative response. Focus on their specific situation while "
                "maintaining professionalism."
            )
        elif sentiment == 'neutral':
            prompt_base = (
                "Create an educational message for someone who seems neutral about "
                "clinical trials. Address potential concerns while providing balanced information."
            )
        else:  # negative
            prompt_base = (
                "Craft a respectful response for someone who has concerns about "
                "clinical trials. Acknowledge their perspective while providing "
                "factual information."
            )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a knowledgeable and empathetic clinical trial "
                            "recruitment specialist. Generate a personalized response "
                            "to a potential participant based on their sentiment."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Context:
                        - User's post: {post_text}
                        - Their sentiment: {sentiment} (confidence: {confidence:.2f})
                        
                        Trial Details:
                        {trial_details}
                        
                        Task: {prompt_base}
                        
                        You must follow the following rules:
                        1. Reference specific points from their post
                        2. Address their particular concerns or interests
                        3. Include relevant trial details
                        4. Be concise but informative
                        5. Maintain a professional tone
                        """
                    }
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating message: {e}")
            return "Error generating personalized message."