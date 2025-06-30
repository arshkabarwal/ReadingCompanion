import anthropic
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Configuration
class Config:
    ANTHROPIC_API_KEY = "KEY"  # Replace with your actual API key
    MODEL = "claude-3-5-sonnet-20241022"  # Latest Sonnet model
    MAX_TOKENS = 4000
    TEMPERATURE = 0.3  # Lower for more consistent responses

# Data structures
@dataclass
class Character:
    name: str
    aliases: List[str]
    description: str
    relationships: Dict[str, str]
    first_appearance: Optional[str] = None
    last_mention: Optional[str] = None

@dataclass
class ReadingContext:
    book_title: str
    author: str
    current_chapter: Optional[str] = None
    current_page: Optional[int] = None
    progress_percentage: Optional[float] = None
    user_notes: Optional[str] = None

class QueryType(Enum):
    CHARACTER_LOOKUP = "character_lookup"
    CATCH_UP_SUMMARY = "catch_up_summary"
    RELATIONSHIP_MAP = "relationship_map"
    NICKNAME_RESOLUTION = "nickname_resolution"
    CHARACTER_EXTRACTION = "character_extraction"

class BookBrainAI:
    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or Config.ANTHROPIC_API_KEY
        )
    
    def _make_request(self, prompt: str, max_tokens: int = Config.MAX_TOKENS) -> str:
        """Make a request to Claude API with error handling"""
        try:
            response = self.client.messages.create(
                model=Config.MODEL,
                max_tokens=max_tokens,
                temperature=Config.TEMPERATURE,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"API Error: {e}")
            return f"Error: Unable to process request - {str(e)}"
    
    def character_lookup(self, character_name: str, context: ReadingContext) -> str:
        """Look up a specific character with spoiler protection"""
        prompt = f"""
You are BookBrain, an AI assistant that helps readers remember characters without spoilers.

Book: "{context.book_title}" by {context.author}
User's Current Position: {self._format_position(context)}
Character Query: "{character_name}"

Provide a character summary that includes:
1. Character's full name and any common nicknames/titles
2. Basic role/occupation in the story
3. Key relationships to main characters (as known up to user's current position)
4. Important personality traits or distinguishing features
5. Brief summary of their role in the plot SO FAR

CRITICAL: Do not include any information about events that happen after the user's current reading position. If there are future developments, simply note "appears in later chapters."

Keep the response concise but helpful - aim for 3-4 sentences that would jog the user's memory.
"""
        return self._make_request(prompt)
    
    def catch_up_summary(self, character_name: str, context: ReadingContext, 
                        user_last_memory: str = None) -> str:
        """Provide a 'catch me up' summary for when users return to a book"""
        last_memory_text = f"\nUser's last memory: \"{user_last_memory}\"" if user_last_memory else ""
        
        prompt = f"""
You are BookBrain, helping a reader who took a break from reading and needs to remember what was happening with a character.

Book: "{context.book_title}" by {context.author}
User's Current Position: {self._format_position(context)}
Character: {character_name}{last_memory_text}

Provide a catch-up summary that includes:
1. Brief reminder of who this character is (1 sentence)
2. What this character was doing in their most recent scenes
3. Current relationships/conflicts involving this character
4. What situation they were in when last seen
5. Any important character development up to this point

Keep it focused and conversational - like you're reminding a friend what happened in a show they stopped watching.
Avoid spoilers beyond their current reading position.
"""
        return self._make_request(prompt)
    
    def resolve_nickname(self, nickname_description: str, context: ReadingContext) -> str:
        """Resolve vague character descriptions to actual character names"""
        prompt = f"""
You are BookBrain, helping a reader identify a character from a vague description.

Book: "{context.book_title}" by {context.author}
User's Current Position: {self._format_position(context)}
User's Description: "{nickname_description}"

Based on the description, identify which character the user is likely referring to:
1. Provide the character's actual name
2. Explain why this character matches the description
3. Give 2-3 key identifying traits that confirm this is the right character
4. Briefly note their role in the story up to the user's current position

If multiple characters could match, list the most likely candidates with brief explanations.
Only consider characters that have appeared up to the user's reading position.
"""
        return self._make_request(prompt)
    
    def map_relationships(self, focus_character: str, context: ReadingContext) -> str:
        """Map character relationships for complex family trees or social networks"""
        prompt = f"""
You are BookBrain, helping a reader understand character relationships.

Book: "{context.book_title}" by {context.author}
User's Current Position: {self._format_position(context)}
Focus: Relationships involving "{focus_character}"

Create a relationship map that includes:
1. Direct family relationships (if applicable)
2. Close friends and allies
3. Enemies or rivals
4. Romantic interests (if any)
5. Professional/hierarchical relationships

Format as a clear list with relationship types:
- Family: [list family members and their relation]
- Allies: [list allies and why they're allied]
- Conflicts: [list conflicts and brief context]
- Other: [other significant relationships]

Only include relationships established up to the user's current reading position.
Keep descriptions brief but clear enough to understand the dynamic.
"""
        return self._make_request(prompt)
    
#     def extract_characters_from_text(self, book_text: str, book_info: ReadingContext) -> str:
#         """Extract and catalog characters from book text (for building initial database)"""
#         prompt = f"""
# You are BookBrain, analyzing book text to extract character information.

# Book: "{book_info.book_title}" by {book_info.author}
# Text to analyze: [First 3000 characters of provided text]

# Extract all named characters and provide:
# 1. Character name (full name if available)
# 2. Any nicknames, titles, or alternative names mentioned
# 3. Basic role/description based on the text
# 4. Key relationships mentioned in this text
# 5. Important traits or characteristics noted

# Format as JSON for easy parsing:
# {
#   "characters": [
#     {
#       "name": "Character Name",
#       "aliases": ["nickname1", "title"],
#       "description": "brief description",
#       "relationships": {"other_character": "relationship_type"},
#       "traits": ["trait1", "trait2"]
#     }
#   ]
# }

# Focus on characters with dialogue or significant mentions, not brief references.
# """
#         # Limit text to avoid token limits
#         limited_text = book_text[:8000] if len(book_text) > 8000 else book_text
#         full_prompt = prompt.replace("[First 3000 characters of provided text]", limited_text)
        
#         return self._make_request(full_prompt, max_tokens=6000)
    
    def _format_position(self, context: ReadingContext) -> str:
        """Format the user's reading position for prompts"""
        position_parts = []
        
        if context.current_chapter:
            position_parts.append(f"Chapter: {context.current_chapter}")
        if context.current_page:
            position_parts.append(f"Page: {context.current_page}")
        if context.progress_percentage:
            position_parts.append(f"Progress: {context.progress_percentage:.0f}%")
        
        return " | ".join(position_parts) if position_parts else "Beginning of book"

# Usage Examples
def main():
    # Initialize the AI client
    ai = BookBrainAI()  # Make sure to set your API key in Config
    
    # Example usage scenarios
    
    # 1. Basic character lookup
    context = ReadingContext(
        book_title="The Name of the Wind",
        author="Patrick Rothfuss",
        current_chapter="Chapter 45",
        progress_percentage=68
    )
    
    print("=== Character Lookup Example ===")
    response = ai.character_lookup("Kilvin", context)
    print(response)
    print("\n" + "="*50 + "\n")
    
    # 2. Nickname resolution
    print("=== Nickname Resolution Example ===")
    response = ai.resolve_nickname("the grumpy mentor guy who teaches magic", context)
    print(response)
    print("\n" + "="*50 + "\n")
    
    # 3. Catch-up summary
    print("=== Catch-Up Summary Example ===")
    response = ai.catch_up_summary(
        "Denna", 
        context, 
        "I remember she was a musician or something and Kvothe liked her"
    )
    print(response)
    print("\n" + "="*50 + "\n")
    
    # 4. Relationship mapping
    print("=== Relationship Mapping Example ===")
    response = ai.map_relationships("Kvothe", context)
    print(response)

if __name__ == "__main__":
    main()

# Additional utility functions for your backend

class CharacterDatabase:
    """Simple in-memory character database for POC"""
    
    def __init__(self):
        self.characters = {}  # book_id -> [Character objects]
        self.ai = BookBrainAI()
    
    def add_book_characters(self, book_id: str, book_text: str, book_info: ReadingContext):
        """Process a book and extract characters"""
        response = self.ai.extract_characters_from_text(book_text, book_info)
        try:
            data = json.loads(response)
            self.characters[book_id] = [
                Character(
                    name=char["name"],
                    aliases=char.get("aliases", []),
                    description=char.get("description", ""),
                    relationships=char.get("relationships", {})
                )
                for char in data.get("characters", [])
            ]
        except json.JSONDecodeError:
            print(f"Failed to parse character data for book {book_id}")
    
    def find_character_by_nickname(self, book_id: str, nickname: str) -> Optional[Character]:
        """Find character by nickname or partial name match"""
        if book_id not in self.characters:
            return None
        
        nickname_lower = nickname.lower()
        for character in self.characters[book_id]:
            if nickname_lower in character.name.lower():
                return character
            for alias in character.aliases:
                if nickname_lower in alias.lower():
                    return character
        return None

# Cost tracking utility
class CostTracker:
    """Track API usage costs for POC budgeting"""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def log_request(self, input_text: str, output_text: str):
        """Log token usage for cost tracking"""
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.request_count += 1
    
    def get_estimated_cost(self) -> float:
        """Calculate estimated cost based on Sonnet pricing"""
        input_cost = (self.total_input_tokens / 1_000_000) * 3.0  # $3 per million
        output_cost = (self.total_output_tokens / 1_000_000) * 15.0  # $15 per million
        return input_cost + output_cost
    
    def print_stats(self):
        """Print usage statistics"""
        print(f"Total Requests: {self.request_count}")
        print(f"Input Tokens: {self.total_input_tokens:,}")
        print(f"Output Tokens: {self.total_output_tokens:,}")
        print(f"Estimated Cost: ${self.get_estimated_cost():.4f}")