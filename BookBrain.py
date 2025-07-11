import anthropic
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import os


app = Flask(__name__)
CORS(app)
load_dotenv()
UPLOAD_FOLDER = "./uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configuration
class Config:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    MODEL = "claude-sonnet-4-20250514"  # Latest Sonnet model
    MAX_TOKENS = 4000
    TEMPERATURE = 0.3 # Lower for more consistent responses

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
                system=(
                "You are BookBrain, a helpful, spoiler-aware AI assistant who helps users recall information from books. "
                "You only answer based on chapters the user has already read. Use a friendly, concise tone. Avoid spoilers."
            ),
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text
        except Exception as e:
            print(f"API Error: {e}")
            return f"Error: Unable to process request - {str(e)}"
    
    def continue_conversation(self, conversation_history: list, max_tokens: int = Config.MAX_TOKENS) -> str:
        """Handle multi-turn conversations with memory (chat mode)"""
        try:
            response = self.client.messages.create(
                model=Config.MODEL,
                max_tokens=max_tokens,
                temperature=Config.TEMPERATURE,
                system=(
                    "You are BookBrain, a helpful, spoiler-aware AI assistant who helps users recall information from books. "
                    "You only answer based on chapters the user has already read. Use a friendly, concise tone. Avoid spoilers."
                ),
                messages=conversation_history
            )
            return response.content[0].text
        except Exception as e:
            print(f"API Error (chat): {e}")
            return f"Error: Unable to process conversation - {str(e)}"
        
    def character_lookup(self, character_name: str, context: ReadingContext) -> str:
        """Look up a specific character with spoiler protection"""
        prompt = (
            f"You're an intelligent reading assistant helping a user recall information from a book they are reading. "
            f"They are currently in Chapter {context.current_chapter}, Page {context.current_page}, "
            f"of '{context.book_title}' by {context.author}. Their question is: "
            f"Who is {character_name}?\n\n"
            "Keep the tone friendly and clear, and avoid spoilers for future chapters if possible. "
            "Ensure that there are no hallucinations. Take your time and provide an accurate answer "
            "without asking any follow-up questions.\n\n"
            "Also, include a confidence rating (e.g. High, Medium, Low) based on how certain you are of the answer."
        )
        print(prompt)
        return self._make_request(prompt)
    
    def catch_up_summary(self, context: ReadingContext, 
                        user_last_memory: str = None) -> str:
        """Provide a 'catch me up' summary for when users return to a book"""

        # prompt = f"""You're an intelligent reading assistant helping a user recall information from a book they are reading. They are currently in Chapter {context.current_chapter}, Page {context.current_page}, of {context.book_title} by {context.author}. Create a quick paragraph summary of what happened in the last chapter."

        # Keep the tone friendly and clear, and avoid spoilers for future chapters. Take your time and provide an accurate answer without asking any follow-up questions. Provide a confidence rating."""

        prompt = f"""
            You are BookBrain, an intelligent reading assistant helping a user recall information from a book they are reading.

            They are currently on Chapter {context.current_chapter}, Page {context.current_page}, of *{context.book_title}* by {context.author}. 
            The user has just finished reading the previous chapter.

            Please generate a **brief, spoiler-free** summary of ONLY the last completed chapter — using information strictly up to this point in the book.

            ⚠️ Do not include events, characters, or developments from later chapters or pages. 
            If you are unsure or lack context, say so politely instead of guessing.

            Use a friendly and clear tone.
            Do not ask any follow-up questions.
            Avoid speculation and hallucination.
            Respond with a confidence rating (e.g. High, Medium, Low) at the end.
            """
        print(prompt)

        return self._make_request(prompt)
    
    def voice_question(self, context: ReadingContext, voice_info: str):

        prompt = f"""You're an intelligent reading assistant helping a user recall information from a book they are reading. They are currently in Chapter {context.current_chapter}, Page {context.current_page}, of {context.book_title} by {context.author}. Their question is: "{voice_info}"

        Keep the tone friendly and clear, and avoid spoilers for future chapters if possible. Ensure that there ar no halluciantions. Take your time and provide an accurate answer without asking any follow-up questions. Provide a confidence rating. This should reflect how confident you feel about the answer provided"""
        print(prompt)
        return self._make_request(prompt)

    def chat_question(self, context: ReadingContext, general_info: str, conversation_history: list) -> str:
        """Handles a new chat question from the user"""
        
        # Add the current user question as a new message
        conversation_history.append({
            "role": "user",
            "content": f"""You're an intelligent reading assistant helping a user recall information from a book they are reading. 
            They are currently in Chapter {context.current_chapter}, Page {context.current_page}, of {context.book_title} by {context.author}. 
            Their question is: "{general_info}"

            Keep the tone friendly and clear, and avoid spoilers for future chapters if possible. Ensure that there are no hallucinations. 
            Take your time and provide an accurate answer without asking any follow-up questions. 
            Provide a confidence rating that reflects how confident you feel about the answer provided."""
        })

        # Pass the full message list into the Claude chat
        return self.continue_conversation(conversation_history)
    
    def resolve_nickname(self, nickname_description: str, context: ReadingContext) -> str:
        """Resolve vague character descriptions to actual character names"""
        prompt = f"""
        You are BookBrain, helping a reader identify a character from a vague description.

        Book: "{context.book_title}" by {context.author}
        User's Current Position: {self._format_position(context)}
        User's Description: "{nickname_description}"

        return self._make_request(prompt)"""
    
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
    
    def create_comprehension_quizzes(self, context: ReadingContext) -> str:
        """Create comprehension quizzes for the user"""
        prompt = f"""You're an intelligent reading assistant helping a user recall information from a book they are reading. 
            The user is currently in Chapter {context.current_chapter}, Page {context.current_page}, of "{context.book_title}" by {context.author}.

            Create a fun and easy reading comprehension quiz for them based only on what has happened so far.

            Requirements:
            - Provide 7-10 multiple choice questions.
            - Each question must include:
                - "question": the question text
                - "options": a list of 4 possible answers
                - "answer": the correct answer (must match one of the options exactly)
                - "explanation": a brief reason why this is the correct answer

            ⚠️ Rules:
            - Abosolutely avoid spoilers from future chapters. This should be a priority.
            - Do not ask follow-up questions.
            - Ensure there are no hallucinations.
            - Be concise, friendly, and accurate.

            Return your response in **valid JSON** with the following format:

            {{
            "questions": [
                {{
                "question": "...",
                "options": ["...", "...", "...", "..."],
                "answer": "...",
                "explanation": "..."
                }},
                ...
            ]
            }}

            Output only the JSON object and nothing else.
            """
        return self._make_request(prompt)

    def preset_characters_in_character_gallery(self, context: ReadingContext) -> str:
        """Add characters to the character gallery"""
        prompt = (
            f"You're an intelligent reading assistant helping display main characters from a book the user is reading. "
            f"They are currently in Chapter {context.current_chapter}, Page {context.current_page}, "
            f"of '{context.book_title}' by {context.author}.\n\n"
            "Extract all the main characters introduced up to this point and provide their details.\n"
            "For each character, include the following fields:\n"
            "- name: Full name if available\n"
            "- nicknames: Any nicknames, titles, or alternative names mentioned\n"
            "- description: A basic 1-sentence role or description\n"
            "- traits: 2 to 3 key personality or physical traits\n"
            "- relationships: Brief notes on their relationships with other characters so far\n"
            "- last_appearance: The last chapter or scene they were mentioned\n"
            "- confidence: Confidence rating (High, Medium, Low)\n\n"

            "⚠️ Important:\n"
            "- Do not include spoilers for future chapters\n"
            "- Only mention what has been revealed so far in the book\n"
            "- Ensure all output is grounded in the book's content (no hallucinations)\n"
            "- Avoid follow-up questions\n\n"

            "Respond with only a valid JSON object in the following format:\n"
            "{\n"
            "  \"characters\": [\n"
            "    {\n"
            "      \"name\": \"...\",\n"
            "      \"nicknames\": [\"...\", \"...\"],\n"
            "      \"description\": \"...\",\n"
            "      \"traits\": [\"...\", \"...\"],\n"
            "      \"relationships\": \"...\",\n"
            "      \"last_appearance\": \"Chapter X, Page Y\",\n"
            "      \"confidence\": \"High\"\n"
            "    },\n"
            "    ...\n"
            "  ]\n"
            "}"
        )
        return self._make_request(prompt)

    # def preset_characters_in_character_gallery(self, context: ReadingContext) -> str:
    #     """Add characters to the character gallery"""
    #     prompt = (
    #         f"You're an intelligent reading assistant helping display main characters from a book they are reading. "
    #         f"They are currently in Chapter {context.current_chapter}, Page {context.current_page}, "
    #         f"of '{context.book_title}' by {context.author}.\n\n"
    #         "Extract all the main characters up to this point in the book and provide:\n"
    #         "1. Character name (full name if available)\n"
    #         "2. Any nicknames, titles, or alternative names mentioned\n"
    #         "3. Basic role/description in a sentence\n"
    #         "4. Important 2-3 traits or characteristics noted\n"
    #         "5. Key relationships with other characters mentioned up to this point\n"
    #         "Keep the tone friendly and clear, and avoid spoilers for future chapters if possible. "
    #         "Ensure that there are no hallucinations. Take your time and provide an accurate answer "
    #         "without asking any follow-up questions.\n\n"
    #         "Also, include a confidence rating (e.g., High, Medium, Low) based on how certain you are of the answer."
    #     )
    #     return self._make_request(prompt)
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


def obtain_and_create_progress(data):
    # TODO: make current page optional
    # required_fields = ["character_name", "book_title", "author", "current_chapter", "current_page"]
    required_fields = ["book_title", "author", "current_chapter", "current_page"]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
    
    context = ReadingContext(
        book_title=data["book_title"],
        author=data["author"],
        current_chapter=data["current_chapter"],
        current_page=data["current_page"]
    )

    return context

# Usage Examples
@app.route("/recall", methods=["POST"])
def recall():
    try:
        data = request.get_json()

        context = obtain_and_create_progress(data)

         # Initialize the AI client
        ai = BookBrainAI()  # Make sure to set your API key in Config
        print("=== Character Lookup Example ===")
        answer = ai.character_lookup(data["character_name"], context)
        print(answer)
        return jsonify({"answer": answer})

        
        # # 2. Nickname resolution
        # Add Nickname is in the future and then customer can fetch it
        # print("=== Nickname Resolution Example ===")
        # response = ai.resolve_nickname("the grumpy mentor guy who teaches magic", context)
        # print(response)
        # print("\n" + "="*50 + "\n")
        
        
        # # 4. Relationship mapping
        # print("=== Relationship Mapping Example ===")
        # response = ai.map_relationships("Kvothe", context)
        # print(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/summary", methods=["POST"])
def summary():
    try:
        data = request.get_json()
        # Validate input
        context = obtain_and_create_progress(data)
         # Initialize the AI client
        ai = BookBrainAI()  # Make sure to set your API key in Config
        print("=== Catch me up Summary ===")
        # response = ai.character_lookup("Taffa", context)
        answer = ai.catch_up_summary(context)
        print(answer)
        return jsonify({"answer": answer})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/comprehension_quiz", methods=["POST"])
def comprehension_quiz():
    try:
        data = request.get_json()

        context = obtain_and_create_progress(data)
         # Initialize the AI client
        ai = BookBrainAI()  # Make sure to set your API key in Config
        print("=== Comprehension Quiz ===")
        # response = ai.character_lookup("Taffa", context)
        answer = ai.create_comprehension_quizzes(context)
        print(answer)
        return jsonify({"answer": answer})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/voice-assistant", methods=["POST"])
def voice_assistant_with_context():
    try:
        data = request.get_json()
        # Validate input
        context = obtain_and_create_progress(data)
         # Initialize the AI client
        ai = BookBrainAI()  # Make sure to set your API key in Config
        print("=== Voice Assistant ===")
        # response = ai.character_lookup("Taffa", context)
        answer = ai.voice_question(context, data["voice_string"])
        print(answer)
        return jsonify({"answer": answer})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    try:
        file = request.files.get('pdf')
        print(file)
        if file and file.filename.endswith(".pdf"):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return {"status": "success", "path": filepath}, 200
        
        return {"error": "No PDF file provided"}, 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/preset_characters_to_gallery", methods=["POST"])
def preset_character_gallery():
    try:
        data = request.get_json()
        # Validate input
        context = obtain_and_create_progress(data)
         # Initialize the AI client
        ai = BookBrainAI()  # Make sure to set your API key in Config
        print("=== Voice Assistant ===")
        # response = ai.character_lookup("Taffa", context)
        answer = ai.preset_characters_in_character_gallery(context)
        print(answer)
        return jsonify({"answer": answer})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# In-memory storage for demo purposes
user_sessions = {}

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        message = data.get("message")
        conversation_history = data.get("conversation_history", [])
        context = obtain_and_create_progress(data)
       
        conversation_history.append({
            "role": "user",
            "content": message
        })

        ai = BookBrainAI()
        print("=== Chat question ===")
        answer = ai.chat_question(context, message, conversation_history)
        print(answer)
        
        conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        return jsonify({
            "answer": answer,
            "conversation_history": conversation_history
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)

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