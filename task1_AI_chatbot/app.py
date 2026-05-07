import json
import logging
import random
import re
import datetime
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

# Set up logging
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Custom Rich Console Theme for a professional look
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "user": "bold green",
    "bot": "bold blue"
})
console = Console(theme=custom_theme)

class ProfessionalChatBot:
    def __init__(self, intents_file='intents.json'):
        self.intents = []
        self.memory = {
            "user_name": None
        }
        self.load_intents(intents_file)
        logging.info("ChatBot initialized successfully.")

    def load_intents(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.intents = data.get('intents', [])
        except FileNotFoundError:
            console.print(f"[danger]Error: Knowledge base '{filepath}' not found.[/danger]")
            logging.error(f"Intents file '{filepath}' missing.")
            self.intents = []

    def save_memory(self, key, value):
        self.memory[key] = value
        logging.info(f"Memory updated: {key} = {value}")

    def evaluate_math(self, expression):
        # Extremely basic and safe math evaluation using regex
        try:
            # Strip out anything that isn't a digit or basic operator
            clean_expr = re.sub(r'[^\d+\-*/().]', '', expression)
            if clean_expr:
                result = eval(clean_expr)
                return f"The result is {result}."
        except Exception as e:
            logging.warning(f"Failed to evaluate math expression: {expression} - {e}")
        return "I couldn't calculate that. Please provide a valid math expression."

    def extract_intent(self, user_input):
        highest_score = 0
        best_intent = None

        for intent in self.intents:
            for pattern in intent['patterns']:
                score = fuzz.token_set_ratio(user_input.lower(), pattern.lower())
                if score > highest_score:
                    highest_score = score
                    best_intent = intent
        
        # Threshold for considering it a match
        if highest_score > 70:
            return best_intent
        return None

    def handle_skills(self, user_input):
        text = user_input.lower()
        
        # Name memory skill
        if "my name is" in text:
            name_match = re.search(r"my name is (\w+)", text)
            if name_match:
                name = name_match.group(1).capitalize()
                self.save_memory("user_name", name)
                return f"Nice to meet you, {name}! I'll remember that."

        # Return user's name if asked
        if "what is my name" in text or "what's my name" in text:
            name = self.memory.get("user_name")
            if name:
                return f"Your name is {name}, if I remember correctly."
            else:
                return "You haven't told me your name yet! You can say 'My name is...'"

        # Calculator skill
        if "calculate" in text or "math" in text:
            # Extract everything after calculate
            parts = text.split("calculate")
            if len(parts) > 1:
                return self.evaluate_math(parts[1])

        # Date/Time skill
        if "time" in text:
            now = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {now}."
        
        if "date" in text:
            today = datetime.datetime.now().strftime("%A, %B %d, %Y")
            return f"Today's date is {today}."

        return None

    def get_response(self, user_input):
        logging.info(f"User: {user_input}")
        
        # Check specific skills first
        skill_response = self.handle_skills(user_input)
        if skill_response:
            logging.info(f"Bot (Skill): {skill_response}")
            return skill_response

        # Use fuzzy matching NLP for intent classification
        intent = self.extract_intent(user_input)
        if intent:
            response = random.choice(intent['responses'])
            # Personalize response if we know the user's name and it's a greeting
            if intent['tag'] == 'greeting' and self.memory.get('user_name'):
                response = f"{response} Nice to see you again, {self.memory.get('user_name')}!"
            
            logging.info(f"Bot: {response}")
            return response

        # Fallback response
        fallback = "I'm sorry, I didn't quite catch that. Could you rephrase?"
        logging.info(f"Bot: {fallback}")
        return fallback

def main():
    console.clear()
    welcome_text = Text("Nexus AI Assistant", justify="center", style="bold white on blue")
    console.print(Panel(welcome_text, border_style="blue"))
    console.print("[info]Type 'help' to see what I can do. Type 'bye' or 'quit' to exit.[/info]\n")

    bot = ProfessionalChatBot()

    while True:
        try:
            user_input = console.input("[user]You: [/user]").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["bye", "exit", "quit", "cya"]:
                farewell = bot.get_response("bye")
                console.print(f"[bot]Nexus:[/bot] {farewell}")
                break

            response = bot.get_response(user_input)
            console.print(f"[bot]Nexus:[/bot] {response}")

        except KeyboardInterrupt:
            console.print("\n[bot]Nexus:[/bot] Goodbye!")
            break
        except Exception as e:
            console.print(f"[danger]An unexpected error occurred: {e}[/danger]")
            logging.error(f"Runtime error: {e}")

if __name__ == "__main__":
    main()
