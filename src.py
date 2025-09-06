import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.clock import Clock # Import Clock for scheduling updates

import sqlite3
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import requests
from bs4 import BeautifulSoup
import re
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer
import ratelimit
from cryptography.fernet import Fernet
import datetime
from googletrans import Translator
import pyttsx3
import speech_recognition as sr # For voice input

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('sentiment/vader_lexicon')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

# Ensure Kivy is compatible
kivy.require('2.0.0') # Adjust as per your Kivy version

class Chatbot(BoxLayout):
    def __init__(self, **kwargs):
        super(Chatbot, self).__init__(**kwargs)
        self.orientation = 'vertical'

        # Chat display area
        self.chat_area = ScrollView(size_hint=(1, 0.7)) # Adjusted size_hint
        self.chat_log_label = Label(text='', size_hint_y=None, markup=True, halign='left', valign='top', padding_x='10dp')
        self.chat_log_label.bind(texture_size=self._set_chat_log_label_height)
        self.chat_area.add_widget(self.chat_log_label)
        self.add_widget(self.chat_area)

        # Input and action buttons layout
        input_layout = BoxLayout(size_hint=(1, 0.3), orientation='vertical')

        self.input_field = TextInput(size_hint=(1, 0.3), multiline=False, hint_text='Type your message here...')
        self.input_field.bind(on_text_validate=self.send_message) # Send on Enter key
        input_layout.add_widget(self.input_field)

        button_layout = BoxLayout(size_hint=(1, 0.7)) # To hold send, voice, feedback buttons

        self.send_button = Button(text='Send', size_hint=(0.33, 1))
        self.send_button.bind(on_press=self.send_message)
        button_layout.add_widget(self.send_button)

        self.voice_button = Button(text='Voice Input', size_hint=(0.33, 1))
        self.voice_button.bind(on_press=self.voice_input)
        button_layout.add_widget(self.voice_button)

        self.feedback_button = Button(text='Feedback', size_hint=(0.34, 1))
        self.feedback_button.bind(on_press=self.provide_feedback)
        button_layout.add_widget(self.feedback_button)

        input_layout.add_widget(button_layout)
        self.add_widget(input_layout)

        # Database initialization
        self.db_path = 'chatbot.db'
        self.init_database()

        # NLP and other tool initialization
        self.model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
        self.tokenizer = AutoTokenizer.from_pretrained('t5-base')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        self.sia = SentimentIntensityAnalyzer()
        self.translator = Translator()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150) # Speed of speech

        self.user_id = self.get_or_create_user() # Assign a user ID
        self.user_count = self.get_user_count()
        self.last_request_time = time.time() # For rate limiting web requests

        # Encryption key management
        self.key_file = 'secret.key'
        self.key = self._load_or_generate_key()
        self.cipher_suite = Fernet(self.key)

        # Load existing knowledge for TF-IDF
        self.knowledge_base_data = self.load_knowledge_base_from_db()
        self.knowledge_vectors = None
        self.knowledge_questions = []
        if self.knowledge_base_data:
            self.knowledge_questions = [item[0] for item in self.knowledge_base_data]
            if self.knowledge_questions:
                self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge_questions)

        self.add_message_to_chat("Chatbot", "Hello! How can I assist you today?")

    def _set_chat_log_label_height(self, instance, value):
        instance.height = max(instance.texture_size[1], self.chat_area.height)
        self.chat_area.scroll_y = 0 # Scroll to bottom automatically

    def init_database(self):
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS conversation (timestamp TEXT, role TEXT, message TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS automation (task TEXT, action TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS code_snippets (name TEXT, code TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, classification TEXT, interaction_data TEXT, created_at TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS feedback (timestamp TEXT, user_id TEXT, feedback_text TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS knowledge (question TEXT, answer TEXT)''')
            conn.commit()
            conn.close()

    def _load_or_generate_key(self):
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            return key

    def encrypt_data(self, data):
        return self.cipher_suite.encrypt(data.encode()).decode() # Store as string

    def decrypt_data(self, encrypted_data):
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()

    def get_or_create_user(self):
        # A very basic user ID generation. In a real app, this would be more robust.
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Check if there's an existing user, or create a new one
        c.execute("SELECT id FROM users ORDER BY created_at DESC LIMIT 1")
        last_user = c.fetchone()
        if last_user:
            user_id = last_user[0]
        else:
            user_id = f"user_{int(time.time())}"
            c.execute("INSERT INTO users (id, classification, interaction_data, created_at) VALUES (?, ?, ?, ?)",
                      (user_id, "new_user", self.encrypt_data("{}"), datetime.datetime.now().isoformat()))
            conn.commit()
        conn.close()
        return user_id

    def get_user_count(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users")
        count = c.fetchone()[0]
        conn.close()
        return count

    @ratelimit.limits(calls=10, period=60)
    def rate_limited_get_url_content(self, url):
        try:
            # Simple rate limiting logic for external requests if not using a decorator
            current_time = time.time()
            if current_time - self.last_request_time < 1: # Enforce 1 second minimum between requests
                time.sleep(1 - (current_time - self.last_request_time))
            self.last_request_time = time.time()

            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10) # Added timeout
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract visible text, avoiding scripts and styles
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
        except requests.exceptions.RequestException as e:
            return f"Error accessing URL: {e}"
        except Exception as e:
            return f"An unexpected error occurred during scraping: {e}"

    def learn_from_feedback(self, feedback):
        timestamp = datetime.datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO feedback (timestamp, user_id, feedback_text) VALUES (?, ?, ?)",
                      (timestamp, self.user_id, self.encrypt_data(feedback)))
            conn.commit()
            self.add_message_to_chat("Chatbot", "Thank you for your feedback! I'll try to learn from it.")
            # In a real system, you would trigger a model retraining or knowledge base update here.
            # For this example, we'll just log it.
        except Exception as e:
            self.add_message_to_chat("Chatbot", f"Error saving feedback: {e}")
        finally:
            conn.close()

    def update_knowledge_base(self, text):
        if not text:
            return

        # Simple approach: assume text contains question-answer pairs
        # A more advanced approach would use NLP to extract Q&A from free text
        # For demonstration, let's treat lines as potential knowledge
        lines = text.split('\n')
        new_knowledge_items = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 10: # Only consider meaningful lines
                # Simple heuristic: if a line ends with '?', it's a question, next line is answer
                if line.endswith('?'):
                    question = line
                    # This needs a more robust way to find an answer.
                    # For now, let's just make a generic answer or skip if no clear answer follows.
                    answer = "I've noted this question."
                    new_knowledge_items.append((question, answer))
                else:
                    # Treat general statements as potential knowledge, can be expanded
                    question = "What is " + line.split(' ')[0] if len(line.split(' ')) > 1 else "Info about " + line
                    answer = line
                    new_knowledge_items.append((question, answer))

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        for question, answer in new_knowledge_items:
            # Check for duplicates before inserting
            c.execute("SELECT COUNT(*) FROM knowledge WHERE question = ? AND answer = ?", (question, answer))
            if c.fetchone()[0] == 0:
                c.execute("INSERT INTO knowledge (question, answer) VALUES (?, ?)", (question, answer))
        conn.commit()
        conn.close()

        # Reload knowledge for TF-IDF
        self.knowledge_base_data = self.load_knowledge_base_from_db()
        self.knowledge_questions = [item[0] for item in self.knowledge_base_data]
        if self.knowledge_questions:
            self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge_questions)
        self.add_message_to_chat("Chatbot", "Knowledge base updated with new information.")

    def load_knowledge_base_from_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT question, answer FROM knowledge")
        data = c.fetchall()
        conn.close()
        return data

    def find_relevant_knowledge(self, query):
        if not self.knowledge_vectors or not self.knowledge_questions:
            return None, 0.0

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.knowledge_vectors).flatten()
        max_similarity_index = similarities.argmax()
        max_similarity_score = similarities[max_similarity_index]

        if max_similarity_score > 0.6: # Threshold for considering a match
            return self.knowledge_base_data[max_similarity_index][1], max_similarity_score
        return None, 0.0

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in self.stop_words]
        return " ".join(lemmas)

    def generate_response(self, user_input):
        # 1. Check for specific commands/patterns first
        user_input_lower = user_input.lower()

        if "hello" in user_input_lower or "hi" in user_input_lower:
            return "Hello there! How can I help you today?"
        if "how are you" in user_input_lower:
            return "I'm just a program, but I'm doing great! Thanks for asking."
        if "what is your name" in user_input_lower:
            return "I am a Kivy Chatbot, designed to assist you."
        if "tell me a joke" in user_input_lower:
            return "Why don't scientists trust atoms? Because they make up everything!"
        if "what time is it" in user_input_lower:
            return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."
        if "what is the date" in user_input_lower:
            return f"Today's date is {datetime.datetime.now().strftime('%Y-%m-%d')}."
        if user_input_lower.startswith("scrape "):
            url = user_input[len("scrape "):].strip()
            if url:
                self.add_message_to_chat("Chatbot", f"Scraping content from {url}...")
                # Schedule the scraping to avoid blocking the GUI
                Clock.schedule_once(lambda dt: self._perform_scraping_and_update_knowledge(url), 0.1)
                return "Please wait while I fetch information from the web."
            else:
                return "Please provide a URL to scrape."

        # 2. Look for answers in the knowledge base (TF-IDF)
        processed_input = self.preprocess_text(user_input)
        knowledge_answer, score = self.find_relevant_knowledge(processed_input)
        if knowledge_answer:
            self.add_message_to_chat("Chatbot", f"(Knowledge match, score: {score:.2f})")
            return knowledge_answer

        # 3. Use the T5 model for more general responses if no direct match
        # This part requires fine-tuning T5 or providing specific prompts.
        # For a simple Q&A, you'd usually have a dataset to train T5 on.
        # Here, it will act as a general text generator.
        inputs = self.tokenizer.encode("question: " + user_input + " context: " + self.chat_log_label.text[-200:],
                                       return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
        model_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not model_response or model_response.strip() == "":
            return "I'm not sure how to respond to that. Could you rephrase?"

        return model_response

    def _perform_scraping_and_update_knowledge(self, url):
        # This function runs on a separate thread/process in a real app or with Clock.schedule_once
        # For now, it will block the GUI briefly if the scraping takes time.
        scraped_text = self.rate_limited_get_url_content(url)
        if scraped_text.startswith("Error"):
            self.add_message_to_chat("Chatbot", scraped_text)
        else:
            self.add_message_to_chat("Chatbot", f"Successfully scraped content from {url}. Updating knowledge base...")
            self.update_knowledge_base(scraped_text)

    def add_message_to_chat(self, sender, message):
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        if sender == "You":
            formatted_message = f"[b]{sender} ({timestamp}):[/b] {message}\n"
        else:
            formatted_message = f"[color=0000ff][b]{sender} ({timestamp}):[/b][/color] {message}\n"
        self.chat_log_label.text += formatted_message
        # Store in conversation history
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO conversation (timestamp, role, message) VALUES (?, ?, ?)",
                  (datetime.datetime.now().isoformat(), sender, self.encrypt_data(message)))
        conn.commit()
        conn.close()

    def send_message(self, instance):
        user_message = self.input_field.text.strip()
        self.input_field.text = '' # Clear input field

        if not user_message:
            return

        self.add_message_to_chat("You", user_message)

        # Get chatbot response
        chatbot_response = self.generate_response(user_message)
        self.add_message_to_chat("Chatbot", chatbot_response)

        # Perform sentiment analysis on user's message
        sentiment_score = self.sia.polarity_scores(user_message)
        if sentiment_score['compound'] >= 0.05:
            sentiment = "positive"
        elif sentiment_score['compound'] <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        # self.add_message_to_chat("Chatbot", f"(Sentiment: {sentiment})") # For debugging sentiment

        # Optionally speak the response
        try:
            self.engine.say(chatbot_response)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error speaking: {e}")

    def voice_input(self, instance):
        self.add_message_to_chat("Chatbot", "Listening for your voice...")
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            self.input_field.text = text
            self.add_message_to_chat("Chatbot", f"You said: {text}")
            self.send_message(self.send_button) # Automatically send the recognized text
        except sr.UnknownValueError:
            self.add_message_to_chat("Chatbot", "Sorry, I could not understand audio.")
        except sr.RequestError as e:
            self.add_message_to_chat("Chatbot", f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            self.add_message_to_chat("Chatbot", f"An error occurred during voice input: {e}")

    def provide_feedback(self, instance):
        def submit_feedback(popup_instance, feedback_text_input):
            feedback = feedback_text_input.text.strip()
            if feedback:
                self.learn_from_feedback(feedback)
                popup_instance.dismiss()
            else:
                self.add_message_to_chat("Chatbot", "Feedback cannot be empty.")

        content = BoxLayout(orientation='vertical')
        feedback_input = TextInput(multiline=True, hint_text='Type your feedback here...')
        content.add_widget(feedback_input)
        submit_button = Button(text='Submit Feedback', size_hint_y=0.2)
        content.add_widget(submit_button)

        popup = Popup(title='Provide Feedback', content=content, size_hint=(0.8, 0.6))
        submit_button.bind(on_press=lambda x: submit_feedback(popup, feedback_input))
        popup.open()


class ChatbotApp(App):
    def build(self):
        return Chatbot()

if __name__ == '__main__':
    ChatbotApp().run()
