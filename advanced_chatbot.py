import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import requests
from bs4 import BeautifulSoup
import re
import time
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import json

import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.utils import get_color_from_hex

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment import SentimentIntensityAnalyzer

import ratelimit
from cryptography.fernet import Fernet
from deep_translator import GoogleTranslator
import pyttsx3
import speech_recognition as sr

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import hashlib
import secrets
import logging

# --- NLTK Data Download (Ensures all necessary data is present) ---
def download_nltk_data():
    required_data = [('corpora/wordnet', 'wordnet'), ('corpora/stopwords', 'stopwords'),
                     ('tokenizers/punkt', 'punkt'), ('sentiment/vader_lexicon', 'vader_lexicon')]
    for path, pkg_id in required_data:
        try:
            nltk.data.find(path)
        except nltk.downloader.DownloadError:
            nltk.download(pkg_id, quiet=True)
download_nltk_data()

MIN_KIVY_VERSION = '2.2.1'
def check_kivy_version():
    current_version = kivy.__version__
    if current_version < MIN_KIVY_VERSION:
        print(f"Kivy version {current_version} is outdated. Recommended: {MIN_KIVY_VERSION} or higher.")
    else:
        print(f"Kivy version {current_version} is sufficient.")
check_kivy_version()
kivy.require(MIN_KIVY_VERSION)

DB_CONFIG = {
    'dbname': 'chatbot_db',
    'user': 'chatbot_user',
    'password': 'your_strong_db_password',
    'host': 'localhost',
    'port': '5432'
}

KEY_FILE = 'secret.key'
def _load_or_generate_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, 'rb') as f:
            key = f.read()
            return key
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as f:
            f.write(key)
        return key

FERNET_KEY = _load_or_generate_key()
CIPHER_SUITE = Fernet(FERNET_KEY)

THREAT_PATTERNS = [
    r"delete database", r"format hard drive", r"access camera",
    r"exploit vulnerability", r"root access", r"malware", r"phishing",
    r"(?:select|insert|update|delete)\s.+\s(?:from|into|where)", # SQLi detection
    r"<script.*?>.*?</script.*?>", # XSS detection
    r"['\"].*?--", # SQL comments
    r"union\s+select", # SQLi Union
    r"onerror\s*=", r"javascript:", r"src\s*=", # XSS vectors
]

class IntrusionDetection:
    """Basic intrusion detection using regex for request logging and anomaly detection."""
    def __init__(self):
        self.suspicious_events = []

    def check_event(self, text):
        for pattern in THREAT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                self.suspicious_events.append((datetime.datetime.now(), text))
                return True
        return False

    def get_events(self):
        return self.suspicious_events

intrusion_system = IntrusionDetection()

def sanitize_input(user_input):
    """Sanitize input to prevent XSS/SQLi, removes suspicious elements."""
    user_input = re.sub(r"<.*?>", "", user_input) # Remove tags
    user_input = re.sub(r"(?:')|(?:--)|(?:#)", "", user_input) # Remove certain SQLi patterns
    user_input = re.sub(r"(?i)union", "", user_input)
    user_input = re.sub(r"(?i)select|insert|update|delete|drop|alter", "", user_input)
    return user_input

class Chatbot(BoxLayout):
    def __init__(self, **kwargs):
        super(Chatbot, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = '10dp'
        self.spacing = '10dp'

        self.chat_area = ScrollView(size_hint=(1, 0.85))
        self.chat_log_label = Label(
            text='', size_hint_y=None, markup=True, halign='left', valign='top',
            padding_x='10dp', font_size='16sp', color=get_color_from_hex('#E0E0E0')
        )
        self.chat_log_label.bind(texture_size=self._set_chat_log_label_height)
        self.chat_area.add_widget(self.chat_log_label)
        self.add_widget(self.chat_area)

        input_layout = BoxLayout(size_hint=(1, 0.15), spacing='5dp')

        self.input_field = TextInput(
            size_hint=(0.7, 1), multiline=False, hint_text='Type your message...',
            font_size='16sp', padding_y='10dp', background_color=get_color_from_hex('#3C3C3C'),
            foreground_color=get_color_from_hex('#FFFFFF')
        )
        self.input_field.bind(on_text_validate=self.send_message)
        input_layout.add_widget(self.input_field)

        button_layout = BoxLayout(orientation='vertical', size_hint=(0.3, 1), spacing='5dp')
        top_button_row = BoxLayout(size_hint=(1, 0.5), spacing='5dp')
        self.send_button = Button(text='Send', on_press=self.send_message, font_size='14sp')
        self.voice_button = Button(text='🎤', on_press=self.voice_input, font_size='20sp', background_color=get_color_from_hex('#4CAF50'))
        top_button_row.add_widget(self.send_button)
        top_button_row.add_widget(self.voice_button)
        bottom_button_row = BoxLayout(size_hint=(1, 0.5), spacing='5dp')
        self.translate_button = Button(text='Translate', on_press=self.show_translation_options, font_size='14sp')
        self.feedback_button = Button(text='Feedback', on_press=self.provide_feedback, font_size='14sp')
        bottom_button_row.add_widget(self.translate_button)
        bottom_button_row.add_widget(self.feedback_button)
        button_layout.add_widget(top_button_row)
        button_layout.add_widget(bottom_button_row)
        input_layout.add_widget(button_layout)
        self.add_widget(input_layout)

        self.init_database()
        self.user_id = self.get_or_create_user()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained('t5-base').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('t5-base')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(preprocessor=self.preprocess_text)
        self.sia = SentimentIntensityAnalyzer()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 170)
        self.knowledge_base_data = []
        self.knowledge_vectors = None
        self.knowledge_questions = []
        self._load_knowledge_from_db_and_vectorize()
        self.add_message_to_chat("Chatbot", "Hello! I am ready to assist you. (Type '/help' for commands)")

        # Intrusion Detection
        self.intrusion_alerted = False

        # Model retraining scheduling
        self.last_retrain_time = time.time()
        self.model_retrain_interval = 3600 # retrain every hour by default

    def _set_chat_log_label_height(self, instance, value):
        instance.height = max(instance.texture_size[1], self.chat_area.height)
        self.chat_area.scroll_y = 0

    def get_db_connection(self, dbname=DB_CONFIG['dbname']):
        try:
            conn = psycopg2.connect(dbname=dbname, user=DB_CONFIG['user'],
                                    password=DB_CONFIG['password'], host=DB_CONFIG['host'],
                                    port=DB_CONFIG['port'])
            return conn
        except Exception:
            return None

    def init_database(self):
        try:
            conn_no_db = self.get_db_connection(dbname='postgres')
            if not conn_no_db:
                return
            conn_no_db.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor_no_db = conn_no_db.cursor()
            try:
                cursor_no_db.execute(sql.SQL("CREATE DATABASE {}"),sql.Identifier(DB_CONFIG['dbname']))
            except psycopg2.errors.DuplicateDatabase:
                pass
            finally:
                cursor_no_db.close(); conn_no_db.close()
            conn = self.get_db_connection()
            if conn:
                with conn.cursor() as c:
                    c.execute('''
                        CREATE TABLE IF NOT EXISTS conversation (
                            id SERIAL PRIMARY KEY, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            user_id TEXT, role TEXT, message TEXT )
                    ''')
                    c.execute('''
                        CREATE TABLE IF NOT EXISTS users (
                            id TEXT PRIMARY KEY, hashed_secret TEXT NOT NULL, classification TEXT,
                            interaction_data TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
                    ''')
                    c.execute('''
                        CREATE TABLE IF NOT EXISTS feedback (
                            id SERIAL PRIMARY KEY, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            user_id TEXT, feedback_text TEXT, sentiment TEXT, resolved BOOLEAN DEFAULT FALSE )
                    ''')
                    c.execute('''
                        CREATE TABLE IF NOT EXISTS knowledge (
                            id SERIAL PRIMARY KEY, question TEXT UNIQUE, answer TEXT,
                            source TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP )
                    ''')
                conn.commit()
                conn.close()
        except Exception:
            pass

    def encrypt_data(self, data: str) -> str:
        return CIPHER_SUITE.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        if not encrypted_data: return ""
        try:
            return CIPHER_SUITE.decrypt(encrypted_data.encode()).decode()
        except Exception:
            return "[DECRYPTION FAILED]"

    def get_or_create_user(self):
        conn = self.get_db_connection()
        if not conn: return "unknown_user"
        try:
            user_id = hashlib.sha256(secrets.token_bytes(16)).hexdigest()
            hashed_secret = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
            encrypted_data = self.encrypt_data(json.dumps({}))
            with conn.cursor() as c:
                c.execute(
                    "INSERT INTO users (id, hashed_secret, classification, interaction_data) VALUES (%s, %s, %s, %s)",
                    (user_id, hashed_secret, "anonymous", encrypted_data)
                )
                conn.commit()
            return user_id
        except Exception:
            return "unknown_user"
        finally:
            if conn: conn.close()

    def log_conversation(self, role: str, message: str):
        conn = self.get_db_connection()
        if not conn: return
        try:
            with conn.cursor() as c:
                c.execute("INSERT INTO conversation (user_id, role, message) VALUES (%s, %s, %s)",
                          (self.user_id, role, self.encrypt_data(message)))
                conn.commit()
        except Exception:
            pass
        finally:
            if conn: conn.close()

    def add_message_to_chat(self, sender: str, message: str):
        color = "50C878" if sender == "Chatbot" else "87CEEB"
        self.chat_log_label.text += f"[b][color=#{color}]{sender}:[/color][/b] {message}\n\n"

    def send_message(self, instance):
        user_input = self.input_field.text.strip()
        if user_input:
            sanitized_input = sanitize_input(user_input)
            self.add_message_to_chat("You", sanitized_input)
            self.log_conversation("user", sanitized_input)
            self.input_field.text = ""
            self.executor.submit(self.process_message, sanitized_input)

    def process_message(self, user_input: str):
        try:
            # 1. Intrusion Detection
            if intrusion_system.check_event(user_input):
                response = "⚠️ Security alert: Suspicious request detected and blocked."
                if not self.intrusion_alerted:
                    Clock.schedule_once(lambda dt: self.add_message_to_chat("Chatbot", response), 0)
                    self.intrusion_alerted = True
                self.log_conversation("bot", response)
                return
            # 2. Command Handling
            if user_input.startswith('/'): 
                self.handle_command(user_input)
                return
            # 3. Knowledge Base
            response = self.find_best_match_in_knowledge(user_input)
            # 4. LLM Model
            if response is None:
                response = self.generate_response_with_llm(user_input)
        except Exception:
            response = "I'm sorry, I encountered an error while processing your request."
        Clock.schedule_once(lambda dt: self.add_message_to_chat("Chatbot", response), 0)
        self.log_conversation("bot", response)
        self.executor.submit(self.text_to_speech, response)
        # 5. Model retraining (scheduled)
        self.schedule_model_retraining()

    def handle_command(self, command: str):
        parts = command.lower().split()
        cmd = parts[0]
        args = parts[1:]
        response = ""
        if cmd == '/help':
            response = (
                "Available Commands:\n"
                "- /help : Show help.\n"
                "- /intrusion : Show intrusion detection log.\n"
                "- /time : Display date and time.\n"
                "- /clear : Clear chat window.\n"
                "- /scrape [URL] : Scrape text from a URL."
            )
        elif cmd == '/intrusion':
            events = intrusion_system.get_events()
            if not events:
                response = "No suspicious activity detected."
            else:
                response = "\n".join([f"{dt}: {txt}" for dt, txt in events])
        elif cmd == '/time':
            response = f"The current date and time is: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            "
        # The complete implementation of the command handling continues here...