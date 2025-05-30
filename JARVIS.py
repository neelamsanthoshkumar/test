from flask import Flask, request, jsonify, render_template_string, send_file
import requests
import time
import os
from datetime import datetime
from googlesearch import search
import uuid
import base64
from io import BytesIO
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import re
import gtts
import concurrent.futures
import webbrowser
import subprocess
import json
import importlib.util

app = Flask(__name__)

# Configure pytesseract (for OCR)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

# Load configuration
try:
    import config
except ImportError:
    print("Warning: config.py not found. Please copy config.template.py to config.py and fill in your API keys.")
    print("Using template configuration for now...")
    import config.template as config

class ModernJARVIS:
    def __init__(self):
        # API configurations from config file
        self.groq_api_url = config.GROQ_API_URL
        self.groq_api_keys = config.GROQ_API_KEYS
        self.current_groq_key_index = 0
        self.groq_model = config.GROQ_MODEL
        
        self.openrouter_api_url = config.OPENROUTER_API_URL
        self.openrouter_api_key = config.OPENROUTER_API_KEY
        self.openrouter_model = config.OPENROUTER_MODEL
        
        # Primary Image generation API configurations (A4F)
        self.image_api_key = config.A4F_API_KEY
        self.image_models_url = config.IMAGE_MODELS_URL
        self.image_generate_url = config.IMAGE_GENERATE_URL
        self.size_options = ["512x512", "1024x1024", "1792x1024", "1024x1792"]
        
        # Primary Image generation models with fallback order (A4F)
        self.image_models = config.IMAGE_MODELS

        # Hugging Face Image Generation API Configuration (Alternative)
        self.huggingface_api_keys = config.HUGGINGFACE_API_KEYS
        self.current_huggingface_key_index = 0
        self.huggingface_models = config.HUGGINGFACE_MODELS
        self.huggingface_api_url = "https://api-inference.huggingface.co/models/"
        
        self.theme = {
            'primary': '#6366f1',
            'secondary': '#8b5cf6',
            'success': '#10b981',
            'warning': '#f59e0b',
            'error': '#ef4444',
            'text': '#f8fafc',
            'subtext': '#94a3b8',
            'bg': '#0f172a',
            'card': '#1e293b',
            'border': '#334155'
        }
        self.last_search_results = []
        self.groq_fail_count = 0
        self.max_groq_failures = 3
        self.force_openrouter = False
        self.user_preferences = {
            'model': 'auto',  # auto, groq, openrouter
            'theme': 'dark',
            'font_size': 'medium'
        }
        
        self.translation_dict = {
            'en': {
                'es': {'hello': 'hola', 'goodbye': 'adiós', 'thank you': 'gracias', 'please': 'por favor', 'yes': 'sí', 'no': 'no', 'water': 'agua', 'food': 'comida', 'friend': 'amigo', 'love': 'amor'},
                'fr': {'hello': 'bonjour', 'goodbye': 'au revoir', 'thank you': 'merci', 'please': 's\'il vous plaît', 'yes': 'oui', 'no': 'non', 'water': 'eau', 'food': 'nourriture', 'friend': 'ami', 'love': 'amour'},
                'de': {'hello': 'hallo', 'goodbye': 'auf wiedersehen', 'thank you': 'danke', 'please': 'bitte', 'yes': 'ja', 'no': 'nein', 'water': 'wasser', 'food': 'nahrung', 'friend': 'freund', 'love': 'liebe'},
                'hi': {'hello': 'नमस्ते', 'goodbye': 'अलविदा', 'thank you': 'धन्यवाद', 'please': 'कृपया', 'yes': 'हाँ', 'no': 'नहीं', 'water': 'पानी', 'food': 'भोजन', 'friend': 'दोस्त', 'love': 'प्यार'}
            }
        }
        
        self.app_mappings = {
            'calculator': 'calc.exe', 'notepad': 'notepad.exe', 'paint': 'mspaint.exe',
            'word': 'winword.exe', 'excel': 'excel.exe', 'powerpoint': 'powerpnt.exe',
            'chrome': 'chrome.exe', 'firefox': 'firefox.exe', 'edge': 'msedge.exe',
            'spotify': 'spotify.exe', 'discord': 'discord.exe',
            'youtube': 'https://youtube.com', 'google': 'https://google.com', 'github': 'https://github.com',
            'twitter': 'https://twitter.com', 'facebook': 'https://facebook.com', 'instagram': 'https://instagram.com',
            'linkedin': 'https://linkedin.com', 'reddit': 'https://reddit.com', 'wikipedia': 'https://wikipedia.org',
            'amazon': 'https://amazon.com', 'netflix': 'https://netflix.com',
            'gui': 'http://localhost:5000/gui' # Assuming your Flask app runs on port 5000 locally
        }
        
        self.uploads_dir = os.path.join(os.path.expanduser("~"), "uploads_jarvis")
        self.tts_dir = os.path.join(os.path.expanduser("~"), "tts_jarvis")
        self.images_dir = os.path.join(os.path.expanduser("~"), "generated_images_jarvis")
        for dir_path in [self.uploads_dir, self.tts_dir, self.images_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def _get_current_info(self):
        now = datetime.now()
        return {'time': now.strftime("%H:%M:%S"), 'date': now.strftime("%Y-%m-%d"), 'day': now.strftime("%A"), 'timezone': time.tzname[0], 'timestamp': int(time.time())}

    def _web_search(self, query, num_results=3):
        try:
            results = list(search(query, num_results=num_results, stop=num_results, pause=2))
            self.last_search_results = results
            return results
        except Exception as e:
            return [f"Search error: {str(e)}"]

    def _get_weather(self, location):
        return f"Weather information for {location}: Sunny, 72°F (22°C)" # Placeholder

    def _get_stock_price(self, symbol):
        return f"Current price of {symbol}: $150.75 (↗ +2.5%)" # Placeholder

    def _calculate(self, expression):
        try:
            # Sanitize expression to allow only basic math operations
            allowed_chars = "0123456789.+-*/()% "
            if not all(char in allowed_chars for char in expression):
                return "⚠️ Invalid characters in expression for calculation."
            result = eval(expression) # Be very careful with eval in production
            return f"Calculation result: {expression} = {result}"
        except Exception as e:
            return f"⚠️ Could not evaluate the mathematical expression: {str(e)}"
            
    def _translate_text(self, text, target_lang):
        if not text or not target_lang: return "Please provide both text and target language"
        target_lang = target_lang.lower()
        source_lang = 'en'
        if source_lang not in self.translation_dict or target_lang not in self.translation_dict[source_lang]:
            supported_langs = ', '.join(self.translation_dict['en'].keys()) if 'en' in self.translation_dict else 'none'
            return f"⚠️ Unsupported language. Supported target languages from English: {supported_langs}"
        words = text.lower().split()
        translated_words = [self.translation_dict[source_lang][target_lang].get(word, word) for word in words]
        return ' '.join(translated_words)
        
    def _generate_tts_file(self, text, lang='en'):
        try:
            filename = f"tts_{uuid.uuid4().hex}.mp3"
            filepath = os.path.join(self.tts_dir, filename)
            tts = gtts.gTTS(text=text, lang=lang, lang_check=False if lang == 'hi' else True) # lang_check=False for custom/less common like 'hi'
            tts.save(filepath)
            return filepath
        except Exception as e:
            print(f"TTS generation error: {str(e)}")
            return None

    def _generate_image_with_model(self, model, prompt, size="1024x1024"):
        headers = {"Authorization": f"Bearer {self.image_api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "prompt": prompt, "n": 1, "size": size}
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(requests.post, self.image_generate_url, json=payload, headers=headers)
                try:
                    response = future.result(timeout=30) # Increased timeout for API call
                    response.raise_for_status()
                    info = response.json()["data"][0]
                    
                    filename = f"image_{uuid.uuid4().hex}.png"
                    filepath = os.path.join(self.images_dir, filename)

                    if "b64_json" in info:
                        with open(filepath, "wb") as f: f.write(base64.b64decode(info["b64_json"]))
                        return {"success": True, "url": f"/download-image?file={filename}", "model": model}
                    elif "url" in info:
                        image_url_from_api = info["url"]
                        print(f"A4F returned URL: {image_url_from_api}. Attempting to download and save locally.")
                        try:
                            image_response = requests.get(image_url_from_api, timeout=30) 
                            image_response.raise_for_status()
                            
                            content_type = image_response.headers.get('Content-Type', '').lower()
                            if 'image' in content_type:
                                with open(filepath, "wb") as f: f.write(image_response.content)
                                print(f"Successfully downloaded and saved image to {filepath}")
                                return {"success": True, "url": f"/download-image?file={filename}", "model": f"{model} (cached)"}
                            else:
                                print(f"A4F URL ({image_url_from_api}) did not provide image content-type. Got: {content_type}")
                                return {"success": False, "error": f"URL content type not image: {content_type}", "model": model}
                        
                        except requests.exceptions.Timeout as e_timeout:
                            print(f"Timeout downloading image from A4F URL {image_url_from_api}: {str(e_timeout)}")
                            return {"success": False, "error": f"Timeout downloading image from {image_url_from_api}", "model": model}
                        except requests.exceptions.RequestException as e_download:
                            print(f"Failed to download image from A4F URL {image_url_from_api}: {str(e_download)}")
                            return {"success": False, "error": f"Download failed from {image_url_from_api}", "model": model}
                        except IOError as e_save:
                            print(f"Failed to save downloaded image from A4F URL to {filepath}: {str(e_save)}")
                            return {"success": False, "error": f"Save failed for downloaded image", "model": model}
                        except Exception as e_general_download:
                            print(f"Unexpected error during image download/save from {image_url_from_api}: {str(e_general_download)}")
                            return {"success": False, "error": f"Unexpected error processing A4F URL", "model": model}

                    print("No b64_json or url in A4F response's data[0].") 
                    return {"success": False, "error": "No image data (b64_json or url) from A4F", "model": model}

                except concurrent.futures.TimeoutError:
                    future.cancel() 
                    print(f"A4F API request timed out for model {model}.")
                    return {"success": False, "error": "A4F request timed out", "model": model}
                except requests.exceptions.HTTPError as e_http:
                    error_detail = f"A4F HTTP Error {e_http.response.status_code}"
                    try: error_detail += f" - {e_http.response.json()}"
                    except ValueError: error_detail += f" - {e_http.response.text[:200]}"
                    print(f"Error for model {model}: {error_detail}")
                    return {"success": False, "error": error_detail, "model": model}
                except Exception as e_api:
                    print(f"Error processing A4F response for {model}: {str(e_api)}")
                    return {"success": False, "error": f"A4F API response error: {str(e_api)}", "model": model}
        
        except Exception as e_executor:
            print(f"ThreadPoolExecutor error for image generation with {model}: {str(e_executor)}")
            return {"success": False, "error": f"Image generation threading error: {str(e_executor)}", "model": model}

    def _generate_image_huggingface(self, prompt):
        if not self.huggingface_api_keys:
            print("Hugging Face API keys list is not configured or is empty.")
            return {"success": False, "error": "Hugging Face API key(s) not configured.", "model": "Hugging Face (Misconfigured)"}

        for model_id in self.huggingface_models:
            api_url = f"{self.huggingface_api_url}{model_id}"
            model_name_display_parts = model_id.split('/')
            model_short_name = model_name_display_parts[1] if len(model_name_display_parts) > 1 else model_id
            model_name_display = f"HF ({model_short_name})"

            print(f"Attempting image generation with Hugging Face model: {model_id}")
            
            key_attempt_count_for_this_model = 0
            original_key_index_for_this_model_cycle = self.current_huggingface_key_index 

            while key_attempt_count_for_this_model < len(self.huggingface_api_keys):
                current_hf_key = self.huggingface_api_keys[self.current_huggingface_key_index]
                headers = {"Authorization": f"Bearer {current_hf_key}"}
                payload = {"inputs": prompt}
                
                print(f"  Trying model {model_id} with HF key index {self.current_huggingface_key_index}")

                try:
                    response = requests.post(api_url, headers=headers, json=payload, timeout=90) 

                    if response.status_code == 200:
                        if 'image' in response.headers.get('Content-Type', '').lower():
                            filename = f"hf_image_{uuid.uuid4().hex}.png"
                            filepath = os.path.join(self.images_dir, filename)
                            with open(filepath, "wb") as f: f.write(response.content)
                            return {"success": True, "url": f"/download-image?file={filename}", "model": model_name_display}
                        else: 
                            error_detail = "Unknown error (200 OK but not an image)"
                            try: error_detail = response.json().get("error", error_detail)
                            except ValueError: pass 
                            print(f"Hugging Face model {model_id} (key index {self.current_huggingface_key_index}) returned 200 OK but no image data: {error_detail}")
                            break 
                    
                    elif response.status_code == 503: 
                        error_json = response.json()
                        estimated_time = error_json.get("estimated_time", "unknown")
                        print(f"Hugging Face model {model_id} is loading (estimated time: {estimated_time}s). Skipping this model for now.")
                        break 
                    
                    else: 
                        error_detail = f"HTTP {response.status_code}"
                        try: error_detail += f" - {response.json().get('error', response.text[:100])}" 
                        except ValueError: error_detail += f" - {response.text[:100]}"
                        print(f"  HF Model {model_id} (key index {self.current_huggingface_key_index}) failed with {error_detail}. Rotating key.")
                        
                        self.current_huggingface_key_index = (self.current_huggingface_key_index + 1) % len(self.huggingface_api_keys)
                        key_attempt_count_for_this_model += 1
                        
                        if self.current_huggingface_key_index == original_key_index_for_this_model_cycle and key_attempt_count_for_this_model > 0 :
                            if len(self.huggingface_api_keys) == 1 and key_attempt_count_for_this_model >= 1:
                                print(f"  Single HF key failed for model {model_id} after HTTP error. Moving to next model.")
                            elif key_attempt_count_for_this_model >= len(self.huggingface_api_keys):
                                print(f"  All keys tried and failed for model {model_id} after HTTP error. Moving to next model.")
                            break 
                        continue 
                
                except requests.exceptions.RequestException as e: 
                    print(f"  HF RequestException for model {model_id} (key {self.current_huggingface_key_index}): {str(e)}. Rotating key.")
                    self.current_huggingface_key_index = (self.current_huggingface_key_index + 1) % len(self.huggingface_api_keys)
                    key_attempt_count_for_this_model += 1
                    if self.current_huggingface_key_index == original_key_index_for_this_model_cycle and key_attempt_count_for_this_model > 0:
                         if len(self.huggingface_api_keys) == 1 and key_attempt_count_for_this_model >=1 :
                             print(f"  Single HF key failed after RequestException for model {model_id}. Moving to next model.")
                             break
                         elif key_attempt_count_for_this_model >= len(self.huggingface_api_keys):
                             print(f"  Cycled through all keys after RequestException for model {model_id}. Moving to next model.")
                             break 
                    continue

                except Exception as e: 
                    print(f"  HF General Exception for model {model_id} (key {self.current_huggingface_key_index}): {str(e)}. Skipping this model.")
                    break 
        
        return {"success": False, "error": "All tried Hugging Face models (with key rotation) failed or timed out.", "model": "Hugging Face (All Failed/Keys Exhausted)"}

    def generate_image(self, prompt):
        if not prompt:
            return {"error": "Please provide a prompt for image generation", "action": None, "model": "Image Generation"}
        
        for model in self.image_models: 
            print(f"Attempting image generation with primary model: {model}")
            result = self._generate_image_with_model(model, prompt)
            if result.get("success"):
                return {"response": f"Here's the image (using {result['model']}): \"{prompt}\"", "action": "image", "image_url": result["url"], "model": f"ImageGen ({result['model']})"}
            else:
                print(f"Primary image generation failed with {model}: {result.get('error', 'Unknown error')}")
        
        print("All primary image models failed. Attempting Hugging Face image generator...")
        hf_result = self._generate_image_huggingface(prompt)
        if hf_result.get("success"):
            return {"response": f"Here's the image (using {hf_result['model']}): \"{prompt}\"", "action": "image", "image_url": hf_result["url"], "model": f"ImageGen ({hf_result['model']})"}
        else:
            print(f"Hugging Face image generation also failed: {hf_result.get('error', 'Unknown error')}")
        
        return {"error": "⚠️ All image generation services failed. Please try again later.", "action": None, "model": "Image Generation"}

    def _open_app_or_website(self, target):
        target_lower = target.lower().strip()
        path = self.app_mappings.get(target_lower)
        
        if path:
            try:
                if path.startswith('http'): webbrowser.open(path); return f"Opened website: {path}"
                else: subprocess.Popen(path); return f"Opened application: {target}"
            except Exception as e: return f"⚠️ Failed to open {target}: {str(e)}"
        
        if target_lower.startswith('http://') or target_lower.startswith('https://'):
            try: webbrowser.open(target_lower); return f"Opened website: {target_lower}"
            except Exception as e: return f"⚠️ Failed to open website: {str(e)}"
        
        try: 
            subprocess.Popen(target) 
            return f"Attempted to open: {target}"
        except Exception:
            search_url = f"https://www.google.com/search?q={requests.utils.quote(target)}"
            webbrowser.open(search_url)
            return f"⚠️ Failed to open '{target}' directly. Searched for it on Google."

    def get_response(self, prompt, extracted_text=""):
        current_info = self._get_current_info()
        
        if prompt.lower().startswith("/"):
            command_parts = prompt[1:].lower().split()
            command = command_parts[0]
            args = " ".join(command_parts[1:])
            
            if command == "clear": return {'response': "✅ Chat history cleared.", 'action': 'clear_chat', 'model': 'System'}
            elif command == "help":
                help_text = """
**Available Commands:**
- `/clear` - Clear chat
- `/help` - This help
- `/weather [loc]` - Weather
- `/stock [sym]` - Stock price
- `/calc [expr]` - Calculate
- `/summarize [text_opt]` - Summarize chat/text/doc
- `/translate [text] to [lang]` - Translate (e.g., /translate hello to es)
- `/tts [text]` - Text-to-speech. Add 'in hindi' for Hindi.
- `/image [prompt]` - Generate image
- `/open [app/site]` - Open app/website (e.g., /open notepad, /open google.com)
- `/list-apps` - List openable apps/sites
- `/gui` - Open voice GUI
                """
                return {'response': help_text, 'action': None, 'model': 'System'}
            elif command == "weather": return {'response': self._get_weather(args.strip() or "your location"), 'action': None, 'model': 'System'}
            elif command == "stock": return {'response': self._get_stock_price(args.strip().upper() or "AAPL"), 'action': None, 'model': 'System'}
            elif command == "calc": return {'response': self._calculate(args.strip() or "2+2"), 'action': None, 'model': 'System'}
            elif command == "summarize":
                text_to_summarize = args.strip() or extracted_text
                if not text_to_summarize: return {'response': "Provide text, upload doc, or `/summarize chat` (pending).", 'action': None, 'model': 'System'}
                return self._get_ai_summary(f"Please summarize concisely:\n\n{text_to_summarize}")
            elif command == "translate":
                match = re.match(r"^(.*)\s+to\s+([a-z]{2,3})$", args.strip(), re.IGNORECASE)
                if match:
                    translation = self._translate_text(match.group(1).strip(), match.group(2).strip().lower())
                    return {'response': f"Translation to {match.group(2).lower()}:\n'{match.group(1).strip()}' → '{translation}'", 'action': None, 'model': 'Translator'}
                return {'response': "Format: /translate [text] to [lang_code] (e.g., /translate hello to es)", 'action': None, 'model': 'System'}
            elif command == "tts":
                text_to_speak = args.strip()
                if not text_to_speak: return {'response': "Specify text for TTS (e.g., /tts hello)", 'action': None, 'model': 'System'}
                lang_code = 'hi' if text_to_speak.lower().endswith(" in hindi") else 'en'
                if lang_code == 'hi': text_to_speak = text_to_speak[:-9].strip()
                tts_file_path = self._generate_tts_file(text_to_speak, lang_code)
                if tts_file_path: return {'response': f"🔊 TTS for: '{text_to_speak}' (Lang: {lang_code})", 'action': 'tts', 'tts_file': tts_file_path, 'model': 'TTS'}
                return {'response': "⚠️ Failed to generate TTS", 'action': None, 'model': 'System'}
            elif command == "image":
                img_prompt = args.strip()
                if not img_prompt: return {'response': "Specify image description (e.g., /image a cat)", 'action': None, 'model': 'System'}
                return self.generate_image(img_prompt)
            elif command == "open":
                target_to_open = args.strip()
                if not target_to_open: return {'response': "Specify app/website (e.g., /open calc or /open youtube.com)", 'action': None, 'model': 'System'}
                return {'response': self._open_app_or_website(target_to_open), 'action': None, 'model': 'System'}
            elif command == "list-apps":
                apps_list = "**Open with /open:**\n" + "\n".join([f"- {app}" for app in sorted(self.app_mappings.keys())]) + "\n\nOr try any URL/system command."
                return {'response': apps_list, 'action': None, 'model': 'System'}
            elif command == "gui": return {'response': self._open_app_or_website("gui"), 'action': None, 'model': 'System'}

        if "file has been removed" in prompt.lower() or "context cleared" in prompt.lower():
            return {'response': "✅ File context removed.", 'action': None, 'model': 'System'}
        
        prompt_lower_normalized = prompt.lower().strip().replace("what's", "what is")
        direct_time_date_queries = [
            "time", "what time is it", "current time", "tell me the time",
            "date", "what is the date", "current date", "tell me the date",
            "day", "what day is it", "what day of the week is it", "current day", "tell me the day",
            "year", "what year is it", "current year", "tell me the year", 
            "today", "what is today"
        ]
        normalized_direct_queries = [q.replace("what's", "what is") for q in direct_time_date_queries]
        if prompt_lower_normalized in normalized_direct_queries:
            return {'response': f"**Time:** {current_info['time']}, **Date:** {current_info['day']}, {current_info['date']}", 'action': None, 'model': 'System'}

        if 'web search' in prompt.lower() or 'search for' in prompt.lower():
            query = prompt.replace('web search', '').replace('search for', '').strip()
            results = self._web_search(query)
            if results and not results[0].startswith("Search error"):
                res_str = f"**Search results for '{query}':**\n" + "\n".join([f"{i+1}. {r}" for i, r in enumerate(results)])
                return {'response': res_str, 'action': 'search_results', 'results': results, 'model': 'Web Search'}
            return {'response': "⚠️ Search failed or no results.", 'action': None, 'model': 'System'}

        if 'open first result' in prompt.lower() and self.last_search_results:
            webbrowser.open(self.last_search_results[0])
            return {'response': f"Opening: {self.last_search_results[0]}", 'action': 'redirect', 'url': self.last_search_results[0], 'model': 'System'}

        system_message = f"You are JARVIS 5.0 AI. Precise Markdown responses. Time: {current_info['time']}, Date: {current_info['date']}"
        if extracted_text: system_message += f"\n\n**Document Context:**\n{extracted_text}"

        preferred_model = self.user_preferences.get('model', 'auto')
        if preferred_model == 'groq': self.force_openrouter = False; self.groq_fail_count = 0
        elif preferred_model == 'openrouter': self.force_openrouter = True

        if not self.force_openrouter and self.groq_fail_count < self.max_groq_failures:
            groq_response = self._try_groq_api(system_message, prompt)
            if groq_response and not groq_response.get('error'): self.groq_fail_count = 0; return groq_response
            self.groq_fail_count += 1
            print(f"Groq API failed ({self.groq_fail_count}/{self.max_groq_failures}). Error: {groq_response.get('error') if groq_response else 'None'}")
            if self.groq_fail_count >= self.max_groq_failures: self.force_openrouter = True; print("Forcing OpenRouter.")

        openrouter_response = self._try_openrouter_api(system_message, prompt)
        if openrouter_response and not openrouter_response.get('error'): return openrouter_response
        
        return {'error': "⚠️ All AI services unavailable. Try later.", 'action': None, 'model': 'Error'}
        
    def _get_ai_summary(self, prompt_with_text): 
        current_info = self._get_current_info()
        system_message = f"JARVIS 5.0 AI Summarizer. Time: {current_info['time']}, Date: {current_info['date']}"
        openrouter_response = self._try_openrouter_api(system_message, prompt_with_text)
        if openrouter_response and not openrouter_response.get('error'): return openrouter_response
        print(f"OpenRouter summarization failed. Error: {openrouter_response.get('error') if openrouter_response else 'None'}")
        
        groq_response = self._try_groq_api(system_message, prompt_with_text)
        if groq_response and not groq_response.get('error'): return groq_response
        print(f"Groq summarization also failed. Error: {groq_response.get('error') if groq_response else 'None'}")
            
        return {'error': "⚠️ Failed to generate summary.", 'action': None, 'model': 'Error'}

    def _try_groq_api(self, system_message, prompt):
        payload = {"messages": [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
                   "model": self.groq_model, "temperature": 0.7, "max_tokens": 2048, "top_p": 0.9}
        try:
            for i in range(len(self.groq_api_keys)): 
                current_key_idx_to_try = (self.current_groq_key_index + i) % len(self.groq_api_keys)
                current_key = self.groq_api_keys[current_key_idx_to_try]
                headers = {"Authorization": f"Bearer {current_key}", "Content-Type": "application/json"}
                
                response = requests.post(self.groq_api_url, headers=headers, json=payload, timeout=25)
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    if not content.strip(): return {'error': "Groq API: empty response.", 'model': f"Groq ({self.groq_model})"}
                    self.current_groq_key_index = current_key_idx_to_try 
                    return {'response': content, 'action': None, 'model': f"Groq ({self.groq_model})"}
                
                print(f"Groq API returned {response.status_code} with key index {current_key_idx_to_try}.")
                if response.status_code not in [401, 429, 500]: 
                    response.raise_for_status() 

            return {'error': f"Groq API failed with all keys. Last status: {response.status_code if 'response' in locals() else 'N/A'}", 'model': f"Groq ({self.groq_model})"}

        except requests.exceptions.HTTPError as http_err:
            err_details = f"HTTPError: {http_err.response.status_code} - {http_err.response.text[:100]}"
            print(f"Groq API Error: {err_details}")
            return {'error': f"Groq API failed: {err_details}", 'model': f"Groq ({self.groq_model})"}
        except requests.exceptions.RequestException as req_err:
            print(f"Groq API Request Exception: {req_err}")
            return {'error': f"Groq API request failed: {str(req_err)}", 'model': f"Groq ({self.groq_model})"}
        except Exception as e:
            print(f"Groq API General Error: {str(e)}")
            return {'error': f"Unexpected error with Groq API: {str(e)}", 'model': f"Groq ({self.groq_model})"}

    def _try_openrouter_api(self, system_message, prompt):
        payload = {"model": self.openrouter_model, "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
                   "temperature": 0.7, "max_tokens": 2048}
        headers = {"Authorization": f"Bearer {self.openrouter_api_key}", "HTTP-Referer": "https://modern-jarvis-app.com", "X-Title": "ModernJARVIS", "Content-Type": "application/json"}
        try:
            response = requests.post(self.openrouter_api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            msg_content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if not msg_content.strip(): return {'error': "OpenRouter API: empty response.", 'model': f"OpenRouter ({self.openrouter_model})"}
            return {'response': msg_content, 'action': None, 'model': f"OpenRouter ({self.openrouter_model})"}
        except requests.exceptions.HTTPError as http_err:
            err_details = f"HTTPError: {http_err.response.status_code} - {http_err.response.text[:100]}"
            print(f"OpenRouter API Error: {err_details}")
            return {'error': f"OpenRouter API failed: {err_details}", 'model': f"OpenRouter ({self.openrouter_model})"}
        except requests.exceptions.RequestException as req_err:
            print(f"OpenRouter API Request Exception: {req_err}")
            return {'error': f"OpenRouter API request failed: {str(req_err)}", 'model': f"OpenRouter ({self.openrouter_model})"}
        except Exception as e:
            print(f"OpenRouter API General Error: {str(e)}")
            return {'error': f"Unexpected error with OpenRouter API: {str(e)}", 'model': f"OpenRouter ({self.openrouter_model})"}

    def process_uploaded_file(self, file):
        temp_filepath = None 
        try:
            temp_filename_base = str(uuid.uuid4())
            original_ext = os.path.splitext(file.filename)[1]
            temp_filename = temp_filename_base + original_ext
            temp_filepath = os.path.join(self.uploads_dir, temp_filename)
            file.save(temp_filepath)
            
            extracted_text = ""
            file_lower = file.filename.lower()

            if file_lower.endswith('.pdf'):
                with open(temp_filepath, 'rb') as pdf_file:
                    pdf_reader = PdfReader(pdf_file)
                    for i, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text: extracted_text += f"[Page {i+1}]\n{re.sub(r'\s+', ' ', page_text).strip()}\n\n"
            elif file_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                try: extracted_text = pytesseract.image_to_string(Image.open(temp_filepath), timeout=30)
                except (pytesseract.TesseractError, RuntimeError) as ocr_err: extracted_text = f"[OCR Error: {str(ocr_err)}]"
            elif file_lower.endswith('.txt'):
                for enc in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(temp_filepath, 'r', encoding=enc) as f: extracted_text = f.read(); break
                    except UnicodeDecodeError:
                        if enc == 'cp1252': extracted_text = "[Error: Could not decode text file]"
            elif file_lower.endswith(('.doc', '.docx')):
                extracted_text = "[Note: .doc/.docx parsing is basic. Using 'strings' like approach.]\n"
                try:
                    with open(temp_filepath, 'r', errors='ignore') as f: 
                        content_sample = f.read(5000) 
                        extracted_text += "".join(filter(lambda x: x in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c", content_sample))
                except Exception as doc_err: extracted_text += f"[Error reading Word doc: {str(doc_err)}]"
            else: extracted_text = f"[File Type Not Supported for Text Extraction: {file.filename}]"
            
            return extracted_text.strip()
        except Exception as e:
            print(f"File processing error: {str(e)}")
            return f"[Error during file processing: {str(e)}]"
        finally:
            if temp_filepath and os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except OSError as e_os: print(f"Error removing temp file {temp_filepath}: {e_os}")

jarvis = ModernJARVIS()

# Create a directory for feedback logs if it doesn't exist
FEEDBACK_DIR = os.path.join(os.path.expanduser("~"), "jarvis_feedback")
if not os.path.exists(FEEDBACK_DIR):
    os.makedirs(FEEDBACK_DIR)

# --- End of First Part of Python Code ---
HTML =  """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JARVIS 5.0 - Advanced AI Assistant</title>

  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.12.313/pdf.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://unpkg.com/tesseract.js@2.1.5/dist/tesseract.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js"></script>
  <link href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet" />

  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV" crossorigin="anonymous">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked-math@1.1.0/dist/marked-math.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous"></script>
  
  <script defer src="https://www.gstatic.com/firebasejs/9.22.1/firebase-app-compat.js"></script>
  <script defer src="https://www.gstatic.com/firebasejs/9.22.1/firebase-auth-compat.js"></script>
  <script defer src="https://www.gstatic.com/firebasejs/9.22.1/firebase-database-compat.js"></script>

  <script>
    pdfjsLib.GlobalWorkerOptions.workerSrc =
      'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.12.313/pdf.worker.min.js';
  </script>
  <style>
    html {
      overflow-x: hidden; /* Prevent horizontal scrollbars if anything unexpectedly overflows */
    }
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --secondary: #8b5cf6;
      --success: #10b981;
      --warning: #f59e0b;
      --error: #ef4444;
      --text: #f8fafc;
      --subtext: #94a3b8;
      --highlight: rgba(99, 102, 241, 0.1);
      --sidebar-width: 280px;
      --primary-rgb: 99, 102, 241; 
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', sans-serif;
      line-height: 1.6;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden; /* Body itself won't show scrollbars, children manage their own */
    }
    /* Styles for Login Prompt Modal */
    .login-prompt-modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.6); /* Slightly less dark backdrop */
        display: flex; 
        align-items: center;
        justify-content: center;
        z-index: 3000;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s ease;
        backdrop-filter: blur(8px); /* Increased blur */
    }
    .login-prompt-modal.visible {
        opacity: 1;
        pointer-events: auto;
    }
    .login-prompt-content {
        background: linear-gradient(145deg, color-mix(in srgb, var(--card) 95%, var(--primary) 5%), color-mix(in srgb, var(--card) 70%, var(--primary) 30%)); /* More colourful gradient */
        padding: 3.5rem; /* Increased padding */
        border-radius: 1.25rem; /* Softer radius */
        box-shadow: 0 20px 45px rgba(0,0,0,0.3), 0 0 0 2px color-mix(in srgb, var(--primary) 60%, transparent); /* Enhanced shadow */
        width: 90%;
        max-width: 500px; /* Slightly wider */
        text-align: center;
        transform: scale(0.95);
        transition: opacity 0.3s ease, transform 0.3s ease;
    }
    .login-prompt-modal.visible .login-prompt-content {
        transform: scale(1);
    }
    .login-prompt-content h2 {
        font-size: 2.2rem; /* Slightly larger */
        color: var(--text);
        margin-bottom: 1.25rem; 
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem; 
    }
    .login-prompt-content h2 i {
        color: var(--primary); /* Kept primary, it contrasts well */
        font-size: 2.5rem; 
        filter: drop-shadow(0 0 5px rgba(var(--primary-rgb), 0.5)); /* Added subtle glow */
        animation: loginIconPulse 2s infinite ease-in-out;
    }

    @keyframes loginIconPulse {
        0%, 100% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.15); opacity: 1; }
    }

    .login-prompt-content p {
        font-size: 1.1rem; 
        color: var(--subtext);
        margin-bottom: 2.8rem; 
        line-height: 1.75; 
        max-width: 400px; 
        margin-left: auto; 
        margin-right: auto; 
    }
    .login-prompt-btn {
        background: linear-gradient(45deg, var(--primary), var(--secondary)); /* Colourful gradient button */
        color: white;
        border: none;
        padding: 1rem 2rem; /* Larger padding */
        border-radius: 0.75rem; 
        font-size: 1.1rem; 
        font-weight: 600; 
        cursor: pointer;
        transition: background-color 0.2s, transform 0.15s, box-shadow 0.2s;
        display: inline-flex;
        align-items: center;
        gap: 0.8rem; 
        box-shadow: 0 5px 15px rgba(var(--primary-rgb), 0.35); 
    }
    .login-prompt-btn:hover {
        background: linear-gradient(45deg, var(--primary-hover), color-mix(in srgb, var(--secondary) 90%, black 10%)); /* Darken on hover */
        transform: translateY(-3px); 
        box-shadow: 0 8px 18px rgba(var(--primary-rgb), 0.45); 
    }
    .login-prompt-btn:active {
        transform: translateY(0px);
    }
    .login-prompt-btn i {
        font-size: 1.2rem;
    }
    body.app-hidden .header,
    body.app-hidden .main-container {
        display: none !important; 
    }
    /* End of Login Prompt Modal Styles */

    .header {
      background: linear-gradient(135deg, var(--primary), var(--secondary));
      padding: 1rem 1.5rem;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      position: sticky;
      top: 0;
      z-index: 100;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .header h1 {
      font-size: 1.5rem;
      font-weight: 700;
      letter-spacing: -0.025em;
      display: flex;
      align-items: center;
      gap: 0.75rem;
      flex-shrink: 1;
      min-width: 0; 
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .header i {
      font-size: 1.25rem;
    }
    .header-actions {
      display: flex;
      gap: 0.75rem;
      flex-shrink: 0;
    }
    .header-btn {
      background: rgba(255, 255, 255, 0.1);
      border: none;
      color: white;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.2s;
    }
    .header-btn:hover {
      background: rgba(255, 255, 255, 0.2);
      transform: translateY(-1px);
    }
    .main-container {
      display: flex;
      flex: 1;
      overflow: hidden;
      position: relative;
    }
    .sidebar {
      width: var(--sidebar-width);
      background: var(--card);
      border-right: 1px solid var(--border);
      padding: 1rem;
      overflow-y: auto;
      transition: transform 0.3s ease;
      z-index: 10;
      position: absolute;
      top: 0;
      left: 0;
      bottom: 0;
      transform: translateX(-100%);
      display: flex;
      flex-direction: column;
    }
    .sidebar.open {
      transform: translateX(0);
    }
    .sidebar-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
    }
    .sidebar-title {
      font-size: 1rem;
      font-weight: 600;
    }
    .close-sidebar {
      background: none;
      border: none;
      color: var(--subtext);
      cursor: pointer;
      font-size: 1.25rem;
    }
    
    .chat-history-group-header {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--subtext);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.75rem 0.25rem 0.25rem;
        margin-top: 0.5rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 0.5rem;
    }
    body.light-theme .chat-history-group-header {
        border-bottom-color: var(--border);
    }

    .chat-history {
      display: flex;
      flex-direction: column;
      gap: 0.65rem;
      flex-grow: 1;
      overflow-y: auto;
      padding-bottom: 0.5rem;
      padding-right: 0.25rem;
      min-height: 0; 
    }
    .chat-item {
      padding: 0.8rem 0.75rem;
      border-radius: 0.375rem;
      cursor: pointer;
      font-size: 0.9rem;
      white-space: nowrap;
      transition: all 0.2s;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-left: 3px solid transparent;
      position: relative;
    }
    .chat-item:hover {
      background: rgba(99, 102, 241, 0.1);
    }
    .chat-item.active {
      background: var(--highlight);
      color: var(--primary);
      font-weight: 500;
      border-left: 3px solid var(--primary);
      padding-left: calc(0.75rem - 3px);
    }
    .chat-item-content {
      flex: 1;
      overflow: hidden; 
      text-overflow: ellipsis;
      margin-right: 0.5rem;
    }
    .chat-item-actions {
      position: relative;
      margin-left: 0.5rem;
      flex-shrink: 0;
    }
    .chat-item-menu-btn {
      background: none;
      border: none;
      color: var(--subtext);
      cursor: pointer;
      padding: 0.35rem;
      font-size: 0.9rem;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      transition: background-color 0.2s, color 0.2s;
    }
    .chat-item-menu-btn:hover {
      background: rgba(99, 102, 241, 0.1);
      color: var(--primary);
    }
    .chat-item-menu {
      position: absolute;
      right: 0;
      top: calc(100% + 5px);
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.375rem;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
      z-index: 20;
      padding: 0.25rem 0;
      min-width: 130px;
      display: none;
      flex-direction: column;
    }
    .chat-item-menu.visible {
      display: flex;
    }
    .chat-item-menu-option {
      background: none;
      border: none;
      color: var(--text);
      padding: 0.6rem 0.85rem;
      text-align: left;
      cursor: pointer;
      font-size: 0.85rem;
      display: flex;
      align-items: center;
      gap: 0.6rem;
      width: 100%;
      transition: background-color 0.2s, color 0.2s;
    }
    .chat-item-menu-option i {
        width: 16px;
        text-align: center;
        color: var(--subtext);
        transition: color 0.2s;
    }
    .chat-item-menu-option:hover {
      background: var(--highlight);
      color: var(--primary);
    }
    .chat-item-menu-option:hover i {
      color: var(--primary);
    }
    .chat-item-menu-option.delete-chat-btn:hover {
        color: var(--error);
        background: rgba(239, 68, 68, 0.05);
    }
    .chat-item-menu-option.delete-chat-btn:hover i {
        color: var(--error);
    }
    .chat-item-delete {
      display: none !important;
    }
    .new-chat-btn {
      width: 100%;
      background: var(--primary);
      color: white;
      border: none;
      padding: 0.75rem;
      border-radius: 0.5rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      font-weight: 500;
      transition: all 0.2s;
      margin-bottom: 1rem; 
    }
    .new-chat-btn:hover {
      background: var(--primary-hover);
    }
    .chat-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      position: relative;
    }
    .chat-messages {
      flex: 1;
      padding: 1.5rem;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      scroll-behavior: smooth;
      min-height: 0; 
    }
    .message {
      max-width: 85%;
      word-wrap: break-word;
      position: relative;
      animation: fadeIn 0.3s ease-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .message-user {
      align-self: flex-end;
      background: var(--primary);
      color: white;
      border-radius: 1.125rem 1.125rem 0 1.125rem;
      padding: 0.75rem 1.25rem;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
      overflow-wrap: break-word; 
    }
    .message-bot {
      align-self: flex-start;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 1.125rem 1.125rem 1.125rem 0;
      padding: 0.75rem 1.25rem;
      overflow-wrap: break-word; 
    }
    .message-error {
      align-self: flex-start; 
      background: rgba(239, 68, 68, 0.1);
      border-left: 3px solid var(--error);
      padding: 0.75rem 1.25rem;
      border-radius: 0.5rem;
      max-width: 100%; 
      overflow-wrap: break-word; 
    }
    .message-time {
      font-size: 0.75rem;
      color: var(--subtext);
      margin-top: 0.25rem;
      text-align: right;
    }
    .message-user .message-time {
        color: rgba(255,255,255,0.7);
    }
    .message-model {
      position: absolute;
      bottom: -0.75rem;
      left: 0;
      background: var(--bg);
      color: var(--subtext);
      font-size: 0.7rem;
      padding: 0.25rem 0.75rem;
      border-radius: 1rem;
      border: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }
    .message-model i {
      font-size: 0.6rem;
    }
    .message-bot.message-typing .typing-indicator, 
    .message-bot.message-typing .typing-text {
        margin: 0.5rem auto; 
    }


    /* --- Start: Improved Bottom UI (Input Area Redesign) --- */
    .input-container {
      padding: 0.75rem; 
      border: 1px solid var(--border); 
      background: var(--card); 
      border-radius: 30px; 
      position: sticky;
      bottom: 0;
      z-index: 5;
      margin: 0.5rem 1rem; 
      display: flex;
      flex-direction: column;
      gap: 0.75rem; 
    }

    .file-display-area { 
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.5rem 0.75rem;
      background: var(--bg); 
      border: 1px solid var(--border);
      border-radius: 0.5rem;
    }
    .file-info {
      font-size: 0.85rem;
      color: var(--subtext);
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .file-info.has-file {
      color: var(--primary);
    }
    .close-file {
      background: transparent;
      border: none;
      color: var(--subtext);
      cursor: pointer;
      padding: 0.25rem;
      opacity: 0.7;
      transition: all 0.2s;
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 50%;
      flex-shrink: 0;
    }
    .close-file:hover {
      opacity: 1;
      color: var(--error);
      background: rgba(239, 68, 68, 0.1);
    }

    .input-box { 
      width: 100%;
      background: var(--bg); 
      border: 1px solid var(--border); 
      border-radius: 20px; 
      padding: 0.9rem 1.1rem; 
      color: var(--text);
      font-family: inherit;
      resize: none;
      min-height: 50px; 
      max-height: 150px; 
      line-height: 1.5;
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
      scrollbar-width: thin;
      overflow-y: auto; 
    }
    .input-box:disabled {
        background-color: color-mix(in srgb, var(--bg) 80%, var(--border) 20%);
        cursor: not-allowed;
        opacity: 0.7;
    }
    body.light-theme .input-box:disabled {
         background-color: color-mix(in srgb, var(--bg) 80%, var(--border) 20%);
    }

    .input-box:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 20%, transparent);
    }
    
    .input-controls-row { 
      display: flex;
      align-items: center;
      gap: 0.5rem; 
      padding: 0 0.25rem; 
    }

    .input-icon-btn { 
      background: var(--bg); 
      color: var(--text);
      border: 1px solid var(--border);
      width: 38px; 
      height: 38px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.2s;
      flex-shrink: 0;
    }
    .input-icon-btn:hover:not(:disabled) {
      background: var(--primary);
      color: white;
    }
    .input-icon-btn:disabled {
        opacity: 0.6;
        cursor: not-allowed;
        background-color: color-mix(in srgb, var(--bg) 80%, var(--border) 20%);
    }
    body.light-theme .input-icon-btn:disabled {
        background-color: color-mix(in srgb, var(--bg) 80%, var(--border) 20%);
    }
    .input-icon-btn.uploading {
        animation: pulse 1.5s infinite;
    }
    
    .send-btn { 
      background: var(--primary);
      color: white;
      border: none;
      width: 38px; 
      height: 38px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.2s;
      flex-shrink: 0;
      margin-left: auto; 
    }
    .send-btn:hover:not(:disabled) {
      background: var(--primary-hover);
      transform: translateY(-1px);
    }
    .send-btn:disabled {
      opacity: 0.7;
      cursor: not-allowed;
      background: var(--subtext);
    }
    .send-btn.stop-replying { 
        background: var(--error) !important;
        opacity: 1 !important; 
        cursor: pointer !important;
    }
    .send-btn.stop-replying:hover {
        background: color-mix(in srgb, var(--error) 80%, black) !important;
    }

    .status-indicator {
      font-size: 0.85rem;
      color: var(--subtext); 
      display: flex;
      align-items: center;
      gap: 0.5rem;
      min-height: 1.2em;   
      padding: 0.1rem 0.5rem; 
      text-align: center; 
      justify-content: center;
    }
    .status-indicator i { 
        color: var(--primary);
    }
    /* --- End: Improved Bottom UI --- */

    /* --- Start: Toast Notification Styles --- */
    .toast-notification-container {
      position: fixed;
      top: 20px; 
      left: 50%;
      transform: translateX(-50%);
      z-index: 5000; 
      display: flex;
      flex-direction: column;
      gap: 10px;
      align-items: center;
      width: max-content; 
      max-width: 90vw;
    }

    .toast-notification {
      padding: 12px 20px;
      border-radius: 8px;
      color: white; 
      font-size: 0.9rem;
      font-weight: 500;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      display: flex;
      align-items: center;
      gap: 10px;
      opacity: 0;
      transform: translateY(-20px) scale(0.95);
      transition: opacity 0.3s ease, transform 0.3s ease;
      min-width: 280px; 
      text-align: left;
    }

    .toast-notification.show {
      opacity: 1;
      transform: translateY(0) scale(1);
    }

    .toast-notification.success {
      background-color: var(--success);
    }
    .toast-notification.error {
      background-color: var(--error);
    }
    .toast-notification.info { /* Added for info toasts */
      background-color: var(--primary); /* Or another distinct color */
    }
    
    .toast-notification i {
      font-size: 1.2em; 
      flex-shrink: 0;
    }
    .toast-notification span {
      flex-grow: 1;
    }
    /* --- End: Toast Notification Styles --- */


    .message-bot strong, .message-bot b {
      color: var(--primary);
      font-weight: 600;
    }
    .message-bot em, .message-bot i {
      color: var(--warning);
      font-style: italic;
    }
    .message-bot code {
      background: var(--highlight);
      color: var(--success);
      padding: 0.2rem 0.4rem;
      border-radius: 0.25rem;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
      font-size: 0.9em;
    }
    .message-bot pre {
      position: relative;
      background: #1a1a1a;
      padding: 1rem;
      border-radius: 0.5rem;
      overflow-x: auto;
      margin: 0.5rem 0;
      border: 1px solid var(--border);
    }
    .message-bot pre code {
      background: transparent;
      padding: 0;
      color: #f8f8f2;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    }
    .message-bot pre .code-actions-container {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      display: flex;
      gap: 0.25rem;
      opacity: 0;
      transition: opacity 0.2s;
      z-index: 1;
    }
    .message-bot pre:hover .code-actions-container {
      opacity: 1;
    }
    .copy-code-btn, .run-code-btn {
      background: rgba(255, 255, 255, 0.1);
      border: none;
      color: var(--text);
      padding: 0.25rem 0.5rem;
      border-radius: 0.25rem;
      cursor: pointer;
      font-size: 0.75rem;
      display: flex;
      align-items: center;
      gap: 0.25rem;
      transition: background-color 0.2s, color 0.2s;
    }
    .copy-code-btn:hover {
      background: rgba(255, 255, 255, 0.2);
    }
    .copy-code-btn.copied {
      background: var(--success);
      color: white;
    }
    .run-code-btn:hover {
      background: var(--primary);
      color: white;
    }
    .code-output-area {
      position: relative;
      border: 1px solid var(--border);
      border-radius: 0.25rem;
      margin-top: 0.5rem;
      padding: 0.5rem;
      background: var(--bg);
      min-height: 30px;
      max-height: 300px;
      overflow: auto;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
      font-size: 0.85em;
      line-height: 1.4;
      color: var(--text);
    }
    .code-output-area iframe {
        width: 100%;
        height: 250px;
        border: none;
        background-color: white;
    }
    .code-output-area pre {
      margin: 0;
      padding: 0;
      background: transparent;
      border: none;
      color: inherit;
      white-space: pre-wrap;
      word-break: break-all;
    }
    .code-output-area div {
      padding: 2px 0;
    }
    .cut-code-output-btn {
      position: absolute;
      top: 0.25rem;
      right: 0.25rem;
      background: rgba(255, 255, 255, 0.1);
      color: var(--subtext);
      border: 1px solid var(--border);
      border-radius: 50%;
      width: 24px;
      height: 24px;
      font-size: 0.8rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
      z-index: 2;
    }
    .cut-code-output-btn:hover {
      background: var(--error);
      color: white;
      border-color: var(--error);
    }
    .message-bot ul, .message-bot ol {
      padding-left: 1.5rem;
      margin: 0.5rem 0;
    }
    .message-bot blockquote {
      border-left: 3px solid var(--primary);
      padding-left: 1rem;
      margin: 0.5rem 0;
      color: var(--subtext);
    }

    .message-bot img:not(.generated-image) { 
      max-width: 100%;
      height: auto;
      border-radius: 0.375rem;
      margin: 0.5rem 0;
      display: block; 
    }
    .message-bot table { 
      width: 100%; 
      max-width: 100%; 
      border-collapse: collapse;
      margin: 1rem 0;
      display: block; 
      overflow-x: auto; 
    }
    .message-bot th, .message-bot td {
      border: 1px solid var(--border);
      padding: 0.5rem 0.75rem;
      text-align: left;
       hyphens: auto;
    }
    .message-bot th {
      background-color: color-mix(in srgb, var(--card) 90%, var(--primary) 10%); 
      font-weight: 600;
    }
    body.light-theme .message-bot th {
      background-color: color-mix(in srgb, var(--card) 95%, var(--border) 5%); 
    }
    /* Highlight important words style */
    .important-word {
      color: var(--error); /* Red color for highlighting */
      font-weight: 600;   /* Slightly bolder */
    }
    body.light-theme .important-word {
      color: var(--error); /* Ensure it remains red in light theme */
    }


    .search-results {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      margin-top: 1rem;
    }
    .search-result {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: var(--primary);
      text-decoration: none;
      padding: 0.5rem;
      border-radius: 0.5rem;
      transition: all 0.2s;
    }
    .search-result:hover {
      background: rgba(99, 102, 241, 0.1);
    }
    .search-result i {
      font-size: 1rem;
    }
    .search-result span {
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .open-all-btn {
      background: var(--success);
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 0.5rem;
      margin-top: 1rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-weight: 500;
      transition: all 0.2s;
    }
    .open-all-btn:hover {
      background: #0d9e6e;
    }
    .api-status {
      position: fixed;
      bottom: 1rem;
      right: 1rem;
      background: var(--card);
      padding: 0.5rem 1rem;
      border-radius: 0.5rem;
      border: 1px solid var(--border);
      font-size: 0.8rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      z-index: 100;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      cursor: pointer;
    }
    .api-status.groq { color: var(--success); }
    .api-status.openrouter { color: var(--warning); }
    .api-status.error { color: var(--error); }
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.05); }
      100% { transform: scale(1); }
    }
    .sidebar-overlay {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.5);
      z-index: 5;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s ease;
    }
    .sidebar-overlay.visible {
      opacity: 1;
      pointer-events: auto;
    }
    .typing-indicator {
      display: flex;
      gap: 0.25rem;
      align-items: center;
    }
    .typing-dot {
      width: 8px;
      height: 8px;
      background: var(--subtext);
      border-radius: 50%;
      animation: typingAnimation 1.4s infinite ease-in-out;
    }
    .typing-dot:nth-child(1) {
      animation-delay: 0s;
    }
    .typing-dot:nth-child(2) {
      animation-delay: 0.2s;
    }
    .typing-dot:nth-child(3) {
      animation-delay: 0.4s;
    }
    @keyframes typingAnimation {
      0%, 60%, 100% { transform: translateY(0); }
      30% { transform: translateY(-5px); }
    }
    .model-selector {
      position: absolute;
      bottom: calc(5rem + 60px); 
      right: 1rem;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 0.75rem;
      width: 250px;
      max-width: calc(100vw - 2rem);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      z-index: 10;
      transform: translateY(10px);
      opacity: 0;
      pointer-events: none;
      transition: all 0.2s ease;
    }
    .model-selector.visible {
      transform: translateY(0);
      opacity: 1;
      pointer-events: auto;
    }
    .model-selector h4 {
      margin-bottom: 0.75rem;
      font-size: 0.9rem;
      color: var(--subtext);
    }
    .model-option {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem;
      border-radius: 0.25rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .model-option:hover {
      background: rgba(99, 102, 241, 0.1);
    }
    .model-option.active {
      background: var(--highlight);
      color: var(--primary);
    }
    .model-option i {
      font-size: 0.8rem;
    }
    .model-option .checkmark {
      margin-left: auto;
      color: var(--success);
      display: none;
    }
    .model-option.active .checkmark {
      display: block;
    }
    .generated-image {
      max-width: 100%;
      border-radius: 0.5rem;
      margin: 0.5rem 0;
      border: 1px solid var(--border);
    }
    .download-btn {
      background: var(--primary);
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 0.5rem;
      margin-top: 0.5rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.8rem;
      transition: all 0.2s;
    }
    .download-btn:hover {
      background: var(--primary-hover);
    }
    
    .settings-view {
        display: flex; 
        flex-direction: column;
        overflow-y: auto;
        position: absolute; 
        top: calc(1rem + 36px + 0.5rem); 
        right: 1rem;
        width: 350px; 
        max-width: calc(100vw - 2rem);
        max-height: calc(100vh - 6rem - 70px); 
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 0.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 10; 
        padding: 1.5rem; 
        transform: translateY(10px);
        opacity: 0;
        pointer-events: none;
        transition: all 0.2s ease;
    }
    .settings-view.visible {
        transform: translateY(0);
        opacity: 1;
        pointer-events: auto;
        display: flex; 
    }

    .settings-view-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem; 
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border);
    }
    .settings-view-header h2 {
        font-size: 1.25rem; 
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin:0; 
    }
    .settings-view-header .header-btn { 
        background: var(--card); 
        color: var(--text);
        border: 1px solid var(--border);
    }
    .settings-view-header .header-btn:hover {
        background: var(--primary);
        color: white;
    }
    .settings-view-content {
        width: 100%;
        overflow-y: auto; 
        flex-grow: 1; 
    }
    .settings-option {
      margin-bottom: 1.75rem; 
    }
    .settings-option label {
      display: block;
      margin-bottom: 0.6rem; 
      font-size: 1rem; 
      font-weight: 500;
      color: var(--text);
    }
    .settings-option select {
      width: 100%;
      padding: 0.85rem 1rem; 
      background: var(--bg); 
      border: 1px solid var(--border);
      border-radius: 0.5rem; 
      color: var(--text);
      font-family: inherit;
      font-size: 0.95rem; 
      appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='%2394a3b8'%3E%3Cpath fill-rule='evenodd' d='M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z' clip-rule='evenodd'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 0.75rem center;
      background-size: 1.25em 1.25em;
    }
    body.light-theme .settings-option select {
        background: var(--bg); 
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='%2364748b'%3E%3Cpath fill-rule='evenodd' d='M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z' clip-rule='evenodd'/%3E%3C/svg%3E");
    }
    .settings-option select:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 20%, transparent);
    }

    .settings-item { 
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.75rem;
      border-radius: 0.5rem;
      cursor: pointer;
      transition: all 0.2s;
      margin-bottom: 0.5rem;
    }
    .settings-item:hover {
      background: rgba(99, 102, 241, 0.1);
    }
    .settings-item i {
      width: 20px;
      text-align: center;
      color: var(--subtext);
    }
    .settings-item span {
      flex: 1;
    }
    .tts-btn {
      background: transparent;
      border: none;
      color: var(--primary);
      cursor: pointer;
      padding: 0.25rem;
      margin-left: 0.5rem;
    }
    .tts-btn:hover {
      color: var(--primary-hover);
    }
    .tts-download-btn {
      background: var(--success);
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 0.5rem;
      margin-top: 0.5rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.8rem;
      transition: all 0.2s;
      text-decoration: none;
    }
    .tts-download-btn:hover {
      background: #0d9e6e;
    }
    .tts-controls {
      display: flex;
      gap: 0.5rem;
      margin-top: 0.5rem;
      justify-content: center; 
    }
    .play-tts-btn {
      background: var(--primary);
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: 0.5rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.8rem;
      transition: all 0.2s;
    }
    .play-tts-btn:hover {
      background: var(--primary-hover);
    }
    .play-tts-btn.playing {
      background: var(--warning);
    }
    .retry-btn { 
      background: transparent;
      border: none;
      color: var(--primary);
      cursor: pointer;
      padding: 0.25rem;
      margin-left: 0.5rem;
      font-size: 0.8rem;
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }
    .retry-btn:hover {
      color: var(--primary-hover);
      text-decoration: underline;
    }
    .message-actions {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      display: flex;
      gap: 0.25rem;
      opacity: 0;
      transition: opacity 0.2s;
    }
    .message-bot:hover .message-actions,
    .message-error:hover .message-actions {
      opacity: 1;
    }
    .copy-btn, .retry-btn-small {
      background: rgba(255, 255, 255, 0.1);
      border: none;
      color: var(--text);
      padding: 0.25rem 0.5rem;
      border-radius: 0.25rem;
      cursor: pointer;
      font-size: 0.7rem; 
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }
    .copy-btn:hover {
      background: rgba(255, 255, 255, 0.2);
    }
    .copy-btn.copied {
      background: var(--success);
      color: white;
    }
    .retry-btn-small:hover { 
      background: rgba(99, 102, 241, 0.2);
       color: var(--primary);
    }
    .typing-text {
      color: var(--subtext);
      font-style: italic;
    }
    .attached-file-indicator {
      font-size: 0.8rem;
      padding: 0.3rem 0.6rem;
      border-radius: 0.3rem;
      margin-top: 0.5rem;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .message-user .attached-file-indicator {
      color: #e0e0ff;
      background-color: rgba(255, 255, 255, 0.2);
    }
    .message-user .attached-file-indicator i {
      font-size: 0.75rem;
    }
    .message-actions-feedback {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.75rem;
        justify-content: flex-start; 
    }
    .action-btn {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--border);
        color: var(--subtext);
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        cursor: pointer;
        font-size: 0.75rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        transition: all 0.2s;
    }
    .action-btn:hover {
        background: rgba(255, 255, 255, 0.1);
        color: var(--text);
        border-color: var(--primary);
    }
    .action-btn.liked, .action-btn.disliked { 
        color: var(--success);
        border-color: var(--success);
    }
     .action-btn.disliked { 
        color: var(--error);
        border-color: var(--error);
    }
    .continue-btn {
        background-color: var(--primary);
        color: white;
        border-color: var(--primary-hover);
    }
    .continue-btn:hover {
        background-color: var(--primary-hover);
    }

    /* START: Image Generation Indicator Styles */
    #imageGenIndicator { /* This ID might be for a specific message, not global */
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        padding: 1.5rem;
        border-radius: 1.125rem 1.125rem 1.125rem 0; /* Match bot message bubble */
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 1rem; /* Ensure some space if messages follow quickly */
    }
    .image-gen-visual-container { /* Applied to the art plate */
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.75rem;
        animation: imageGenContainerPulse 2.5s infinite ease-in-out; /* ADDED Animation */
    }
     @keyframes imageGenContainerPulse { /* ADDED Keyframes */
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.03); }
    }
    .image-gen-icon {
        font-size: 2.5rem; /* Larger icon */
        color: white;
        filter: drop-shadow(0 0 8px rgba(255,255,255,0.5));
        animation: imageGenIconPulse 2s infinite ease-in-out;
    }
    @keyframes imageGenIconPulse { /* MODIFIED for more flair */
        0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.85; }
        50% { transform: scale(1.15) rotate(5deg); opacity: 1; }
    }
    .image-gen-text {
        font-size: 1rem;
        font-weight: 500;
        margin: 0.5rem 0;
    }
    .image-gen-progress-bar-container {
        width: 80%;
        max-width: 250px;
        height: 8px;
        background-color: rgba(255,255,255,0.25);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .image-gen-progress-bar-fill {
        width: 0%;
        height: 100%;
        background-color: white;
        border-radius: 4px;
        animation: fillImageGenProgress 5s linear infinite; 
    }
    @keyframes fillImageGenProgress {
        0% { width: 0%; opacity: 0.8; }
        50% { width: 100%; opacity: 1;}
        100% { width: 0%; opacity: 0.8;}
    }
    /* END: Image Generation Indicator Styles */


    @media (min-width: 768px) {
      .sidebar {
        position: relative;
        transform: translateX(0);
      }
      .sidebar-overlay {
        display: none;
      }
    }
    @media (max-width: 767px) {
      .sidebar {
        width: 85vw;
        max-width: 320px;
      }
      .message {
        max-width: 90%;
      }
      .message-actions { 
        opacity: 1; 
      }
      .suggestion-prompts-container {
        grid-template-columns: 1fr;
      }
      .model-selector {
      }
      .scroll-down-btn {
        right: 0.5rem;
      }
      .message-actions-feedback {
        flex-wrap: wrap;
      }
      .settings-view {
        width: calc(100vw - 2rem); 
        right: 1rem; left: 1rem;
        top: 1rem; 
        max-height: calc(100vh - 2rem); 
      }
      .settings-view-header h2 {
        font-size: 1.1rem; 
      }
      .toast-notification-container {
        top: 10px; 
        width: auto; 
        padding: 0 10px; 
      }
      .toast-notification {
        min-width: unset; 
        width: 100%; 
      }
      .input-container {
        margin: 0.5rem; 
      }
    }

    @media (max-width: 576px) {
      .input-container {
        padding: 0.6rem;
        gap: 0.6rem;
      }
    }

    @media (max-width: 399px) {
      .header h1 {
        font-size: 1.25rem;
      }
      .header h1 i {
        font-size: 1.1rem;
      }
      .header-actions .header-btn {
        width: 32px;
        height: 32px;
      }
      .header-actions .header-btn i {
          font-size: 0.9rem;
      }
       .input-container {
        margin: 0.25rem;
        padding: 0.5rem; 
        gap: 0.5rem;    
      }
      .input-box {
        padding: 0.7rem 0.9rem; 
        min-height: 40px;    
      }
      .input-controls-row {
        gap: 0.35rem;
      }
      .input-icon-btn, .send-btn {
        width: 36px;
        height: 36px;
      }
       .input-icon-btn i, .send-btn i {
          font-size: 0.8rem;
      }
    }
    @media (max-width: 359px) {
      .header {
        padding: 0.75rem 1rem;
      }
      .sidebar {
        padding: 0.75rem;
      }
      .chat-messages {
        padding: 1rem;
      }
      .about-panel, .model-selector, .settings-view {
        padding: 1rem;
      }
      .home-screen-container {
          padding: 1rem;
      }
      .suggestion-prompts-container {
          gap: 0.75rem;
      }
      .suggestion-prompt {
          padding: 1rem;
      }
      .input-container {
        padding: 0.4rem; 
        gap: 0.4rem;     
        margin: 0.25rem;
        border-radius: 25px; 
      }
      .input-box {
        padding: 0.6rem 0.8rem; 
        min-height: 38px;    
        border-radius: 18px; 
      }
      .input-controls-row {
        gap: 0.25rem; 
        padding: 0 0.1rem;
      }
      .input-icon-btn, .send-btn {
        width: 34px; 
        height: 34px;
      }
      .input-icon-btn i, .send-btn i {
          font-size: 0.8rem; 
      }
      .file-display-area {
        padding: 0.4rem 0.6rem;
      }
      .file-info {
        font-size: 0.8rem;
      }
      .close-file {
        width: 28px;
        height: 28px;
      }
       .close-file i {
        font-size: 0.9rem;
      }
      .status-indicator {
        font-size: 0.8rem;
        padding: 0.05rem 0.25rem;
      }
    }
    ::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    ::-webkit-scrollbar-track {
      background: var(--bg);
    }
    ::-webkit-scrollbar-thumb {
      background: var(--border);
      border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: var(--primary);
    }
    .home-screen-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      margin-top: auto;
      margin-bottom: auto;
      width: 100%;
      padding: 2rem;
      text-align: center;
      animation: fadeIn 0.5s ease-out;
      background: linear-gradient(135deg, var(--bg) 0%, color-mix(in srgb, var(--bg) 85%, var(--primary) 15%) 100%);
    }
    .home-screen-subtitle {
      font-size: 1.6rem;
      font-weight: 500;
      color: var(--text);
      margin-top: 1rem;
      margin-bottom: 2.5rem;
      max-width: 600px;
    }
    .suggestion-prompts-container {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 1.25rem;
      width: 100%;
      max-width: 800px;
    }
    .suggestion-prompt {
      background: var(--card);
      border: 1px solid var(--border);
      padding: 1.25rem;
      border-radius: 0.75rem;
      cursor: pointer;
      transition: all 0.2s ease;
      font-size: 0.9rem;
      text-align: left;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      min-height: 120px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.07);
    }
    .suggestion-prompt:hover {
      transform: translateY(-4px);
      box-shadow: 0 6px 14px rgba(0,0,0,0.15);
      border-color: var(--primary);
    }
    .suggestion-prompt strong {
        display: block;
        margin-bottom: 0.5rem;
        color: var(--text);
        font-weight: 600;
        font-size: 1.05em;
    }
    .suggestion-prompt p {
        font-size: 0.85rem;
        color: var(--subtext);
        line-height: 1.5;
    }
    .about-panel {
      position: absolute;
      top: calc(1rem + 36px + 0.5rem);
      right: 1rem;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 1.5rem;
      width: 320px;
      max-width: calc(100vw - 2rem);
      max-height: calc(100vh - 6rem - 50px);
      overflow-y: auto;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      z-index: 10;
      transform: translateY(10px);
      opacity: 0;
      pointer-events: none;
      transition: all 0.2s ease;
    }
    .about-panel.visible {
      transform: translateY(0);
      opacity: 1;
      pointer-events: auto;
    }
    .about-panel h4 {
      margin-bottom: 1rem;
      font-size: 1.1rem;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 0.5rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.75rem;
    }
    .about-content {
      font-size: 0.9rem;
      line-height: 1.7;
    }
    .about-content p,
    .about-content h5,
    .about-content ul {
      margin-bottom: 0.75rem;
    }
    .about-content h5 {
      font-size: 1rem;
      color: var(--primary);
      margin-top: 1rem;
    }
    .about-content ul {
      list-style: none;
      padding-left: 0;
    }
    .about-content ul li {
      margin-bottom: 0.4rem;
      padding-left: 1.25rem;
      position: relative;
    }
    .about-content ul li::before {
      content: "\f00c";
      font-family: "Font Awesome 6 Free";
      font-weight: 900;
      position: absolute;
      left: 0;
      top: 2px;
      color: var(--primary);
      font-size: 0.8em;
    }
    .about-content code {
      background: var(--highlight);
      color: var(--success);
      padding: 0.2em 0.4em;
      border-radius: 3px;
      font-size: 0.85em;
    }
    .feedback-modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s ease;
    }
    .feedback-modal.visible {
        opacity: 1;
        pointer-events: auto;
    }
    .feedback-modal-content {
        background: var(--card);
        padding: 1.5rem 2rem;
        border-radius: 0.75rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        width: 90%;
        max-width: 500px;
        border: 1px solid var(--border);
    }
    .feedback-modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
    }
    .feedback-modal-header h4 {
        margin: 0;
        font-size: 1.1rem;
        color: var(--text);
    }
    .feedback-modal-close-btn {
        background: none;
        border: none;
        font-size: 1.5rem;
        color: var(--subtext);
        cursor: pointer;
        transition: color 0.2s;
    }
    .feedback-modal-close-btn:hover {
        color: var(--text);
    }
    .feedback-modal-body label {
        display: block;
        font-size: 0.9rem;
        color: var(--subtext);
        margin-bottom: 0.5rem;
    }
    .feedback-modal-body textarea {
        width: 100%;
        min-height: 100px;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid var(--border);
        background-color: var(--bg);
        color: var(--text);
        font-family: inherit;
        resize: vertical;
        margin-bottom: 1rem;
    }
    .feedback-modal-body textarea:focus {
        outline: none;
        border-color: var(--primary);
    }
    .feedback-modal-footer {
        text-align: right;
    }
    .feedback-submit-btn {
        background-color: var(--primary);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 0.5rem;
        cursor: pointer;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .feedback-submit-btn:hover {
        background-color: var(--primary-hover);
    }
    .feedback-submit-btn:disabled {
        background-color: var(--subtext);
        cursor: not-allowed;
    }
    .sidebar-user-profile {
        margin-top: auto;
        padding: 0.85rem 1rem;
        border-top: 1px solid var(--border);
        display: flex;
        align-items: flex-start; 
        gap: 0.75rem;
        background: var(--card);
        position: sticky; 
        bottom: 0; 
        z-index: 15;
        flex-shrink: 0; 
    }
    .sidebar-user-profile img {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid var(--primary);
    }
    .sidebar-user-profile span {
        font-size: 0.9rem;
        color: var(--text);
        flex-grow: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        padding-top: 0.15rem; 
    }
    .user-profile-actions {
        position: relative; 
        flex-shrink: 0;
    }
    .auth-btn-menu { 
        background: transparent;
        border: none;
        color: var(--subtext);
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: background-color 0.2s, color 0.2s;
    }
    .auth-btn-menu:hover {
        background: var(--highlight);
        color: var(--primary);
    }
    .user-profile-menu {
        position: absolute;
        bottom: calc(100% + 5px); 
        right: 0;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 0.375rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        z-index: 25; 
        padding: 0.25rem 0;
        min-width: 150px;
        display: none; 
        flex-direction: column;
    }
    .user-profile-menu.visible {
        display: flex;
    }
    .user-profile-menu-item {
        background: none;
        border: none;
        color: var(--text);
        padding: 0.6rem 0.85rem;
        text-align: left;
        cursor: pointer;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        width: 100%;
        transition: background-color 0.2s, color 0.2s;
    }
    .user-profile-menu-item i {
        width: 16px;
        text-align: center;
        color: var(--subtext);
        transition: color 0.2s;
    }
    .user-profile-menu-item:hover {
        background: var(--highlight);
        color: var(--primary);
    }
    .user-profile-menu-item:hover i {
        color: var(--primary);
    }
    .user-profile-menu-item#menuSignOutBtn:hover {
        color: var(--error);
        background: rgba(239, 68, 68, 0.05);
    }
    .user-profile-menu-item#menuSignOutBtn:hover i {
        color: var(--error);
    }

    .confirm-modal {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0, 0, 0, 0.6);
      display: none; 
      align-items: center;
      justify-content: center;
      z-index: 2000; 
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s ease;
    }
    .confirm-modal.visible {
      display: flex; 
      opacity: 1;
      pointer-events: auto;
    }
    .confirm-modal-content {
      background: var(--card);
      padding: 1.5rem 2rem;
      border-radius: 0.75rem;
      box-shadow: 0 5px 15px rgba(0,0,0,0.2);
      width: 90%;
      max-width: 400px;
      border: 1px solid var(--border);
      text-align: center;
    }
    .confirm-modal-content h4 {
      margin-top: 0;
      margin-bottom: 1rem;
      font-size: 1.2rem;
      color: var(--text);
    }
    .confirm-modal-content p {
      margin-bottom: 1.5rem;
      color: var(--subtext);
      font-size: 0.95rem;
      line-height: 1.5;
    }
    .confirm-modal-actions {
      display: flex;
      justify-content: flex-end;
      gap: 0.75rem;
    }
    .confirm-modal-actions button {
      padding: 0.6rem 1.2rem;
      border: none;
      border-radius: 0.5rem;
      cursor: pointer;
      font-weight: 500;
      transition: background-color 0.2s;
      font-size: 0.9rem;
    }
    .confirm-modal-actions .btn-secondary {
      background-color: var(--border);
      color: var(--text);
    }
    .confirm-modal-actions .btn-secondary:hover {
      background-color: color-mix(in srgb, var(--border) 80%, black);
    }
    .confirm-modal-actions .btn-danger {
      background-color: var(--error);
      color: white;
    }
    .confirm-modal-actions .btn-danger:hover {
      background-color: color-mix(in srgb, var(--error) 80%, black);
    }
    .confirm-modal-actions .btn-primary { /* Added for rename save button */
      background-color: var(--primary);
      color: white;
    }
    .confirm-modal-actions .btn-primary:hover {
      background-color: var(--primary-hover);
    }


    body.light-theme {
      --bg: #f8fafc;
      --card: #ffffff;
      --border: #e2e8f0;
      --text: #1e293b;
      --subtext: #64748b;
    }
    body.light-theme .input-container {
        background: var(--card);
        border-color: var(--border);
    }
    body.light-theme .input-box {
        background: var(--bg); 
        border-color: var(--border);
        color: var(--text);
    }
    body.light-theme .input-box:focus {
      box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 20%, transparent);
    }
    body.light-theme .input-icon-btn {
        background: var(--bg); 
        color: var(--text);
        border-color: var(--border);
    }
    body.light-theme .input-icon-btn:hover:not(:disabled) {
        background: var(--primary);
        color: white;
    }
    body.light-theme .file-display-area {
        background: var(--bg); 
        border-color: var(--border);
    }
    body.light-theme .message-bot {
      background: white;
      border: 1px solid var(--border);
    }
    body.light-theme .message-user .message-time {
        color: rgba(25,35,55,0.7);
    }
    body.light-theme .message-model {
      background: var(--bg);
    }
    body.light-theme .message-bot pre {
      background: #f1f5f9;
    }
    body.light-theme .message-bot pre code {
      color: #334155;
    }
    body.light-theme .file-info.has-file {
      color: var(--primary);
    }
    body.light-theme .copy-btn,
    body.light-theme .retry-btn-small {
      background: rgba(0, 0, 0, 0.05);
      color: var(--text);
    }
     body.light-theme .retry-btn-small:hover {
      background: rgba(99, 102, 241, 0.2); 
      color: var(--primary);
    }
    body.light-theme .copy-code-btn,
    body.light-theme .run-code-btn {
      background: rgba(0, 0, 0, 0.05);
      color: var(--text);
    }
    body.light-theme .copy-code-btn:hover {
      background: rgba(0, 0, 0, 0.1);
    }
    body.light-theme .copy-code-btn.copied {
      background: var(--success);
      color: white;
    }
    body.light-theme .run-code-btn:hover {
      background: var(--primary);
      color: white;
    }
    body.light-theme .code-output-area {
      background: #f8fafc;
      border-color: var(--border);
      color: var(--text);
    }
    body.light-theme .code-output-area iframe {
      background-color: white;
    }
    body.light-theme .cut-code-output-btn {
      background: rgba(0, 0, 0, 0.05);
      color: var(--subtext);
      border-color: var(--border);
    }
    body.light-theme .cut-code-output-btn:hover {
      background: var(--error);
      color: white;
      border-color: var(--error);
    }
    body.light-theme .suggestion-prompt {
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    body.light-theme .suggestion-prompt:hover {
        box-shadow: 0 6px 14px rgba(0,0,0,0.1);
    }
    body.light-theme .about-panel,
    body.light-theme .settings-view { 
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
     body.light-theme .about-content ul li::before {
      color: var(--primary);
    }
    body.light-theme .home-screen-container {
        background: linear-gradient(135deg, var(--bg) 0%, color-mix(in srgb, var(--bg) 85%, var(--primary) 10%) 100%);
    }
    body.light-theme .message-user .attached-file-indicator {
        color: var(--primary-hover);
        background-color: rgba(0,0,0,0.05);
    }
    body.light-theme .action-btn {
        background: rgba(0, 0, 0, 0.03);
        border-color: var(--border);
        color: var(--subtext);
    }
    body.light-theme .action-btn:hover {
        background: rgba(0, 0, 0, 0.06);
        color: var(--text);
        border-color: var(--primary);
    }
    body.light-theme .action-btn.liked, body.light-theme .action-btn.disliked {
        color: var(--success);
        border-color: var(--success);
    }
    body.light-theme .action-btn.disliked { 
        color: var(--error);
        border-color: var(--error);
    }
    body.light-theme .continue-btn {
        background-color: var(--primary);
        color: white;
        border-color: var(--primary-hover);
    }
    body.light-theme .continue-btn:hover {
        background-color: var(--primary-hover);
    }
    body.light-theme .feedback-modal-content {
        background: var(--card);
        border: 1px solid var(--border);
    }
    body.light-theme .feedback-modal-body textarea {
        background-color: var(--bg);
        color: var(--text);
        border: 1px solid var(--border);
    }
    body.light-theme .sidebar-user-profile {
        background: var(--card);
        border-top-color: var(--border);
    }
    body.light-theme .sidebar-user-profile img {
        border: 2px solid var(--primary); 
    }
    body.light-theme .sidebar-user-profile span {
        color: var(--text);
    }
    body.light-theme .settings-view { 
        background-color: var(--card);
    }
    body.light-theme .settings-view-header .header-btn {
        background: var(--card); 
        color: var(--text);
        border: 1px solid var(--border);
    }
    body.light-theme .settings-view-header .header-btn:hover {
        background: var(--primary);
        color: white;
    }

    body.light-theme .chat-item-menu-btn:hover,
    body.light-theme .chat-item-menu-option:hover,
    body.light-theme .user-profile-menu-item:hover {
        background: var(--highlight);
        color: var(--primary);
    }
    body.light-theme .chat-item-menu-option:hover i,
    body.light-theme .user-profile-menu-item:hover i {
      color: var(--primary);
    }
    body.light-theme .chat-item-menu-option.delete-chat-btn:hover,
    body.light-theme .user-profile-menu-item#menuSignOutBtn:hover {
        color: var(--error);
        background: rgba(239, 68, 68, 0.05);
    }
    body.light-theme .chat-item-menu-option.delete-chat-btn:hover i,
    body.light-theme .user-profile-menu-item#menuSignOutBtn:hover i {
        color: var(--error);
    }
    body.light-theme .confirm-modal-content { 
        background: var(--card);
        border: 1px solid var(--border);
    }
    body.light-theme .confirm-modal-content h4 { color: var(--text); }
    body.light-theme .confirm-modal-content p { color: var(--subtext); }
    body.light-theme .confirm-modal-actions .btn-secondary {
        background-color: var(--border); color: var(--text);
    }
    body.light-theme .confirm-modal-actions .btn-secondary:hover {
        background-color: color-mix(in srgb, var(--border) 90%, #000000 10%);
    }
    body.light-theme .confirm-modal-actions .btn-primary { /* Ensure light theme consistency */
        background-color: var(--primary); color: white;
    }
    body.light-theme .confirm-modal-actions .btn-primary:hover {
        background-color: var(--primary-hover);
    }

    .katex { font-size: 1.1em !important; }
    body.light-theme .katex { color: var(--text) !important; }
    body.dark-theme .katex { color: var(--text) !important; } 
    .katex .mfrac .frac-line { border-bottom-color: var(--text) !important; }
    .katex-display {
      margin: 0.75em 0 !important;
      overflow-x: auto !important;
      overflow-y: hidden !important;
    }
    .katex-display > .katex {
      display: inline-block !important;
      white-space: nowrap !important;
      text-align: left !important;
    }
    .katex .base {
      margin: 0.15em 0 !important;
    }
    .scroll-down-btn {
      position: absolute;
      right: 1rem;
      background: var(--primary);
      color: white;
      width: 40px;
      height: 40px;
      border-radius: 50%;
      border: none;
      display: none;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: 0 2px 5px rgba(0,0,0,0.2);
      transition: opacity 0.3s, transform 0.3s, bottom 0.2s ease-out, background-color 0.2s, color 0.2s;
      z-index: 20;
      font-size: 1rem;
    }
    .scroll-down-btn:hover {
      background: var(--primary-hover);
      transform: translateY(-2px) scale(1.05);
    }
    .scroll-down-btn.visible {
      display: flex;
    }
    body.dark-theme .scroll-down-btn {
        background-color: #f8fafc;
        color: #0f172a;
    }
    body.dark-theme .scroll-down-btn:hover {
        background-color: #e2e8f0;
    }
    body.light-theme .scroll-down-btn {
        background-color: #1e293b;
        color: #f8fafc;
    }
    body.light-theme .scroll-down-btn:hover {
        background-color: #334155;
    }
  </style>
</head>
<body class="app-hidden">

  <div class="login-prompt-modal" id="loginPromptModal">
    <div class="login-prompt-content">
        <h2><i class="fas fa-brain"></i> Welcome to JARVIS 5.0</h2>
        <p>Please sign in with your Google account to continue and access your personalized AI assistant.</p>
        <button class="login-prompt-btn" id="promptSignInBtn">
            <i class="fab fa-google"></i> Sign In with Google
        </button>
    </div>
  </div>

  <div class="header">
    <h1 id="appTitle"><i class="fas fa-robot"></i> JARVIS 5.0</h1>
    <div class="header-actions">
      <button class="header-btn" id="menuBtn" title="Toggle Sidebar"><i class="fas fa-bars"></i></button>
      <button class="header-btn" id="newChatBtnHeader" title="New Chat"><i class="fas fa-pen-to-square"></i></button>
      <button class="header-btn" id="aboutAppBtn" title="About JARVIS 5.0"><i class="fas fa-info-circle"></i></button>
    </div>
  </div>

  <div class="main-container">
    <div class="sidebar-overlay" id="sidebarOverlay"></div>
    <div class="sidebar" id="sidebar">
      <div class="sidebar-header">
        <div class="sidebar-title">Chat History</div>
        <button class="close-sidebar" id="closeSidebar"><i class="fas fa-times"></i></button>
      </div>

      <button class="new-chat-btn" id="sidebarNewChatBtn">
        <i class="fas fa-pen-to-square"></i> New Chat
      </button>

      <div class="chat-history" id="chatHistory">
      </div>

      <div class="sidebar-user-profile" id="sidebarUserProfile">
          <img id="userProfileImage" src="https://via.placeholder.com/32/cccccc/000000?text=U" alt="User Profile">
          <span id="userProfileName">User</span>
          <div class="user-profile-actions">
              <button id="userProfileMenuBtn" class="auth-btn-menu" title="User options"><i class="fas fa-ellipsis-h"></i></button>
              <div class="user-profile-menu" id="userProfileMenu">
                  <button id="menuSettingsBtn" class="user-profile-menu-item"><i class="fas fa-cog"></i> Settings</button>
                  <button id="menuUpgradeBtn" class="user-profile-menu-item"><i class="fas fa-rocket"></i> Upgrade Plan</button>
                  <button id="menuSignOutBtn" class="user-profile-menu-item"><i class="fas fa-sign-out-alt"></i> Log Out</button>
              </div>
          </div>
      </div>
    </div>

    <div class="chat-container" id="chatContainer">
      <div class="settings-view" id="settingsView"> 
        <div class="settings-view-header">
            <h2 id="settingsViewTitle"><i class="fas fa-cog"></i> Settings</h2>
            <button class="header-btn" id="closeSettingsViewBtn" title="Close Settings"><i class="fas fa-times"></i></button> 
        </div>
        <div class="settings-view-content">
            <div class="settings-option">
              <label for="themeSelect">Theme</label>
              <select id="themeSelect">
                <option value="dark">Dark</option>
                <option value="light">Light</option>
                <option value="system">System</option>
              </select>
            </div>
            <div class="settings-option">
              <label for="fontSizeSelect">Font Size</label>
              <select id="fontSizeSelect">
                <option value="small">Small</option>
                <option value="medium" selected>Medium</option>
                <option value="large">Large</option>
              </select>
            </div>
            <div class="settings-option">
              <label for="modelSelectGlobal">Default Model</label>
              <select id="modelSelectGlobal">
                <option value="auto">Auto (Recommended)</option>
                <option value="groq">Groq (Fastest)</option>
                <option value="openrouter">OpenRouter (Free)</option>
              </select>
            </div>
        </div>
      </div>
      
      <div class="chat-messages" id="chatMessages">
      </div>

      <div class="input-container" id="inputContainer">
        <div class="file-display-area" id="fileDisplayArea" style="display: none;">
          <div class="file-info" id="fileInfo">No file selected</div>
          <button class="close-file" id="closeFile">
            <i class="fas fa-times"></i>
          </button>
        </div>

        <textarea class="input-box" id="messageInput" placeholder="Type your message or /help for commands..." rows="1"></textarea>
        
        <div class="input-controls-row">
          <button class="input-icon-btn" id="webSearchBtn" title="Perform Web Search">
            <i class="fas fa-globe"></i>
          </button>
          <button class="input-icon-btn" id="imageGenBtn" title="Generate Image (e.g., /image sunset)"> 
            <i class="fas fa-palette"></i>
          </button>
          <button class="input-icon-btn model-btn" id="modelBtn" title="Change AI Model">
            <i class="fas fa-robot"></i>
          </button>
          <button class="input-icon-btn attach-btn" id="attachFileIconBtn" title="Attach File">
            <i class="fas fa-paperclip"></i>
          </button>
          <input type="file" id="fileInput" accept="image/*,.pdf,.txt,.docx" style="display: none;" />
          <button class="send-btn" id="sendBtn" disabled>
            <i class="fas fa-paper-plane"></i>
          </button>
        </div>

        <div class="status-indicator" id="statusText"></div>
      </div>

      <button class="scroll-down-btn" id="scrollDownBtn" title="Scroll to Bottom">
        <i class="fas fa-arrow-down"></i>
      </button>
    </div>
  </div>

  <div class="model-selector" id="modelSelector">
    <h4>Select AI Model</h4>
    <div class="model-option" data-model="groq">
      <i class="fas fa-bolt"></i>
      <span>Groq (Mixtral) - Fastest</span>
      <i class="fas fa-check checkmark"></i>
    </div>
    <div class="model-option" data-model="openrouter">
      <i class="fab fa-docker"></i>
      <span>OpenRouter (DeepSeek) - Free</span>
      <i class="fas fa-check checkmark"></i>
    </div>
    <div class="model-option" data-model="auto">
      <i class="fas fa-sync-alt"></i>
      <span>Auto (Recommended)</span>
      <i class="fas fa-check checkmark"></i>
    </div>
  </div>

  <div class="about-panel" id="aboutPanel">
    <h4><i class="fas fa-info-circle"></i> About JARVIS 5.0</h4>
    <div class="about-content">
      <p><strong>JARVIS 5.0</strong> is an advanced AI assistant designed to enhance your productivity and provide intelligent support across various tasks.</p>
    <h5>Core Features:</h5>
      <ul>
        <li><strong>Multi-Model Support:</strong> Seamlessly switch between Groq (Fastest), OpenRouter (Free), or Auto mode for optimal performance.</li>
        <li><strong>Persistent Chat History:</strong> Conversations are saved to your Google account via Firebase. Chats are automatically grouped by timeframes (Today, Yesterday, etc.).</li>
        <li><strong>Intelligent File Handling:</strong> Attach images (JPG, PNG, GIF), PDFs, TXT, or DOCX files. JARVIS extracts relevant text for context-aware responses.</li>
        <li><strong>Rich Text Formatting:</strong> Enjoy clear and readable responses with Markdown support, syntax highlighting for code blocks, and mathematical expression rendering. Important words in AI responses can be highlighted for emphasis.</li>
        <li><strong>Text-to-Speech (TTS):</strong> Listen to AI responses aloud, or generate downloadable MP3 audio for any text.</li>
        <li><strong>AI Image Generation:</strong> Create unique images directly within the chat using the <code>/image [your prompt]</code> command or the dedicated image generation button.</li>
        <li><strong>Real-time Web Search:</strong> JARVIS can perform web searches to provide you with up-to-date information (via `/search` command or dedicated button).</li>
        <li><strong>Integrated Translation:</strong> Get text translated into various languages on demand.</li>
        <li><strong>Live Voice Chat:</strong> Engage in live voice conversations by using the <code>/gui</code> command.</li>
      </ul>
      <h5>GUI Overview:</h5>
      <ul>
        <li><strong>Header Controls:</strong> Quickly access the sidebar menu, initiate a new chat session, or open this "About" panel.</li>
        <li><strong>Sidebar Navigation:</strong> Manage your chat history, automatically grouped by date, with options to rename, share, or delete chats. Login with Google is required.</li>
        <li><strong>Interactive Chat Area:</strong> View the conversation flow and interact directly with JARVIS.</li>
        <li><strong>Dynamic Input Area:</strong> Compose messages, attach files with previews, select your preferred AI model, perform web searches, generate images, and send your queries using the redesigned input pill. UI elements are optimized during AI responses.</li>
        <li><strong>User Profile Options:</strong> Access settings or log out (with confirmation) via the menu in the sidebar user profile section.</li>
      </ul>
      <p><em>Powered by cutting-edge AI and modern web technologies to deliver a responsive and intelligent user experience.</em></p>
    </div>
  </div>

  <div class="feedback-modal" id="feedbackModal">
    <div class="feedback-modal-content">
        <div class="feedback-modal-header">
            <h4>Report an Issue / Provide Feedback</h4>
            <button class="feedback-modal-close-btn" id="closeFeedbackModalBtn">&times;</button>
        </div>
        <div class="feedback-modal-body">
            <label for="feedbackText">Please describe the issue or your feedback:</label>
            <textarea id="feedbackText" rows="5" placeholder="E.g., The response was inaccurate, irrelevant, or I encountered a bug..."></textarea>
        </div>
        <div class="feedback-modal-footer">
            <button class="feedback-submit-btn" id="submitFeedbackBtn">Submit Report</button>
        </div>
    </div>
  </div>

  <div class="confirm-modal" id="confirmDeleteModal"> 
    <div class="confirm-modal-content">
      <h4>Confirm Deletion</h4>
      <p id="confirmDeleteMessage">Are you sure you want to delete this chat?</p>
      <div class="confirm-modal-actions">
        <button id="cancelDeleteBtn" class="btn-secondary">Cancel</button>
        <button id="confirmDeleteBtn" class="btn-danger">Delete</button>
      </div>
    </div>
  </div>

  <div class="confirm-modal" id="confirmLogoutModal"> 
    <div class="confirm-modal-content">
      <h4>Confirm Log Out</h4>
      <p id="confirmLogoutMessage">Are you sure you want to log out?</p>
      <div class="confirm-modal-actions">
        <button id="cancelLogoutBtn" class="btn-secondary">Cancel</button>
        <button id="confirmLogoutBtn" class="btn-danger">Log Out</button>
      </div>
    </div>
  </div>

  <div class="confirm-modal" id="renameChatModal">
    <div class="confirm-modal-content">
      <h4>Rename Chat</h4>
      <p>Enter a new name for this chat:</p>
      <input type="text" id="renameChatInput" class="input-box" style="width: 100%; margin-bottom: 1rem; background: var(--bg); border-color: var(--border); color: var(--text);" placeholder="New chat name">
      <div class="confirm-modal-actions">
        <button id="cancelRenameBtn" class="btn-secondary">Cancel</button>
        <button id="confirmRenameBtn" class="btn-primary">Save</button>
      </div>
    </div>
  </div>

  <div id="toastNotificationContainer" class="toast-notification-container">
  </div>

<script>
// Initialize libraries
if (window.markedMath && typeof window.katex !== 'undefined') {
  try {
    marked.use(window.markedMath);
    console.log("Marked-math extension registered successfully with Marked.");
  } catch (e) {
    console.error("Error registering marked-math extension with Marked:", e);
  }
} else {
  let missingLibraries = [];
  if (!window.markedMath) missingLibraries.push("marked-math");
  if (typeof window.katex === 'undefined') missingLibraries.push("KaTeX");
  if (missingLibraries.length > 0) console.warn(`Missing libraries for math rendering: ${missingLibraries.join(' and ')}`);
}

marked.setOptions({
  breaks: true,
  gfm: true,
  highlight: function(code, lang) {
    const language = lang || 'plaintext';
    if (typeof Prism !== 'undefined' && Prism.languages && Prism.languages[language]) {
      return Prism.highlight(code, Prism.languages[language], language);
    } else if (typeof Prism !== 'undefined' && Prism.util && Prism.util.encode) {
      return Prism.util.encode(code);
    } else {
      return code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    }
  },
  katexOptions: { throwOnError: false }
});

const allSuggestionPrompts = [
  { title: "Explain Simply", description: "The basics of quantum computing for a beginner.", prompt: "Explain quantum computing in simple terms" },
  { title: "Write Code", description: "A Python script for organizing files by their type.", prompt: "Write a Python script to organize files in a directory by extension" },
  { title: "Get Ideas", description: "Quick and healthy dinner recipes for weekdays.", prompt: "What are some healthy and quick dinner ideas for a busy weeknight?" },
  { title: "Translate Text", description: "A common greeting translated into French.", prompt: "Translate 'Good morning, have a wonderful day!' to French" }
];

function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

// DOM elements
const loginPromptModal = document.getElementById('loginPromptModal');
const promptSignInBtn = document.getElementById('promptSignInBtn');
const appBody = document.body;
const appTitle = document.getElementById('appTitle');

const menuBtn = document.getElementById('menuBtn');
const newChatBtnHeader = document.getElementById('newChatBtnHeader');
const aboutAppBtn = document.getElementById('aboutAppBtn');
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const closeSidebar = document.getElementById('closeSidebar');
const sidebarNewChatBtn = document.getElementById('sidebarNewChatBtn');
const chatHistory = document.getElementById('chatHistory');
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const modelBtn = document.getElementById('modelBtn');
const webSearchBtn = document.getElementById('webSearchBtn'); 
const imageGenBtn = document.getElementById('imageGenBtn');
const fileInput = document.getElementById('fileInput');
const attachFileIconBtn = document.getElementById('attachFileIconBtn');
const fileDisplayArea = document.getElementById('fileDisplayArea');
const fileInfo = document.getElementById('fileInfo');
const closeFile = document.getElementById('closeFile');
const statusText = document.getElementById('statusText'); 
const modelSelector = document.getElementById('modelSelector');
const modelOptions = document.querySelectorAll('.model-option');

const settingsView = document.getElementById('settingsView');
const closeSettingsViewBtn = document.getElementById('closeSettingsViewBtn');
const settingsViewTitle = document.getElementById('settingsViewTitle');

const aboutPanel = document.getElementById('aboutPanel');
const themeSelect = document.getElementById('themeSelect');
const fontSizeSelect = document.getElementById('fontSizeSelect');
const modelSelectGlobal = document.getElementById('modelSelectGlobal');
const scrollDownBtn = document.getElementById('scrollDownBtn');
const inputContainerElement = document.getElementById('inputContainer');
const chatContainer = document.getElementById('chatContainer');
const feedbackModal = document.getElementById('feedbackModal');
const feedbackTextEl = document.getElementById('feedbackText'); 
const closeFeedbackModalBtn = document.getElementById('closeFeedbackModalBtn');
const submitFeedbackBtn = document.getElementById('submitFeedbackBtn');
let currentFeedbackData = {};

const confirmDeleteModal = document.getElementById('confirmDeleteModal');
const confirmDeleteMessage = document.getElementById('confirmDeleteMessage');
const cancelDeleteBtn = document.getElementById('cancelDeleteBtn');
const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');
let chatIdToDelete = null;

const renameChatModal = document.getElementById('renameChatModal'); // New
let chatIdToRename = null; // New

const confirmLogoutModal = document.getElementById('confirmLogoutModal');
const cancelLogoutBtn = document.getElementById('cancelLogoutBtn');
const confirmLogoutBtn = document.getElementById('confirmLogoutBtn');


const sidebarUserProfile = document.getElementById('sidebarUserProfile');
const userProfileImage = document.getElementById('userProfileImage');
const userProfileName = document.getElementById('userProfileName');
const userProfileMenuBtn = document.getElementById('userProfileMenuBtn');
const userProfileMenu = document.getElementById('userProfileMenu');
const menuSettingsBtn = document.getElementById('menuSettingsBtn');
const menuUpgradeBtn = document.getElementById('menuUpgradeBtn'); // New
const menuSignOutBtn = document.getElementById('menuSignOutBtn');

let toastNotificationContainer; 

// State management
let currentChatId = null; 
let chats = {}; 
let extractedText = "";
let currentSearchResults = [];
let hasActiveFile = false;
let activeFileName = "";
let selectedModel = 'auto'; 
let isTypingGlobal = false; 
let userSettings = {
  theme: 'dark',
  fontSize: 'medium',
  model: 'auto'
};
let currentAudio = null;
let lastUserSentMessage = "";
let currentAbortController = null;
let isSettingsViewOpen = false;

const modelDisplayNames = { 'groq': 'Groq (Fast)', 'openrouter': 'OpenRouter (Free)', 'auto': 'Auto (Recommended)' };
const modelIcons = { 'groq': 'fas fa-bolt', 'openrouter': 'fab fa-docker', 'auto': 'fas fa-sync-alt' };

const firebaseConfig = {
  apiKey: "AIzaSyAbFIxS1MfSt1UlgHxiRBg-MYkN-SGcuQg",
  authDomain: "jarvis-57069.firebaseapp.com",
  projectId: "jarvis-57069",
  storageBucket: "jarvis-57069.firebasestorage.app",
  messagingSenderId: "252338648688",
  appId: "1:252338648688:web:7500c989e003c3eae626b5",
  databaseURL: "https://jarvis-57069-default-rtdb.firebaseio.com"
};

let auth;
let googleProvider;
let database; 
let previousAuthUserId = null;
let initialAuthCheckDone = false; 

function initializeFirebase() {
    if (typeof firebase !== 'undefined' && firebase.app && firebase.auth && firebase.database) {
        try {
            if (!firebase.apps.length) { 
                firebase.initializeApp(firebaseConfig);
                console.log("Firebase app initialized with config:", firebaseConfig);
            } else {
                firebase.app(); 
                console.log("Using existing Firebase app instance.");
            }
            auth = firebase.auth();
            database = firebase.database();
            googleProvider = new firebase.auth.GoogleAuthProvider();
            console.log("Firebase services (Auth & Database) obtained.");
            console.log("Firebase SDKs (auth, database) seem available.");


            auth.onAuthStateChanged(handleAuthStateChanged);

            if (promptSignInBtn) {
                promptSignInBtn.addEventListener('click', signInWithGoogle);
            }
            if (menuSignOutBtn) { 
                menuSignOutBtn.addEventListener('click', () => {
                    confirmLogoutModal.classList.add('visible');
                    userProfileMenu.classList.remove('visible'); 
                });
            }
        } catch (e) {
            console.error("Firebase initialization error:", e);
            showError("Could not initialize authentication/database. Please refresh.");
            if (loginPromptModal) { 
                loginPromptModal.innerHTML = `<div class="login-prompt-content"><p style="color:var(--error)">Critical Error: Firebase services could not be initialized. Please check your connection or try again later.</p></div>`;
                loginPromptModal.classList.add('visible');
                appBody.classList.add('app-hidden');
            }
        }
    } else {
        console.warn("Firebase SDK not fully loaded yet. Retrying initialization in 500ms...");
        setTimeout(initializeFirebase, 500);
    }
}

async function handleAuthStateChanged(user) {
    const currentAuthUserId = user ? user.uid : null;
    console.log("Auth state changed. User:", user ? user.uid : "null");

    if (user) { 
        updateUserProfileUI(user);
        console.log("User is authenticated. Proceeding with data load for UID:", user.uid);

        if (!initialAuthCheckDone || previousAuthUserId !== currentAuthUserId) {
            console.log("User logged in or changed:", currentAuthUserId, "Previous UID:", previousAuthUserId);
            if (statusText) statusText.innerHTML = '<i class="fas fa-spinner fa-pulse"></i> Loading your data...';
            
            chats = {}; 
            userSettings = { theme: 'dark', fontSize: 'medium', model: 'auto' }; 

            await loadUserSettingsFromFirebase(currentAuthUserId); 
            await loadChatsFromFirebase(currentAuthUserId);      
            
            setupInitialChatState(); 
            
            appBody.classList.remove('app-hidden');
            if (loginPromptModal) loginPromptModal.classList.remove('visible');
            
            if (statusText) statusText.innerHTML = "&nbsp;"; 
        } else {
            if (appBody.classList.contains('app-hidden')) {
                appBody.classList.remove('app-hidden');
            }
            if (loginPromptModal && loginPromptModal.classList.contains('visible')) {
                loginPromptModal.classList.remove('visible');
            }
            console.log("Auth state confirmed for existing user, no data reload needed as user ID and initial check indicate stability.");
        }
        previousAuthUserId = currentAuthUserId;

    } else { 
        console.log("User is signed out or not yet signed in. Clearing local data and showing login prompt.");
        appBody.classList.add('app-hidden'); 
        if (loginPromptModal) loginPromptModal.classList.add('visible'); 
        
        closeSettingsView(); 
        chats = {}; 
        currentChatId = null;
        userSettings = { theme: 'dark', fontSize: 'medium', model: 'auto' }; 
        applySettingsToUI(); 

        if(chatHistory) chatHistory.innerHTML = ''; 
        if(chatMessages) chatMessages.innerHTML = ''; 
        renderHomeScreen(); 
        updateUserProfileUI(null); 
        
        if(messageInput) messageInput.value = "";
        handleInputChange(); 
        clearFile(); 
        if(statusText) statusText.innerHTML = "&nbsp;"; 
        
        previousAuthUserId = null; 
    }
    initialAuthCheckDone = true;
}

async function loadUserSettingsFromFirebase(userId) {
    console.log(`Attempting to load settings for UID: ${userId} from path: users/${userId}/settings`);
    if (!userId || !database) {
        console.error("loadUserSettingsFromFirebase: Skipped. User ID or database not available.", {userId, databaseAvailable: !!database});
        showError("Cannot load settings: Critical error (user ID or database missing).");
        applySettingsToUI(); 
        return;
    }
    try {
        const snapshot = await database.ref(`users/${userId}/settings`).once('value');
        const firebaseSettings = snapshot.val();
        if (firebaseSettings) {
            userSettings = { ...userSettings, ...firebaseSettings }; 
            console.log("Successfully loaded settings from Firebase:", firebaseSettings);
        } else {
            console.log(`No settings found in Firebase for UID: ${userId}. Will save defaults.`);
            await saveUserSettingsToFirebase(userId, userSettings); 
        }
    } catch (error) {
        console.error(`Error loading settings for UID: ${userId} from Firebase:`, error);
        showError("Failed to load settings. Using defaults. Details in console.");
    }
    applySettingsToUI(); 
}

async function saveUserSettingsToFirebase(userId, settingsToSave) {
    console.log(`Attempting to save settings for UID: ${userId} to path: users/${userId}/settings. Data:`, settingsToSave);
    if (!userId || !database) {
        console.error("Cannot save settings: userId or database unavailable.", {userId, databaseAvailable: !!database});
        return;
    }
    try {
        await database.ref(`users/${userId}/settings`).set(settingsToSave);
        console.log("Successfully saved settings to Firebase for UID:", userId);
    } catch (error) {
        console.error(`Error saving settings for UID: ${userId} to Firebase:`, error);
        showError("Failed to save settings to your account. Details in console.");
    }
}

function applySettingsToUI() {
    if (themeSelect) themeSelect.value = userSettings.theme;
    if (fontSizeSelect) fontSizeSelect.value = userSettings.fontSize;
    if (modelSelectGlobal) modelSelectGlobal.value = userSettings.model;
    
    updateTheme(false); 
    updateFontSize(false); 
    selectedModel = userSettings.model; 
    updateModelSelection(); 
}


async function loadChatsFromFirebase(userId) {
    console.log(`Attempting to load chats for UID: ${userId} from path: users/${userId}/chats`);
    if (!userId || !database) {
        console.error("loadChatsFromFirebase: Skipped. User ID or database not available.", {userId, databaseAvailable: !!database});
        showError("Cannot load chats: Critical error (user ID or database missing). Details in console.");
        chats = {}; 
        return;
    }
    try {
        const snapshot = await database.ref(`users/${userId}/chats`).once('value');
        const firebaseChats = snapshot.val();
        if (firebaseChats && typeof firebaseChats === 'object' && Object.keys(firebaseChats).length > 0) {
            const validatedChats = {};
            let prunedCount = 0;
            for (const id in firebaseChats) {
                if (firebaseChats[id] && 
                    typeof firebaseChats[id] === 'object' && 
                    firebaseChats[id].messages && 
                    Array.isArray(firebaseChats[id].messages) &&
                    (firebaseChats[id].title !== undefined && typeof firebaseChats[id].title === 'string') && 
                    (firebaseChats[id].createdAt !== undefined && typeof firebaseChats[id].createdAt === 'string') && 
                    (firebaseChats[id].updatedAt !== undefined && typeof firebaseChats[id].updatedAt === 'string')
                    ) {
                    validatedChats[id] = firebaseChats[id];
                } else {
                    console.warn(`loadChatsFromFirebase: Pruning invalid or incomplete chat entry loaded from Firebase for ID: ${id}. Chat data:`, firebaseChats[id]);
                    prunedCount++;
                }
            }
            chats = validatedChats;
            console.log(`Successfully loaded chats from Firebase for UID: ${userId}. ${Object.keys(chats).length} conversations found${prunedCount > 0 ? `, ${prunedCount} invalid entries pruned.` : '.'}`);
        } else {
            chats = {}; 
            console.log(`No chats found in Firebase for UID: ${userId}, or data was empty/invalid type.`);
        }
    } catch (error) {
        console.error(`Error loading chats for UID: ${userId} from Firebase:`, error);
        showError("Failed to load chats from your account. Using empty chat list. Details in console.");
        chats = {}; 
    }
}

function setupInitialChatState() {
    loadChatHistory(); 

    const sortedIds = Object.keys(chats).sort((a, b) => {
        const dateA = chats[a] && chats[a].updatedAt ? new Date(chats[a].updatedAt).getTime() : 0;
        const dateB = chats[b] && chats[b].updatedAt ? new Date(chats[b].updatedAt).getTime() : 0;
        return dateB - dateA;
    });

    if (sortedIds.length > 0 && chats[sortedIds[0]]) {
        currentChatId = sortedIds[0];
        renderChat(currentChatId);
    } else {
        if (auth.currentUser) { 
            console.log("No existing chats found for logged-in user. Creating new chat.");
            createNewChat(); 
        } else { 
            renderHomeScreen(); 
        }
    }
    if (messageInput) messageInput.focus();
}

function init() {
  initializeFirebase(); 
  toastNotificationContainer = document.getElementById('toastNotificationContainer');
  if (statusText) statusText.innerHTML = "&nbsp;";

  menuBtn.addEventListener('click', toggleSidebar);
  closeSidebar.addEventListener('click', toggleSidebar);
  sidebarOverlay.addEventListener('click', toggleSidebar);
  newChatBtnHeader.addEventListener('click', createNewChat);
  aboutAppBtn.addEventListener('click', toggleAboutPanel);
  sidebarNewChatBtn.addEventListener('click', createNewChat);
  
  menuSettingsBtn.addEventListener('click', () => {
    openSettingsView();
    userProfileMenu.classList.remove('visible'); 
  });

  if (menuUpgradeBtn) {
    menuUpgradeBtn.addEventListener('click', () => {
        openSubscriptionPage();
        if (userProfileMenu) userProfileMenu.classList.remove('visible');
    });
  }

  closeSettingsViewBtn.addEventListener('click', closeSettingsView);
  userProfileMenuBtn.addEventListener('click', toggleUserProfileMenu);


  messageInput.addEventListener('input', handleInputChange);
  messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (sendBtn.classList.contains('stop-replying')) {
        stopAIResponse();
      } else {
        sendMessage();
      }
    }
  });

  sendBtn.addEventListener('click', sendMessage);
  modelBtn.addEventListener('click', toggleModelSelector);
  
  if (webSearchBtn) {
      webSearchBtn.addEventListener('mousedown', (e) => {
          e.preventDefault(); 
      });
      webSearchBtn.addEventListener('click', () => {
        messageInput.focus(); 
        const currentMessage = messageInput.value.trim();
        if (currentMessage === '') {
            showError("Please type a query to search.");
            messageInput.focus();
            return;
        }
        if (selectedModel === 'openrouter') {
            showError("Web search is not available with the OpenRouter (DeepSeek) model. Please switch to Groq or Auto mode for web search.");
            return;
        }
        messageInput.value = `/search ${currentMessage}`;
        handleInputChange(); 
    });
  }

  if (imageGenBtn) {
      imageGenBtn.addEventListener('mousedown', (e) => {
          e.preventDefault(); 
      });
      imageGenBtn.addEventListener('click', () => {
          messageInput.focus();
          const currentMessage = messageInput.value;
          if (currentMessage.trim() === '' || !currentMessage.toLowerCase().startsWith('/image ')) {
              messageInput.value = `/image ${currentMessage.trim()}`;
          } else if (currentMessage.toLowerCase().startsWith('/image ') && currentMessage.substring(7).trim() === '') {
             // Already has /image but no prompt, do nothing
          }
          handleInputChange();
      });
  }

  attachFileIconBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleFileUpload);
  closeFile.addEventListener('click', clearFile);

  themeSelect.addEventListener('change', () => updateTheme(true));
  fontSizeSelect.addEventListener('change', () => updateFontSize(true));
  modelSelectGlobal.addEventListener('change', updateDefaultModel);

  if (scrollDownBtn && inputContainerElement) { 
    scrollDownBtn.addEventListener('click', () => {
      chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    });
  }

  if (chatMessages && scrollDownBtn && inputContainerElement) {
    chatMessages.addEventListener('scroll', () => {
      const SCROLL_THRESHOLD = 100;
      const isScrolledToBottom = chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight < 1;
      
      const inputHeight = inputContainerElement.offsetHeight || 70; 
      const modelSelectorHeight = modelSelector.classList.contains('visible') ? modelSelector.offsetHeight : 0;
      const bottomOffset = inputHeight + modelSelectorHeight + 10; 

      if (chatMessages.scrollTop + chatMessages.clientHeight < chatMessages.scrollHeight - SCROLL_THRESHOLD && !isScrolledToBottom) {
        scrollDownBtn.classList.add('visible');
        scrollDownBtn.style.bottom = `${bottomOffset}px`;
      } else {
        scrollDownBtn.classList.remove('visible');
      }
    });
    
    if (modelSelector && inputContainerElement) {
        const inputHeight = inputContainerElement.offsetHeight || 70;
        const baseMarginBottom = parseFloat(getComputedStyle(inputContainerElement).marginBottom) || (0.5 * parseFloat(getComputedStyle(document.documentElement).fontSize || 16)); 
        modelSelector.style.bottom = `${inputHeight + baseMarginBottom + 10}px`;
    }
  }


  if (closeFeedbackModalBtn) closeFeedbackModalBtn.addEventListener('click', () => feedbackModal.classList.remove('visible'));
  if (submitFeedbackBtn) submitFeedbackBtn.addEventListener('click', submitFeedback);
  if (feedbackModal) feedbackModal.addEventListener('click', (e) => { if (e.target === feedbackModal) feedbackModal.classList.remove('visible'); });
  
  if (confirmDeleteBtn) confirmDeleteBtn.addEventListener('click', () => {
    if (chatIdToDelete && auth.currentUser && database) { 
        const userId = auth.currentUser.uid;
        database.ref(`users/${userId}/chats/${chatIdToDelete}`).remove()
            .then(() => {
                const chatTitle = (chats[chatIdToDelete] && chats[chatIdToDelete].title !== "New Chat" ? `"${chats[chatIdToDelete].title}"` : "this chat") || "this chat";
                delete chats[chatIdToDelete]; 
                handlePostChatDeletion(chatTitle + " (from your account)");
            })
            .catch(error => {
                console.error(`Error deleting chat ${chatIdToDelete} from Firebase:`, error);
                showError("Failed to delete chat from your account. Details in console.");
                confirmDeleteModal.classList.remove('visible');
                chatIdToDelete = null;
            });
    } else {
        showError("Cannot delete chat: Not logged in, chat ID missing, or database unavailable. Details in console.");
        console.error("Delete chat precondition failed:", {chatIdToDelete, isLoggedIn: !!auth.currentUser, isDbAvailable: !!database});
        confirmDeleteModal.classList.remove('visible');
        chatIdToDelete = null;
    }
  });

  function handlePostChatDeletion(deletedItemDescription) {
    showSuccess(`Deleted ${deletedItemDescription}.`);
    if (Object.keys(chats).length === 0) {
        if (auth.currentUser) createNewChat(); else renderHomeScreen();
    } else if (chatIdToDelete === currentChatId) {
        const sortedIds = Object.keys(chats).sort((a, b) => {
             const dateA = chats[a] && chats[a].updatedAt ? new Date(chats[a].updatedAt).getTime() : 0;
             const dateB = chats[b] && chats[b].updatedAt ? new Date(chats[b].updatedAt).getTime() : 0;
             return dateB - dateA;
        });
        currentChatId = sortedIds.length > 0 ? sortedIds[0] : null;
        if (currentChatId && chats[currentChatId]) { 
            renderChat(currentChatId);
        } else { 
            if (auth.currentUser) createNewChat(); else renderHomeScreen();
        }
    }
    loadChatHistory(); 
    confirmDeleteModal.classList.remove('visible');
    chatIdToDelete = null;
  }

  if (cancelDeleteBtn) cancelDeleteBtn.addEventListener('click', () => { confirmDeleteModal.classList.remove('visible'); chatIdToDelete = null; });
  if (confirmDeleteModal) confirmDeleteModal.addEventListener('click', (e) => { if (e.target === confirmDeleteModal) { confirmDeleteModal.classList.remove('visible'); chatIdToDelete = null; }});

  if (renameChatModal) { 
    const confirmRenameBtn = document.getElementById('confirmRenameBtn');
    const cancelRenameBtn = document.getElementById('cancelRenameBtn');
    const renameChatInput = document.getElementById('renameChatInput');

    confirmRenameBtn.addEventListener('click', () => {
        if (!chatIdToRename || !chats[chatIdToRename]) {
            showError("Error: Chat context lost for renaming.");
            renameChatModal.classList.remove('visible');
            chatIdToRename = null;
            return;
        }
        const currentChat = chats[chatIdToRename];
        const newTitle = renameChatInput.value.trim();

        if (newTitle !== "" && newTitle !== (currentChat.title || "Untitled Chat")) {
            currentChat.title = newTitle;
            // currentChat.updatedAt = new Date().toISOString(); // MODIFIED: Do not update timestamp on rename
            saveChats().then(() => {
                loadChatHistory();
                showSuccess(`Chat renamed to "${newTitle}".`);
                if (currentChatId === chatIdToRename && appTitle.textContent !== 'JARVIS 5.0') {
                    // Potentially update main app title if it shows chat name
                }
            }).catch(err => {
                showError("Failed to save renamed chat. Details in console.");
                console.error("Error saving renamed chat:", err);
            });
        } else if (newTitle === "") {
            showError("Chat name cannot be empty.");
            renameChatInput.focus();
            return; // Keep modal open
        }
        renameChatModal.classList.remove('visible');
        chatIdToRename = null;
    });

    cancelRenameBtn.addEventListener('click', () => {
        renameChatModal.classList.remove('visible');
        chatIdToRename = null;
    });

    renameChatModal.addEventListener('click', (e) => {
        if (e.target === renameChatModal) {
            renameChatModal.classList.remove('visible');
            chatIdToRename = null;
        }
    });
  }


  if (confirmLogoutBtn) confirmLogoutBtn.addEventListener('click', () => {
    signOutGoogle();
    confirmLogoutModal.classList.remove('visible');
  });
  if (cancelLogoutBtn) cancelLogoutBtn.addEventListener('click', () => {
    confirmLogoutModal.classList.remove('visible');
  });
  if (confirmLogoutModal) confirmLogoutModal.addEventListener('click', (e) => {
    if (e.target === confirmLogoutModal) {
      confirmLogoutModal.classList.remove('visible');
    }
  });

  document.addEventListener('click', (e) => {
    document.querySelectorAll('.chat-item-menu.visible').forEach(menu => {
        const menuButton = menu.previousElementSibling; 
        if (menuButton && !menuButton.contains(e.target) && !menu.contains(e.target)) menu.classList.remove('visible');
    });
    if (userProfileMenu && userProfileMenu.classList.contains('visible') && !userProfileMenuBtn.contains(e.target) && !userProfileMenu.contains(e.target)) {
        userProfileMenu.classList.remove('visible');
    }
    if (sidebar && !e.target.closest('.sidebar') && sidebar.classList.contains('open') && !menuBtn.contains(e.target) && window.innerWidth <= 767) {
    }
    if (modelSelector && modelSelector.classList.contains('visible') && !modelBtn.contains(e.target) && !modelSelector.contains(e.target) && !e.target.closest('.model-selector')) modelSelector.classList.remove('visible');
    if (aboutPanel && aboutPanel.classList.contains('visible') && !aboutAppBtn.contains(e.target) && !aboutPanel.contains(e.target) && !e.target.closest('.about-panel')) aboutPanel.classList.remove('visible');
    if (settingsView && settingsView.classList.contains('visible') && !menuSettingsBtn.contains(e.target) && !settingsView.contains(e.target) && !e.target.closest('.settings-view')) { 
    }
  });

  setupModelSelector();
  handleInputChange(); 
}

function openSubscriptionPage() {
    const newWindow = window.open("http://127.0.0.1:5000/plan", "_blank");
    if (!newWindow) {
        showError("Could not open subscription page. Please check your popup blocker.");
    }
    // showToast("The 'Upgrade Plan' feature is currently for demonstration purposes and will be available soon.", "info");
    // console.log("Upgrade Plan button clicked. Placeholder action executed.");
}


function toggleUserProfileMenu() {
    if (userProfileMenu) userProfileMenu.classList.toggle('visible');
}


function updateUserProfileUI(user) {
    if (!sidebarUserProfile || !userProfileImage || !userProfileName || !userProfileMenuBtn) {
        return;
    }
    if (user) {
        userProfileImage.src = user.photoURL || `https://ui-avatars.com/api/?name=${encodeURIComponent(user.displayName || user.email || 'U')}&background=6366f1&color=fff&size=36`; 
        userProfileName.textContent = user.displayName || user.email || 'Authenticated User';
        userProfileName.title = user.displayName || user.email || 'Authenticated User';
        userProfileMenuBtn.style.display = 'flex';
    } else { 
        userProfileImage.src = 'https://via.placeholder.com/36/cccccc/000000?text=U'; 
        userProfileName.textContent = 'Not Signed In';
        userProfileName.title = 'Not Signed In';
        userProfileMenuBtn.style.display = 'none';
        if (userProfileMenu) userProfileMenu.classList.remove('visible');
    }
}

function signInWithGoogle() {
    if (!auth || !googleProvider) {
        showError("Authentication service is not ready. Please try again shortly.");
        return;
    }
    auth.signInWithPopup(googleProvider)
        .then((result) => {
            const user = result.user;
            showSuccess(`Welcome, ${user.displayName || 'User'}! Signed in successfully.`);
        }).catch((error) => {
            console.error("Google Sign-In Error:", error);
            let errorMessage = `Google Sign-In Failed: ${error.message}`;
            if (error.code === 'auth/popup-closed-by-user') errorMessage = "Sign-in cancelled by user.";
            else if (error.code === 'auth/network-request-failed') errorMessage = "Network error during sign-in. Please check your connection.";
            else if (error.code === 'auth/cancelled-popup-request') errorMessage = "Sign-in cancelled. Multiple popups may be open.";
            showError(errorMessage);
        });
}

function signOutGoogle() {
    if (!auth) {
        showError("Authentication service is not ready.");
        return;
    }
    auth.signOut().then(() => {
        showSuccess("Signed out successfully.");
    }).catch((error) => {
        console.error("Google Sign-Out Error:", error);
        showError(`Google Sign-Out Failed: ${error.message}`);
    });
}

function openSettingsView() {
    isSettingsViewOpen = true;
    if (settingsView) settingsView.classList.add('visible');

    if (modelSelector) modelSelector.classList.remove('visible');
    if (aboutPanel) aboutPanel.classList.remove('visible');
    if (sidebar && sidebar.classList.contains('open') && window.innerWidth <= 767) toggleSidebar();
    
    if (themeSelect) themeSelect.value = userSettings.theme;
    if (fontSizeSelect) fontSizeSelect.value = userSettings.fontSize;
    if (modelSelectGlobal) modelSelectGlobal.value = userSettings.model;
}

function closeSettingsView() {
    isSettingsViewOpen = false;
    if (settingsView) settingsView.classList.remove('visible');
}


function clearHomeScreenIfExists() {
  const homeScreen = chatMessages.querySelector('.home-screen-container');
  if (homeScreen) homeScreen.remove();
}

function renderHomeScreen() {
  if (!chatMessages) return; 
  chatMessages.innerHTML = '';
  const homeScreenContainer = document.createElement('div');
  homeScreenContainer.className = 'home-screen-container';
  const shuffledPrompts = shuffleArray([...allSuggestionPrompts]).slice(0, Math.min(4, allSuggestionPrompts.length));
  let promptsHTML = shuffledPrompts.map(p => `
      <div class="suggestion-prompt" data-prompt="${p.prompt.replace(/"/g, '&quot;')}">
        <strong>${p.title}</strong><p>${p.description}</p>
      </div>`).join('');
  homeScreenContainer.innerHTML = `<p class="home-screen-subtitle">Welcome! How can I assist you today?</p><div class="suggestion-prompts-container">${promptsHTML}</div>`;
  chatMessages.appendChild(homeScreenContainer);
  document.querySelectorAll('.suggestion-prompt').forEach(promptEl => {
    promptEl.addEventListener('click', () => {
      if (!auth || !auth.currentUser) { 
          showError("Please sign in to start chatting.");
          if (loginPromptModal) loginPromptModal.classList.add('visible'); 
          return;
      }
      clearHomeScreenIfExists();
      messageInput.value = promptEl.dataset.prompt;
      handleInputChange();
      sendMessage();
    });
  });
  if (scrollDownBtn) scrollDownBtn.classList.remove('visible');
}

function toggleModelSelector() {
  if (!modelSelector) return;
  modelSelector.classList.toggle('visible');
  if (modelSelector.classList.contains('visible')) {
    if (settingsView) settingsView.classList.remove('visible'); 
    if (aboutPanel) aboutPanel.classList.remove('visible');

    if (inputContainerElement) {
        const inputHeight = inputContainerElement.offsetHeight || 70; 
        const baseMarginBottom = parseFloat(getComputedStyle(inputContainerElement).marginBottom) || (0.5 * parseFloat(getComputedStyle(document.documentElement).fontSize || 16));
        const bottomPosition = inputHeight + baseMarginBottom + 10;
        modelSelector.style.bottom = `${bottomPosition}px`;
    }
  }
}

function toggleAboutPanel() {
  if (!aboutPanel) return;
  aboutPanel.classList.toggle('visible');
  if (aboutPanel.classList.contains('visible')) {
    if (settingsView) settingsView.classList.remove('visible');
    if (modelSelector) modelSelector.classList.remove('visible');
  }
}

function updateTheme(showMessage = true) {
  const theme = themeSelect.value;
  userSettings.theme = theme;
  if (auth && auth.currentUser) { 
      saveUserSettingsToFirebase(auth.currentUser.uid, userSettings);
  }
  document.body.classList.remove('light-theme', 'dark-theme');
  if (theme === 'system') {
    document.body.classList.add(window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light-theme' : 'dark-theme');
  } else {
    document.body.classList.add(theme === 'light' ? 'light-theme' : 'dark-theme');
  }
  if (showMessage && auth.currentUser) showSuccess(`Theme set to: ${theme}`); 
}

function updateFontSize(showMessage = true) {
  const fontSize = fontSizeSelect.value;
  userSettings.fontSize = fontSize;
   if (auth && auth.currentUser) { 
      saveUserSettingsToFirebase(auth.currentUser.uid, userSettings);
  }
  let size = fontSize === 'small' ? '14px' : (fontSize === 'large' ? '18px' : '16px');
  document.body.style.fontSize = size;
  if (showMessage && auth.currentUser) showSuccess(`Font size set to: ${fontSize}`); 
}

function updateDefaultModel() {
  const model = modelSelectGlobal.value; 
  userSettings.model = model;
  if (auth && auth.currentUser) { 
      saveUserSettingsToFirebase(auth.currentUser.uid, userSettings);
  }
  selectedModel = model; 
  updateModelSelection(); 
  if (auth.currentUser) showSuccess(`Default model set to: ${model}`); 
}

function setupModelSelector() {
  modelOptions.forEach(option => {
    option.addEventListener('click', () => {
      selectedModel = option.dataset.model; 
      updateModelSelection();
      if (modelSelector) modelSelector.classList.remove('visible');
      if (auth.currentUser) showSuccess(`Model for this session set to: ${option.querySelector('span').textContent}`);
    });
  });
}

function updateModelSelection() {
  modelOptions.forEach(option => {
    option.classList.toggle('active', option.dataset.model === selectedModel);
  });
  const modelIconElement = modelBtn.querySelector('i');
  if (modelIconElement) modelIconElement.className = modelIcons[selectedModel] || 'fas fa-robot';
}

function toggleSidebar() {
  if (sidebar) sidebar.classList.toggle('open');
  if (sidebarOverlay) sidebarOverlay.classList.toggle('visible');
}

function createNewChat() {
  if (!auth || !auth.currentUser) {
    showError("Please sign in to create a new chat.");
    if(loginPromptModal && !loginPromptModal.classList.contains('visible')) {
        loginPromptModal.classList.add('visible');
    }
    return;
  }
  if (settingsView) settingsView.classList.remove('visible'); 
  currentChatId = Date.now().toString();
  chats[currentChatId] = {
    title: "New Chat",
    messages: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  };
  saveChats(); 
  renderChat(currentChatId); 
  loadChatHistory();          
  if(messageInput) messageInput.value = "";    
  handleInputChange();        
  if(messageInput) messageInput.focus();
  if (sidebar && sidebar.classList.contains('open') && window.innerWidth <= 767) toggleSidebar();
}

function getStartOfDay(dateInput) {
    const date = new Date(dateInput);
    date.setHours(0, 0, 0, 0);
    return date;
}

function isToday(date) {
    const today = getStartOfDay(new Date());
    return getStartOfDay(date).getTime() === today.getTime();
}

function isYesterday(date) {
    const yesterday = getStartOfDay(new Date());
    yesterday.setDate(yesterday.getDate() - 1);
    return getStartOfDay(date).getTime() === yesterday.getTime();
}

function isWithinLastNDays(date, nDays) {
    const today = getStartOfDay(new Date());
    const dateToCheck = getStartOfDay(date);
    const nDaysAgo = new Date(today);
    nDaysAgo.setDate(today.getDate() - (nDays - 1)); 
    return dateToCheck.getTime() >= nDaysAgo.getTime() && dateToCheck.getTime() <= today.getTime();
}

function loadChatHistory() {
  if (!chatHistory) return;
  chatHistory.innerHTML = ''; 

  const allChatEntries = Object.entries(chats)
    .filter(([id, chat]) => chat && chat.updatedAt) 
    .sort((a, b) => new Date(b[1].updatedAt).getTime() - new Date(a[1].updatedAt).getTime());

  if (allChatEntries.length === 0) {
    return;
  }

  const todayChats = [];
  const yesterdayChats = [];
  const previous7DaysChats = [];
  const previous30DaysChats = [];
  const olderChats = [];

  allChatEntries.forEach(([id, chat]) => {
    const chatDate = new Date(chat.updatedAt);
    if (isToday(chatDate)) {
      todayChats.push([id, chat]);
    } else if (isYesterday(chatDate)) {
      yesterdayChats.push([id, chat]);
    } else if (isWithinLastNDays(chatDate, 7)) { 
      previous7DaysChats.push([id, chat]);
    } else if (isWithinLastNDays(chatDate, 30)) { 
      previous30DaysChats.push([id, chat]);
    } else {
      olderChats.push([id, chat]);
    }
  });

  const renderGroup = (title, groupChats) => {
    if (groupChats.length > 0) {
      const groupHeader = document.createElement('h3');
      groupHeader.className = 'chat-history-group-header';
      groupHeader.textContent = title;
      chatHistory.appendChild(groupHeader);

      groupChats.forEach(([id, chat]) => {
        if (!chat || typeof chat !== 'object' || !chat.messages || !Array.isArray(chat.messages) || typeof chat.title !== 'string') { 
            console.warn("Skipping malformed chat entry in loadChatHistory for ID:", id, chat);
            return;
        }
        const chatItem = document.createElement('div');
        chatItem.className = `chat-item ${id === currentChatId ? 'active' : ''}`;
        chatItem.dataset.id = id;
        const chatContent = document.createElement('div');
        chatContent.className = 'chat-item-content';
        chatContent.textContent = chat.title || "Untitled Chat"; 
        const actionsContainer = document.createElement('div');
        actionsContainer.className = 'chat-item-actions';
        const menuBtnEl = document.createElement('button');
        menuBtnEl.className = 'chat-item-menu-btn';
        menuBtnEl.innerHTML = '<i class="fas fa-ellipsis-v"></i>';
        menuBtnEl.title = 'More options';
        const menuDropdown = document.createElement('div');
        menuDropdown.className = 'chat-item-menu';
        
        const shareOption = document.createElement('button');
        shareOption.className = 'chat-item-menu-option share-chat-btn';
        shareOption.innerHTML = '<i class="fas fa-share-alt"></i> Share';
        shareOption.dataset.chatId = id;

        const renameOption = document.createElement('button');
        renameOption.className = 'chat-item-menu-option rename-chat-btn';
        renameOption.innerHTML = '<i class="fas fa-pencil-alt"></i> Rename';
        renameOption.dataset.chatId = id;
        
        const deleteOption = document.createElement('button');
        deleteOption.className = 'chat-item-menu-option delete-chat-btn';
        deleteOption.innerHTML = '<i class="fas fa-trash"></i> Delete';
        deleteOption.dataset.chatId = id;

        menuDropdown.appendChild(shareOption);
        menuDropdown.appendChild(renameOption); 
        menuDropdown.appendChild(deleteOption);
        
        actionsContainer.appendChild(menuBtnEl);
        actionsContainer.appendChild(menuDropdown);
        menuBtnEl.addEventListener('click', (e) => { e.stopPropagation(); document.querySelectorAll('.chat-item-menu.visible, .user-profile-menu.visible').forEach(m => { if (m !== menuDropdown) m.classList.remove('visible'); }); menuDropdown.classList.toggle('visible'); });
        shareOption.addEventListener('click', (e) => { e.stopPropagation(); shareChat(e.currentTarget.dataset.chatId); menuDropdown.classList.remove('visible'); });
        renameOption.addEventListener('click', (e) => { e.stopPropagation(); promptRenameChat(e.currentTarget.dataset.chatId); menuDropdown.classList.remove('visible'); }); 
        deleteOption.addEventListener('click', (e) => { e.stopPropagation(); deleteChat(e.currentTarget.dataset.chatId); menuDropdown.classList.remove('visible'); });
        
        chatItem.appendChild(chatContent);
        chatItem.appendChild(actionsContainer);
        chatItem.addEventListener('click', () => { currentChatId = id; renderChat(id); if (settingsView) settingsView.classList.remove('visible'); document.querySelectorAll('.chat-item-menu.visible, .user-profile-menu.visible').forEach(m => m.classList.remove('visible')); if (sidebar && sidebar.classList.contains('open') && window.innerWidth <= 767) toggleSidebar(); });
        chatHistory.appendChild(chatItem);
      });
    }
  };

  renderGroup('Today', todayChats);
  renderGroup('Yesterday', yesterdayChats);
  renderGroup('Previous 7 Days', previous7DaysChats);
  renderGroup('Previous 30 Days', previous30DaysChats);
  renderGroup('Older', olderChats);
}

function promptRenameChat(chatId) {
    const chat = chats[chatId];
    if (!chat) {
        showError("Chat not found for renaming.");
        return;
    }
    chatIdToRename = chatId;

    const renameInput = document.getElementById('renameChatInput');

    if (renameInput) {
        renameInput.value = chat.title || "Untitled Chat";
    }
    if (renameChatModal) {
        renameChatModal.classList.add('visible');
        if (renameInput) {
            renameInput.focus();
            renameInput.select();
        }
    } else {
        console.error("Rename chat modal not found.");
    }
}


function shareChat(chatId) {
  const chat = chats[chatId];
  if (!chat) {
    console.error("Attempted to share non-existent chat:", chatId);
    showError("Cannot share: Chat not found.");
    return;
  }

  const chatTitle = chat.title || "JARVIS 5.0 Conversation";
  let shareText = `Chat: ${chatTitle}\n\n`;

  chat.messages.forEach(msg => {
    const prefix = msg.role === 'user' ? "You: " : "JARVIS: ";
    let contentToShare = msg.content;

    if (msg.role === 'assistant') {
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = marked.parse(contentToShare); 
        contentToShare = tempDiv.textContent || tempDiv.innerText || ""; 
    }
    
    if (msg.image) {
        contentToShare += ` (Shared Image: ${msg.image})`; 
    } else if (msg.tts_file) {
        contentToShare += ` (Audio was generated for this message)`;
    }
    
    shareText += `${prefix}${contentToShare.trim()}\n`;
  });

  shareText += `\nShared from JARVIS 5.0`;

  if (navigator.share) {
    navigator.share({
      title: chatTitle,
      text: shareText,
    })
    .then(() => showSuccess("Chat shared successfully!"))
    .catch((error) => {
        if (error.name !== 'AbortError') {
            console.error('Error sharing via Web Share API:', error);
            copyToClipboardFallback(shareText, chatTitle);
        } else {
            console.log("Sharing aborted by user.");
        }
    });
  } else {
    copyToClipboardFallback(shareText, chatTitle);
  }
}

function copyToClipboardFallback(textToCopy, title) {
    navigator.clipboard.writeText(textToCopy)
      .then(() => showSuccess(`"${title}" content copied to clipboard!`))
      .catch(err => {
        console.error('Error copying to clipboard:', err);
        showError(`Failed to copy chat: ${err.message}. Try manual copy.`);
        const fallbackModalId = 'fallbackShareModal';
        let existingModal = document.getElementById(fallbackModalId);
        if (existingModal) existingModal.remove();

        const fallbackModal = document.createElement('div');
        fallbackModal.id = fallbackModalId;
        fallbackModal.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.8); display: flex; align-items: center;
            justify-content: center; z-index: 4000; padding: 20px;
        `;
        fallbackModal.innerHTML = `
            <div style="background: var(--card); padding: 20px 25px; border-radius: 8px; max-width: 600px; width:90%; max-height: 80vh; display:flex; flex-direction:column; box-shadow: 0 5px 15px rgba(0,0,0,0.3);">
                <h4 style="color: var(--text); margin-top:0; margin-bottom:15px; border-bottom: 1px solid var(--border); padding-bottom:10px;">Copy Chat Content: ${title}</h4>
                <textarea style="width: 100%; flex-grow:1; margin-bottom: 15px; background: var(--bg); color: var(--text); border: 1px solid var(--border); padding: 10px; border-radius: 4px; resize: none;" readonly>${textToCopy}</textarea>
                <button id="closeFallbackShareBtn" style="background:var(--primary); color:white; border:none; padding:10px 15px; border-radius:4px; cursor:pointer;">Close</button>
            </div>
        `;
        document.body.appendChild(fallbackModal);
        fallbackModal.querySelector('#closeFallbackShareBtn').onclick = () => fallbackModal.remove();
        fallbackModal.onclick = (e) => { if (e.target === fallbackModal) fallbackModal.remove(); };
        fallbackModal.querySelector('textarea').select(); 
      });
}


function deleteChat(chatId) {
  chatIdToDelete = chatId;
  const chat = chats[chatIdToDelete];
  let message = `Are you sure you want to delete ${chat && chat.title && chat.title !== "New Chat" ? `the chat titled "${chat.title}"` : "this chat"}? This action cannot be undone.`;
  confirmDeleteMessage.textContent = message;
  confirmDeleteModal.classList.add('visible');
}

function renderChat(chatId) {
  if (!chatMessages) return;
  chatMessages.innerHTML = ''; 
  if (!chatId || !chats[chatId]) {
    console.warn("renderChat: Invalid or non-existent chatId:", chatId, ". Attempting recovery or new chat.");
    const sortedIds = Object.keys(chats).sort((a, b) => new Date(chats[b]?.updatedAt || 0).getTime() - new Date(chats[a]?.updatedAt || 0).getTime());
    if (sortedIds.length > 0 && chats[sortedIds[0]]) {
        currentChatId = sortedIds[0];
        chatId = currentChatId; 
    } else {
        if (auth.currentUser) createNewChat(); 
        else renderHomeScreen(); 
        return; 
    }
  }
  const chat = chats[chatId]; 
  document.querySelectorAll('.chat-item').forEach(item => item.classList.toggle('active', item.dataset.id === chatId));
  if (!chat.messages || chat.messages.length === 0) {
    renderHomeScreen();
    return;
  }
  chat.messages.forEach((msg, index) => {
    let originalUserQueryForThisBotMsg = "";
    let originalFileNameForThisBotMsg = null;
    let originalExtractedTextForThisBotMsg = "";

    if (msg.role === 'assistant' && index > 0 && chat.messages[index-1]?.role === 'user') {
        originalUserQueryForThisBotMsg = chat.messages[index-1].content;
        originalFileNameForThisBotMsg = chat.messages[index-1].fileName || null;
        originalExtractedTextForThisBotMsg = chat.messages[index-1].extractedText || "";
    }
    const options = { 
        isStoppedResponse: msg.content === "AI response stopped by user." && msg.role === 'assistant', 
        originalUserQuery: originalUserQueryForThisBotMsg,
        originalFileName: originalFileNameForThisBotMsg,
        originalExtractedText: originalExtractedTextForThisBotMsg
    };

    const messageDiv = document.createElement('div'); 

    if (msg.role === 'user') {
        populateUserMessageIntoDiv(messageDiv, msg.content, msg.fileName);
    } else if (msg.role === 'assistant') {
        if (msg.image) {
            populateGeneratedImageIntoDiv(messageDiv, msg.content, msg.image, msg.model, options);
        } else if (msg.tts_file) {
            populateTTSMessageIntoDiv(messageDiv, msg.content, msg.tts_file, msg.model, options);
        } else {
            populateBotTextIntoDiv(messageDiv, msg.content, msg.model, options);
        }
    }
    if(chatMessages) chatMessages.appendChild(messageDiv);
  });
  setTimeout(() => { 
      if (chatMessages) { 
          chatMessages.scrollTop = chatMessages.scrollHeight; 
          addCodeBlockActions(); 
          chatMessages.dispatchEvent(new Event('scroll')); 
      }
  }, 100);
}

function addCodeBlockActions() {
  document.querySelectorAll('.message-bot pre, .message-error pre').forEach(pre => {
    if (pre.closest('.katex-display')) return;
    let actionsContainer = pre.querySelector('.code-actions-container');
    if (!actionsContainer) { actionsContainer = document.createElement('div'); actionsContainer.className = 'code-actions-container'; pre.insertBefore(actionsContainer, pre.firstChild || null); }
    else actionsContainer.innerHTML = '';
    const copyBtn = document.createElement('button'); copyBtn.className = 'copy-code-btn'; copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy'; copyBtn.title = 'Copy code'; copyBtn.addEventListener('click', copyCodeBlock); actionsContainer.appendChild(copyBtn);
    const codeElement = pre.querySelector('code');
    let lang = '';
    if (codeElement) lang = (Array.from(codeElement.classList).find(cls => cls.startsWith('language-')) || '').replace('language-', '');
    if (['html', 'javascript', 'js', 'css'].includes(lang.toLowerCase())) {
      const runBtn = document.createElement('button'); runBtn.className = 'run-code-btn'; runBtn.innerHTML = '<i class="fas fa-play"></i> Run'; runBtn.title = `Run ${lang.toUpperCase()} code`;
      runBtn.addEventListener('click', () => runCode(codeElement ? codeElement.textContent : pre.textContent, lang.toLowerCase(), pre));
      actionsContainer.appendChild(runBtn);
    }
  });
}

function runCode(code, lang, preElement) {
  let existingOutput = preElement.parentNode.querySelector(`.code-output-area[data-pre-id="${preElement.id}"]`);
  if (existingOutput) existingOutput.remove();
  if (!preElement.id) preElement.id = `pre-${Date.now()}-${Math.random().toString(36).substring(2,7)}`;
  const outputArea = document.createElement('div'); outputArea.className = 'code-output-area'; outputArea.dataset.preId = preElement.id;
  const cutBtn = document.createElement('button'); cutBtn.className = 'cut-code-output-btn'; cutBtn.innerHTML = '<i class="fas fa-times"></i>'; cutBtn.title = 'Close Output'; cutBtn.addEventListener('click', () => outputArea.remove()); outputArea.appendChild(cutBtn);
  
  const insertOutput = () => preElement.parentNode.insertBefore(outputArea, preElement.nextSibling);

  if (lang === 'html') {
    const iframe = document.createElement('iframe'); iframe.sandbox = 'allow-scripts allow-same-origin'; outputArea.appendChild(iframe); insertOutput();
    setTimeout(() => { try { iframe.srcdoc = code; } catch (e) { iframe.contentDocument.write("Error rendering HTML: " + e.message); iframe.contentDocument.close(); } }, 50);
  } else if (lang === 'javascript' || lang === 'js') {
    const iframe = document.createElement('iframe'); iframe.sandbox = 'allow-scripts allow-same-origin'; iframe.style.display = 'none'; document.body.appendChild(iframe);
    let logs = []; const logContainer = document.createElement('pre'); outputArea.appendChild(logContainer); insertOutput();
    const captureLog = (type) => (...args) => { logs.push({type: type, msg: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)).join(' ')}); renderLogs(); };
    iframe.contentWindow.console = { log: captureLog('log'), error: captureLog('error'), warn: captureLog('warn'), info: captureLog('info') };
    iframe.contentWindow.onerror = (message, source, lineno, colno, error) => { logs.push({type: 'error', msg: `Error: ${message} at ${source||'script'}:${lineno}:${colno}\nStack: ${error?.stack || 'N/A'}`}); renderLogs(); return true; };
    function renderLogs() { logContainer.innerHTML = ''; logs.forEach(log => { const entry = document.createElement('div'); entry.textContent = `[${log.type.toUpperCase()}] ${log.msg}`; if (log.type === 'error') entry.style.color = 'var(--error)'; logContainer.appendChild(entry);}); outputArea.scrollTop = outputArea.scrollHeight;}
    try { const script = iframe.contentDocument.createElement('script'); script.textContent = `try { ${code} } catch (e) { console.error(e.message, e.stack || ''); }`; iframe.contentDocument.body.appendChild(script); } catch (e) { logs.push({type: 'error', msg: 'Execution error: ' + e.message}); renderLogs(); }
    setTimeout(() => iframe.remove(), 3000);
  } else if (lang === 'css') {
    const sampleHTML = `<div class="sample-css-preview"><h3>Styled Preview</h3><p>Content styled by your CSS.</p><button>Button</button></div>`;
    const iframe = document.createElement('iframe'); iframe.sandbox = 'allow-same-origin'; outputArea.appendChild(iframe); insertOutput();
    setTimeout(() => { try { iframe.srcdoc = `<!DOCTYPE html><html><head><style>body{margin:15px;font-family:'Inter',sans-serif;background-color:#fff;color:#333;}.sample-css-preview{border:1px dashed #ccc;padding:15px;background-color:#f9f9f9;border-radius:4px;} ${code}</style></head><body>${sampleHTML}</body></html>`;} catch (e) { iframe.contentDocument.write("Error rendering CSS: " + e.message); iframe.contentDocument.close();}}, 50);
  } else {
    const messageP = document.createElement('p'); messageP.textContent = `Running ${lang.toUpperCase()} client-side not supported.`; outputArea.appendChild(messageP); insertOutput();
  }
  outputArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}


function retryFailedMessage(errorDiv) {
    const originalUserQuery = errorDiv.dataset.originalMessage;
    if (!originalUserQuery) return;

    const originalFileName = errorDiv.dataset.originalFileName || null;
    const originalExtractedText = errorDiv.dataset.originalExtractedText || "";

    errorDiv.innerHTML = `<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div><span class="typing-text"> Retrying...</span>`;
    errorDiv.className = 'message message-bot message-typing'; 

    executeApiCall(originalUserQuery, originalFileName, originalExtractedText, errorDiv);
}

function addBotMessageExtras(messageDiv, model, originalUserQuery, botResponseContent, options = {}) {
    const { isStoppedResponse = false } = options;

    if (model) {
        const modelTag = document.createElement('div');
        modelTag.className = 'message-model';
        let icon = modelIcons[model.split(' ')[0].toLowerCase()] || modelIcons.auto;
        if (model.toLowerCase().includes("system")) icon = 'fas fa-cog';
        else if (model.toLowerCase().includes("web search")) icon = 'fas fa-globe';
        else if (model.toLowerCase().includes("imagegen")) icon = 'fas fa-image'; // For image gen model
        else if (model.toLowerCase().includes("tts")) icon = 'fas fa-volume-up'; // For TTS model
        modelTag.innerHTML = `<i class="${icon}"></i> ${model}`;
        messageDiv.appendChild(modelTag);
    }

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    messageDiv.appendChild(timeDiv);

    if (isStoppedResponse) {
        const continueBtn = document.createElement('button');
        continueBtn.className = 'continue-btn action-btn';
        continueBtn.innerHTML = '<i class="fas fa-play-circle"></i> Continue';
        continueBtn.title = 'Continue generating';
        continueBtn.onclick = () => handleContinueGeneration(originalUserQuery, options.originalFileName, options.originalExtractedText);
        
        let feedbackActionsDiv = messageDiv.querySelector('.message-actions-feedback');
        if (!feedbackActionsDiv) {
            feedbackActionsDiv = document.createElement('div');
            feedbackActionsDiv.className = 'message-actions-feedback';
            (messageDiv.querySelector('.message-model') || messageDiv.querySelector('.message-time') || messageDiv).before(feedbackActionsDiv);
        }
        feedbackActionsDiv.appendChild(continueBtn);
    } else {
        addFeedbackButtons(messageDiv, originalUserQuery, botResponseContent);
    }
}


function populateBotTextIntoDiv(messageDiv, content, model, options = {}) {
    messageDiv.className = 'message message-bot'; 
    messageDiv.innerHTML = ''; 

    const contentWrapper = document.createElement('div');
    // Pre-process content for <highlight> tags
    let processedContent = content.replace(/<highlight>(.*?)<\/highlight>/g, '<span class="important-word">$1</span>');
    contentWrapper.innerHTML = marked.parse(processedContent);
    messageDiv.appendChild(contentWrapper);

    if (typeof renderMathInElement === 'function') {
        renderMathInElement(contentWrapper, { delimiters: [{left:"$$",right:"$$",display:true},{left:"$",right:"$",display:false},{left:"\\(",right:"\\)",display:false},{left:"\\[",right:"\\]",display:true}], throwOnError: false });
    }
    
    addBotMessageExtras(messageDiv, model, options.originalUserQuery, content, options); // Pass original content
    addMessageActions(messageDiv); 
    
    setTimeout(() => addCodeBlockActions(), 50); 
}


function populateTTSMessageIntoDiv(messageDiv, content, ttsFile, model, options = {}) {
    messageDiv.className = 'message message-bot';
    messageDiv.innerHTML = ''; 

    const contentWrapper = document.createElement('div');
    // Pre-process content for <highlight> tags
    let processedContent = content.replace(/<highlight>(.*?)<\/highlight>/g, '<span class="important-word">$1</span>');
    contentWrapper.innerHTML = marked.parse(processedContent);
    messageDiv.appendChild(contentWrapper);

    const controlsDiv = document.createElement('div');
    controlsDiv.className = 'tts-controls';

    const playBtn = document.createElement('button');
    playBtn.className = 'play-tts-btn';
    playBtn.innerHTML = '<i class="fas fa-play"></i> Play';
    playBtn.addEventListener('click', () => playTTS(ttsFile, playBtn));
    controlsDiv.appendChild(playBtn);

    const downloadBtn = document.createElement('a');
    downloadBtn.href = `/download-tts?file=${encodeURIComponent(ttsFile)}`;
    downloadBtn.className = 'tts-download-btn';
    downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download';
    downloadBtn.download = 'tts-audio.mp3';
    controlsDiv.appendChild(downloadBtn);
    messageDiv.appendChild(controlsDiv);

    addBotMessageExtras(messageDiv, model || "TTS", options.originalUserQuery, content, options); // Pass original content
    addMessageActions(messageDiv); 
}


function populateGeneratedImageIntoDiv(messageDiv, description, imageData, model, options = {}) {
    messageDiv.className = 'message message-bot';
    messageDiv.innerHTML = ''; 

    const descDiv = document.createElement('div');
    // Pre-process content for <highlight> tags
    let processedDescription = description.replace(/<highlight>(.*?)<\/highlight>/g, '<span class="important-word">$1</span>');
    descDiv.innerHTML = marked.parse(processedDescription);
    messageDiv.appendChild(descDiv);

    const img = document.createElement('img');
    img.src = imageData;
    img.className = 'generated-image';
    img.alt = description; // Alt text should be original description
    messageDiv.appendChild(img);

    const dlBtn = document.createElement('a');
    dlBtn.href = imageData;
    dlBtn.download = 'jarvis-img.png';
    dlBtn.className = 'download-btn';
    dlBtn.innerHTML = '<i class="fas fa-download"></i> Download';
    messageDiv.appendChild(dlBtn);

    addBotMessageExtras(messageDiv, model || "ImageGen", options.originalUserQuery, description, options); // Pass original content
    addMessageActions(messageDiv); 
}

function populateErrorIntoDiv(messageDiv, errorMessage, model, options = {}) {
    const { originalUserQuery = "", originalFileName = null, originalExtractedText = null } = options;
    messageDiv.className = 'message message-error';
    messageDiv.innerHTML = ''; 

    messageDiv.innerHTML = `<p>${errorMessage}</p>`;
    
    if (originalUserQuery) messageDiv.dataset.originalMessage = originalUserQuery;
    if (originalFileName) messageDiv.dataset.originalFileName = originalFileName;
    if (originalExtractedText) messageDiv.dataset.originalExtractedText = originalExtractedText;

    if (model) {
        const modelTag = document.createElement('div');
        modelTag.className = 'message-model';
        let icon = modelIcons[model.split(' ')[0].toLowerCase()] || 'fas fa-exclamation-triangle';
        modelTag.innerHTML = `<i class="${icon}"></i> ${model}`;
        messageDiv.appendChild(modelTag);
    }

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    messageDiv.appendChild(timeDiv);

    addMessageActions(messageDiv); 
}

function populateUserMessageIntoDiv(messageDiv, content, fileName = null) {
    messageDiv.className = 'message message-user';
    messageDiv.innerHTML = ''; 

    const textContentDiv = document.createElement('div');
    textContentDiv.textContent = content;
    messageDiv.appendChild(textContentDiv);

    if (fileName) {
        const fileIndicator = document.createElement('div');
        fileIndicator.className = 'attached-file-indicator';
        fileIndicator.innerHTML = `<i class="fas fa-paperclip"></i> ${fileName}`;
        messageDiv.appendChild(fileIndicator);
    }

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    messageDiv.appendChild(timeDiv);
}


function playTTS(ttsFile, button) {
  if (currentAudio && currentAudio.src.includes(ttsFile) && !currentAudio.paused) { currentAudio.pause(); button.innerHTML = '<i class="fas fa-play"></i> Play'; button.classList.remove('playing'); return; }
  if (currentAudio) { currentAudio.pause(); document.querySelectorAll('.play-tts-btn.playing').forEach(btn => { if (btn !== button) { btn.innerHTML = '<i class="fas fa-play"></i> Play'; btn.classList.remove('playing'); }}); }
  currentAudio = new Audio(`/download-tts?file=${encodeURIComponent(ttsFile)}`);
  currentAudio.play().then(() => { button.innerHTML = '<i class="fas fa-stop"></i> Stop'; button.classList.add('playing'); }).catch(err => { showError("Play audio failed: " + err.message); button.innerHTML = '<i class="fas fa-play"></i> Play'; button.classList.remove('playing'); currentAudio = null; });
  currentAudio.onended = () => { button.innerHTML = '<i class="fas fa-play"></i> Play'; button.classList.remove('playing'); currentAudio = null; };
  currentAudio.onerror = (e) => { showError("Error playing audio: " + (e.target.error?.message || "Unknown")); button.innerHTML = '<i class="fas fa-play"></i> Play'; button.classList.remove('playing'); currentAudio = null; };
}

function speakText(text) {
  if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(text); utterance.rate = 1.0; utterance.pitch = 1.0; utterance.volume = 1.0;
    window.speechSynthesis.cancel();
    const voices = window.speechSynthesis.getVoices();
    utterance.voice = voices.find(v => (v.name.includes('Male') || v.name.toLowerCase().includes('david') || v.name.toLowerCase().includes('zira')) && (v.lang.startsWith('en') || v.lang.startsWith('hi-IN'))) || voices.find(v => v.lang.startsWith('en'));
    window.speechSynthesis.speak(utterance);
  } else showError("TTS not supported in this browser");
}

function copyCodeBlock(e) {
  const button = e.currentTarget; const pre = button.closest('pre'); if (!pre) return;
  const textToCopy = (pre.querySelector('code') || pre).textContent;
  navigator.clipboard.writeText(textToCopy).then(() => { button.innerHTML = '<i class="fas fa-check"></i> Copied!'; button.classList.add('copied'); setTimeout(() => { button.innerHTML = '<i class="fas fa-copy"></i> Copy'; button.classList.remove('copied'); }, 2000); }).catch(err => showError("Copy failed: " + err.message));
}

async function saveChats() {
    if (auth && auth.currentUser && database) {
        const userId = auth.currentUser.uid;
        console.log(`Attempting to save chats for user ${userId}...`);

        // Only update 'updatedAt' if a new message is being added or if explicitly intended (not during a simple rename)
        if (currentChatId && chats[currentChatId] && currentChatId === currentChatBeingActivelyMessaged) { // currentChatBeingActivelyMessaged needs to be managed
            chats[currentChatId].updatedAt = new Date().toISOString();
        } else if (currentChatId && chats[currentChatId] && !chats[currentChatId].updatedAt) { // If it's somehow missing, set it
             chats[currentChatId].updatedAt = new Date().toISOString();
        } else if (currentChatId) {
            // For other operations like rename, updatedAt is handled by that specific function if needed.
            // If not, it retains its old value.
        }


        try {
            const validChats = {};
            let hasInvalidData = false;
            for (const id in chats) {
                const chat = chats[id];
                if (chat && 
                    typeof chat === 'object' && 
                    chat.messages && 
                    Array.isArray(chat.messages) &&
                    typeof chat.title === 'string' && 
                    typeof chat.createdAt === 'string' && 
                    typeof chat.updatedAt === 'string') {
                    
                    let messagesAreValid = true;
                    for(const msg of chat.messages) {
                        if (!msg || typeof msg.role !== 'string' || (msg.content === undefined && msg.image === undefined && msg.tts_file === undefined) ) { 
                            messagesAreValid = false;
                            console.warn(`saveChats: Pruning chat ID ${id} due to invalid message structure:`, msg);
                            break;
                        }
                        if (msg.content !== undefined && typeof msg.content !== 'string') {
                             messagesAreValid = false;
                             console.warn(`saveChats: Pruning chat ID ${id} due to invalid message content type:`, msg);
                             break;
                        }
                    }
                    if (messagesAreValid) {
                        validChats[id] = chat;
                    } else {
                        hasInvalidData = true;
                    }

                } else {
                     console.warn(`saveChats: Pruning invalid or incomplete chat entry during save for ID: ${id}. Chat data:`, chat);
                     hasInvalidData = true; 
                }
            }
            
            await database.ref(`users/${userId}/chats`).set(validChats);
            console.log("Chats successfully saved to Firebase for user:", userId, (hasInvalidData ? "(some invalid entries were pruned during save)" : ""));
            
        } catch (error) {
            console.error(`Firebase save error for user ${userId}:`, error);
            const dataSample = Object.fromEntries(
                Object.entries(chats)
                      .slice(0, 3) 
                      .map(([k, v]) => [k, { title: v.title, messageCount: v.messages?.length, firstMessageContent: v.messages?.[0]?.content?.substring(0,50) }])
            );
            console.error("Sample of data attempted to save:", dataSample);
            showError("Failed to save chat. Changes might be lost. Details in console.");
        }
    } else {
        let reason = "";
        if (!auth || !auth.currentUser) reason += "User not authenticated. ";
        if (!database) reason += "Database not available.";
        console.warn("saveChats: Skipped. Reason:", reason.trim() || "Unknown (auth/db checks failed)");
    }
}
// Add a new state variable to track when a chat is being actively messaged.
let currentChatBeingActivelyMessaged = null; 


function handleInputChange() {
  if (!messageInput) return; 
  const message = messageInput.value.trim();
  const isImageCommand = messageInput.value.toLowerCase().startsWith('/image ');
  
  if (sendBtn && !sendBtn.classList.contains('stop-replying')) {
      sendBtn.disabled = message === '' || (isImageCommand && message.substring(7).trim() === '');
  }
  if (webSearchBtn) { 
      webSearchBtn.disabled = message === '' || isImageCommand; 
  }
  if (imageGenBtn) { 
      // No specific disable logic for imageGenBtn based on input here.
  }
}

function handleFileUpload() {
  const file = fileInput.files[0]; if (!file) return;
  attachFileIconBtn.classList.add('uploading'); fileInfo.textContent = file.name; activeFileName = file.name; fileInfo.classList.add('has-file'); fileDisplayArea.style.display = 'flex'; 
  if (statusText) statusText.innerHTML = '<i class="fas fa-spinner fa-pulse"></i> Extracting...'; hasActiveFile = true; 
  const formData = new FormData(); formData.append('file', file);
  fetch('/upload', { method: 'POST', body: formData }).then(res => res.json())
  .then(data => { if (data.error) { showError(data.error); clearFile(); } else { extractedText = data.text; showSuccess("File processed."); } attachFileIconBtn.classList.remove('uploading'); if (statusText) statusText.innerHTML = '&nbsp;'; }) 
  .catch(err => { showError("Upload error: " + err.message); clearFile(); attachFileIconBtn.classList.remove('uploading'); if (statusText) statusText.innerHTML = '&nbsp;'; }); 
}

function clearFile() {
  fileInput.value = ""; fileInfo.textContent = "No file selected"; fileInfo.classList.remove('has-file'); fileDisplayArea.style.display = "none"; extractedText = ""; hasActiveFile = false; activeFileName = ""; 
  showSuccess("File removed."); 
  if (statusText) statusText.innerHTML = '&nbsp;'; 
}

function showToast(message, type = 'success') {
    if (!toastNotificationContainer) {
        console.error("Toast container not found!");
        return;
    }

    const toast = document.createElement('div');
    toast.className = `toast-notification ${type}`;
    
    let iconClass = 'fas fa-check-circle'; 
    if (type === 'error') {
        iconClass = 'fas fa-exclamation-circle';
    } else if (type === 'info') { 
        iconClass = 'fas fa-info-circle';
    }

    toast.innerHTML = `<i class="${iconClass}"></i><span>${message}</span>`;
    
    toastNotificationContainer.appendChild(toast);

    toast.offsetHeight; 

    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            if (toast.parentNode === toastNotificationContainer) {
                 toastNotificationContainer.removeChild(toast);
            }
        }, 300); 
    }, 3000); 
}

function showSuccess(message) {
    showToast(message, 'success');
}

function showError(message) {
    showToast(message, 'error');
}

function setUIInteractionState(isInteractive) {
    const UIElementsToToggle = [modelBtn, attachFileIconBtn, fileInput, messageInput, webSearchBtn, imageGenBtn];
    UIElementsToToggle.forEach(el => {
        if (el) el.disabled = !isInteractive;
    });

    if (messageInput) {
        messageInput.placeholder = isInteractive ? "Type your message or /help for commands..." : "AI is responding...";
    }
}

function showGlobalTypingIndicator() {
  if(document.getElementById('typingIndicator')) return; 
  clearHomeScreenIfExists(); 
  isTypingGlobal = true;
  const typingDiv = document.createElement('div'); 
  typingDiv.className = 'message message-bot'; 
  typingDiv.id = 'typingIndicator'; 
  typingDiv.innerHTML = `<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
  if (chatMessages) { chatMessages.appendChild(typingDiv); chatMessages.scrollTop = chatMessages.scrollHeight; }
  
  setUIInteractionState(false); 

  if (sendBtn) {
    sendBtn.innerHTML = '<i class="fas fa-stop"></i>'; 
    sendBtn.title = 'Stop AI Response'; 
    sendBtn.classList.add('stop-replying'); 
    sendBtn.disabled = false; 
    sendBtn.removeEventListener('click', sendMessage); 
    sendBtn.addEventListener('click', stopAIResponse);
  }
}

function hideGlobalTypingIndicator() { 
  isTypingGlobal = false; 
  const typingIndicator = document.getElementById('typingIndicator'); 
  if (typingIndicator) typingIndicator.remove();
}

function showImageGenerationIndicator() {
    if (document.getElementById('imageGenIndicatorGlobal')) return null;
    clearHomeScreenIfExists();
    
    const indicatorDiv = document.createElement('div');
    indicatorDiv.id = 'imageGenIndicatorGlobal';
    indicatorDiv.className = 'message message-bot'; 
    indicatorDiv.style.cssText = `
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        padding: 1.5rem;
        border-radius: 1.125rem 1.125rem 1.125rem 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    `; // Removed margin-bottom as it's a global indicator
    indicatorDiv.innerHTML = `
        <div class="image-gen-visual-container">
            <i class="fas fa-palette image-gen-icon"></i>
            <p class="image-gen-text">Generating your masterpiece...</p>
            <div class="image-gen-progress-bar-container">
                <div class="image-gen-progress-bar-fill"></div>
            </div>
        </div>
    `;
    if (chatMessages) {
        chatMessages.appendChild(indicatorDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    setUIInteractionState(false);
    if (sendBtn) {
        sendBtn.innerHTML = '<i class="fas fa-stop"></i>';
        sendBtn.title = 'Stop AI Response';
        sendBtn.classList.add('stop-replying');
        sendBtn.disabled = false;
        sendBtn.removeEventListener('click', sendMessage);
        sendBtn.addEventListener('click', stopAIResponse);
    }
    return indicatorDiv;
}

function hideImageGenerationIndicator() {
    const indicatorDiv = document.getElementById('imageGenIndicatorGlobal');
    if (indicatorDiv) indicatorDiv.remove();
}


function showTypingInDiv(targetDiv, abortSignalForTyping) {
    targetDiv.innerHTML = ''; 
    targetDiv.className = 'message message-bot message-typing'; 
    
    const typingIndicatorEl = document.createElement('div');
    typingIndicatorEl.className = 'typing-indicator';
    typingIndicatorEl.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    targetDiv.appendChild(typingIndicatorEl);

    const typingTextEl = document.createElement('span');
    typingTextEl.className = 'typing-text';
    targetDiv.appendChild(typingTextEl);

    let dots = '.';
    const intervalId = setInterval(() => {
        if (abortSignalForTyping && abortSignalForTyping.aborted) { 
            clearInterval(intervalId);
            if (targetDiv.contains(typingIndicatorEl)) typingIndicatorEl.remove();
            if (targetDiv.contains(typingTextEl)) typingTextEl.remove();
            return;
        }
        dots = dots.length < 3 ? dots + '.' : '.';
        typingTextEl.textContent = dots;
    }, 300);

    targetDiv.dataset.typingIntervalId = intervalId.toString();

    if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingInDiv(targetDiv) {
    const intervalId = targetDiv.dataset.typingIntervalId;
    if (intervalId) {
        clearInterval(parseInt(intervalId));
        delete targetDiv.dataset.typingIntervalId;
    }
    const typingIndicatorEl = targetDiv.querySelector('.typing-indicator');
    if (typingIndicatorEl) typingIndicatorEl.remove();
    const typingTextEl = targetDiv.querySelector('.typing-text');
    if (typingTextEl) typingTextEl.remove();
    targetDiv.classList.remove('message-typing');
}


function resetSendButtonToNormal() {
    if (sendBtn) {
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>'; 
        sendBtn.title = 'Send Message'; 
        sendBtn.classList.remove('stop-replying');
        sendBtn.removeEventListener('click', stopAIResponse); 
        sendBtn.addEventListener('click', sendMessage); 
    }
}

function stopAIResponse() { 
    if (currentAbortController) {
        currentAbortController.abort(); 
    }
}


function typeMessage(messageDiv, content, model, options, onCompleteCallback) {
    messageDiv.innerHTML = ''; // Clear for fresh typing
    
    const contentHolder = document.createElement('div');
    messageDiv.appendChild(contentHolder);

    let i = 0;
    const speed = 15; // Adjust for typing speed
    let cancelLocalTyping = false;

    if (currentAbortController && currentAbortController.signal) {
        currentAbortController.signal.addEventListener('abort', () => {
            cancelLocalTyping = true;
        }, { once: true });
    }
    
    const placeholder = document.createElement('div');
    placeholder.className = 'typing-text';
    let dots = '.';
    placeholder.textContent = dots;
    contentHolder.appendChild(placeholder); // Placeholder inside contentHolder

    let dotsInterval = setInterval(() => {
        if (cancelLocalTyping) {
            clearInterval(dotsInterval);
            if (placeholder.parentNode) placeholder.remove();
            return;
        }
        dots = dots.length < 3 ? dots + '.' : '.';
        placeholder.textContent = dots;
    }, 300);

    // Pre-process the *entire* content string once before typing begins
    const preProcessedFullContent = content.replace(/<highlight>(.*?)<\/highlight>/g, '<span class="important-word">$1</span>');

    setTimeout(() => { // Delay before actual typing starts
        clearInterval(dotsInterval);
        if (placeholder.parentNode) placeholder.remove();

        if (cancelLocalTyping) {
            if (onCompleteCallback) onCompleteCallback(true, content); // Pass original content for saving
            return;
        }

        function typingEffect() {
            if (cancelLocalTyping) {
                if (onCompleteCallback) onCompleteCallback(true, content); // Pass original content
                return;
            }
            if (i < preProcessedFullContent.length) {
                let chunk = preProcessedFullContent.substring(i, i + Math.floor(Math.random() * 3 + 1));
                contentHolder.innerHTML = marked.parse(preProcessedFullContent.substring(0, i + chunk.length));
                if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
                i += chunk.length;
                setTimeout(typingEffect, speed);
            } else {
                contentHolder.innerHTML = marked.parse(preProcessedFullContent); // Final full parse
                if (typeof renderMathInElement === 'function') {
                     renderMathInElement(contentHolder, { delimiters: [{left:"$$",right:"$$",display:true},{left:"$",right:"$",display:false},{left:"\\(",right:"\\)",display:false},{left:"\\[",right:"\\]",display:true}], throwOnError: false });
                }
                addBotMessageExtras(messageDiv, model, options.originalUserQuery, content, options); // Pass original content
                addMessageActions(messageDiv); 
                setTimeout(() => addCodeBlockActions(), 0);
                
                removeContinueButtons(); 
                if (onCompleteCallback) onCompleteCallback(false, content); // Pass original content
            }
        }
        typingEffect();
    }, 500); 
}


function addMessageActions(messageDiv) {
    let actionsDiv = messageDiv.querySelector('.message-actions');
    if (!actionsDiv) {
        actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        messageDiv.insertBefore(actionsDiv, messageDiv.firstChild || null);
    } else {
        actionsDiv.innerHTML = ''; 
    }

    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-btn';
    copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
    copyBtn.title = 'Copy';
    copyBtn.addEventListener('click', () => copyMessageContent(messageDiv, copyBtn));
    actionsDiv.appendChild(copyBtn);

    if (messageDiv.classList.contains('message-bot')) {
        if (!messageDiv.querySelector('.generated-image') && !messageDiv.querySelector('.tts-controls')) {
            const tempDiv = messageDiv.cloneNode(true);
            tempDiv.querySelectorAll('.message-model, .message-time, .message-actions, .search-results, .open-all-btn, .code-actions-container, .code-output-area, .katex-display, .katex, .message-actions-feedback, .tts-controls, .download-btn').forEach(el => el.remove());
            const textToSpeak = tempDiv.textContent.trim();
            if (textToSpeak) {
                const ttsBtn = document.createElement('button');
                ttsBtn.className = 'tts-btn';
                ttsBtn.innerHTML = '<i class="fas fa-volume-up"></i>';
                ttsBtn.title = 'Read aloud';
                ttsBtn.addEventListener('click', () => speakText(textToSpeak));
                actionsDiv.appendChild(ttsBtn);
            }
        }
    } else if (messageDiv.classList.contains('message-error')) {
        if (messageDiv.dataset.originalMessage && !actionsDiv.querySelector('.retry-btn-small')) {
            const retryBtnSmall = document.createElement('button');
            retryBtnSmall.className = 'retry-btn-small';
            retryBtnSmall.innerHTML = '<i class="fas fa-redo"></i>';
            retryBtnSmall.title = 'Retry';
            retryBtnSmall.addEventListener('click', () => retryFailedMessage(messageDiv));
            actionsDiv.appendChild(retryBtnSmall);
        }
    }
}


function copyMessageContent(messageDiv, button) {
  const tempDiv = messageDiv.cloneNode(true); tempDiv.querySelectorAll('.message-model, .message-time, .message-actions, .search-results, .open-all-btn, .tts-controls, .download-btn, .code-actions-container, .code-output-area, .attached-file-indicator, .katex-display, .katex, .message-actions-feedback').forEach(el => el.remove());
  navigator.clipboard.writeText(tempDiv.textContent.trim()).then(() => { button.innerHTML = '<i class="fas fa-check"></i>'; button.classList.add('copied'); setTimeout(() => { button.innerHTML = '<i class="fas fa-copy"></i>'; button.classList.remove('copied'); }, 2000); }).catch(err => showError("Copy failed: " + err.message));
}

async function sendMessage() {
  if (!auth || !auth.currentUser) { showError("Please sign in to send messages."); if(loginPromptModal && !loginPromptModal.classList.contains('visible')) { loginPromptModal.classList.add('visible'); } return; }
  
  if (!currentChatId) {
      console.error("sendMessage: currentChatId is null or empty. Attempting to create a new chat.");
      showError("No active chat. Starting a new one.");
      createNewChat(); 
      if (!currentChatId) { 
          showError("Critical error: Could not start a new chat. Please refresh. Details in console.");
          return;
      }
  }

  const message = messageInput.value.trim(); if (!message) return;
  if (settingsView && settingsView.classList.contains('visible')) closeSettingsView(); 
  clearHomeScreenIfExists(); removeContinueButtons(); 
  
  const currentActiveFileNameForCall = hasActiveFile ? activeFileName : null;
  const currentExtractedTextForCall = hasActiveFile ? extractedText : "";

  const userMessageDiv = document.createElement('div');
  populateUserMessageIntoDiv(userMessageDiv, message, currentActiveFileNameForCall);
  if (chatMessages) chatMessages.appendChild(userMessageDiv);


  if (!chats[currentChatId]) { 
    console.warn(`sendMessage: Chat object missing for currentChatId "${currentChatId}". This indicates a state inconsistency. Creating a new chat object for this ID.`);
    chats[currentChatId] = { title: message.substring(0,30) + (message.length > 30 ? "..." : ""), messages: [], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
  }

  chats[currentChatId].messages.push({ 
      role: 'user', 
      content: message, 
      fileName: currentActiveFileNameForCall,
      extractedText: currentExtractedTextForCall 
  });
  if (chats[currentChatId].messages.length === 1 && chats[currentChatId].title === "New Chat") { chats[currentChatId].title = message.substring(0, 30) + (message.length > 30 ? "..." : ""); loadChatHistory(); }
  
  currentChatBeingActivelyMessaged = currentChatId; // Set this before calling saveChats
  await saveChats(); 
  currentChatBeingActivelyMessaged = null; // Reset after save
  
  messageInput.value = ""; handleInputChange(); 
  if (chatMessages) {
      chatMessages.scrollTop = chatMessages.scrollHeight;
      chatMessages.dispatchEvent(new Event('scroll'));
  }

  if (message.toLowerCase().startsWith('/image ')) {
    const prompt = message.substring(7).trim();
    if (!prompt) { 
        const errorOptions = { 
            originalUserQuery: message,
            originalFileName: currentActiveFileNameForCall,
            originalExtractedText: currentExtractedTextForCall
        };
        const errDiv = document.createElement('div');
        populateErrorIntoDiv(errDiv, "Provide image description (e.g., /image sunset)", "Input Error", errorOptions);
        if (chatMessages) chatMessages.appendChild(errDiv);
        addMessageActions(errDiv); 
        return; 
    }
    const imageBotResponseDiv = document.createElement('div');
    if (chatMessages) chatMessages.appendChild(imageBotResponseDiv);
    generateImage(prompt, imageBotResponseDiv); 
    return;
  }

  executeApiCall(message, currentActiveFileNameForCall, currentExtractedTextForCall);
}

async function executeApiCall(originalUserQueryContext, associatedFileName = null, associatedExtractedText = "", targetDivToReplace = null) {
    lastUserSentMessage = originalUserQueryContext; 
    currentAbortController = new AbortController();
    const modelForThisRequest = selectedModel;

    if (targetDivToReplace) {
        showTypingInDiv(targetDivToReplace, currentAbortController.signal);
    } else {
        showGlobalTypingIndicator(); 
    }

    try {
        const res = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                messages: chats[currentChatId].messages, 
                extracted_text: associatedExtractedText,
                model_preference: modelForThisRequest
            }),
            signal: currentAbortController.signal
        });

        const data = await res.json();
        const messageDisplayOptions = { 
            originalUserQuery: originalUserQueryContext,
            originalFileName: associatedFileName,
            originalExtractedText: associatedExtractedText
        };
        
        if (targetDivToReplace) {
             hideTypingInDiv(targetDivToReplace);
        } else {
             hideGlobalTypingIndicator(); 
        }

        const divToUse = targetDivToReplace || document.createElement('div');
        if (!targetDivToReplace) {
            divToUse.className = 'message message-bot'; // Set class for new bot message div
            if (chatMessages) chatMessages.appendChild(divToUse);
        } else {
            divToUse.className = 'message message-bot'; // Ensure retried error becomes a bot message
        }
        
        if (!res.ok || data.error) {
            populateErrorIntoDiv(divToUse, data.error || `Server error: ${res.status}`, data.model || "Error", messageDisplayOptions);
            chats[currentChatId].messages.push({ role: 'assistant', content: data.error || `Server error: ${res.status}`, model: data.model || "Error" });
            currentChatBeingActivelyMessaged = currentChatId; await saveChats(); currentChatBeingActivelyMessaged = null;
            return; 
        }
        
        let assistantResponseContent = data.response;
        let assistantResponse = { role: 'assistant', content: assistantResponseContent, model: data.model };

        if (data.action === 'search_results') { 
            currentSearchResults = data.results; 
            typeMessage(divToUse, assistantResponseContent, data.model, messageDisplayOptions, (cancelled, finalContent) => { 
                assistantResponse.content = finalContent; // Update with potentially stopped content
                if (!cancelled) displaySearchResults(divToUse, data.results); 
                chats[currentChatId].messages.push(assistantResponse);
                currentChatBeingActivelyMessaged = currentChatId; saveChats().then(() => currentChatBeingActivelyMessaged = null);
            });
        } else if (data.action === 'redirect') { 
            window.open(data.url, "_blank"); 
            typeMessage(divToUse, assistantResponseContent, data.model, messageDisplayOptions, (cancelled, finalContent) => {
                assistantResponse.content = finalContent;
                chats[currentChatId].messages.push(assistantResponse);
                currentChatBeingActivelyMessaged = currentChatId; saveChats().then(() => currentChatBeingActivelyMessaged = null);
            });
        } else if (data.action === 'tts') { 
            assistantResponse.tts_file = data.tts_file; 
            populateTTSMessageIntoDiv(divToUse, assistantResponseContent, data.tts_file, data.model, messageDisplayOptions); 
            chats[currentChatId].messages.push(assistantResponse);
            currentChatBeingActivelyMessaged = currentChatId; await saveChats(); currentChatBeingActivelyMessaged = null;
        } else if (data.action === 'image') { 
            // This case should be handled by generateImage function directly
            assistantResponse.image = data.image_url; 
            populateGeneratedImageIntoDiv(divToUse, assistantResponseContent, data.image_url, data.model, messageDisplayOptions); 
            chats[currentChatId].messages.push(assistantResponse);
            currentChatBeingActivelyMessaged = currentChatId; await saveChats(); currentChatBeingActivelyMessaged = null;
        } else { // Standard text response
            typeMessage(divToUse, assistantResponseContent, data.model, messageDisplayOptions, (cancelled, finalContent) => { 
                assistantResponse.content = finalContent;
                 chats[currentChatId].messages.push(assistantResponse);
                 currentChatBeingActivelyMessaged = currentChatId; saveChats().then(() => currentChatBeingActivelyMessaged = null);
            });
        }

    } catch (err) {
        const modelNameForStop = modelDisplayNames[modelForThisRequest] || modelForThisRequest;
        const stopMessageContent = "AI response stopped by user.";
        const stopOptions = { 
            isStoppedResponse: true, 
            originalUserQuery: lastUserSentMessage,
            originalFileName: associatedFileName,
            originalExtractedText: associatedExtractedText
        };

        if (err.name === 'AbortError') { 
            console.log('Fetch aborted.'); 
            const divToUpdate = targetDivToReplace || document.getElementById('typingIndicator') || document.getElementById('imageGenIndicatorGlobal');
            
            if (divToUpdate && (divToUpdate.id === 'typingIndicator' || divToUpdate.id === 'imageGenIndicatorGlobal')) {
                hideGlobalTypingIndicator(); 
                hideImageGenerationIndicator();
                const newStopDiv = document.createElement('div');
                newStopDiv.className = 'message message-bot';
                if(chatMessages) chatMessages.appendChild(newStopDiv);
                populateBotTextIntoDiv(newStopDiv, stopMessageContent, modelNameForStop, stopOptions);
            } else if (divToUpdate) { // Retrying an error div
                hideTypingInDiv(divToUpdate);
                divToUpdate.className = 'message message-bot';
                populateBotTextIntoDiv(divToUpdate, stopMessageContent, modelNameForStop, stopOptions);
            }
            
            chats[currentChatId].messages.push({ role: 'assistant', content: stopMessageContent, model: modelForThisRequest }); 
            currentChatBeingActivelyMessaged = currentChatId; await saveChats(); currentChatBeingActivelyMessaged = null;
        } else { 
            if (targetDivToReplace) {
                 hideTypingInDiv(targetDivToReplace);
            } else {
                 hideGlobalTypingIndicator(); 
                 hideImageGenerationIndicator();
            }
            const networkErrorOptions = { originalUserQuery: lastUserSentMessage, originalFileName: associatedFileName, originalExtractedText: associatedExtractedText };
            const divToUse = targetDivToReplace || document.createElement('div');
            if (!targetDivToReplace && chatMessages) chatMessages.appendChild(divToUse);
            populateErrorIntoDiv(divToUse, "⚠️ Network error: " + err.message, "Network Error", networkErrorOptions);

            chats[currentChatId].messages.push({ role: 'assistant', content: "⚠️ Network error: " + err.message, model: "Network Error" });
            currentChatBeingActivelyMessaged = currentChatId; await saveChats(); currentChatBeingActivelyMessaged = null;
        }
    } finally { 
        currentAbortController = null; 
        setUIInteractionState(true); 
        resetSendButtonToNormal();   
        handleInputChange();         
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
            chatMessages.dispatchEvent(new Event('scroll'));
        }
    }
}


async function generateImage(prompt, targetDivToPopulate) {
  if (!auth || !auth.currentUser) { showError("Please sign in to generate images."); if(loginPromptModal && !loginPromptModal.classList.contains('visible')) { loginPromptModal.classList.add('visible'); } return; }
  if (settingsView && settingsView.classList.contains('visible')) closeSettingsView();
  removeContinueButtons(); 
  
  const userMessageForImage = `/image ${prompt}`;
  lastUserSentMessage = userMessageForImage; 

  currentAbortController = new AbortController();
  const modelForThisRequest = "ImageGen"; 
  
  const imageGenIndicatorElement = showImageGenerationIndicator(); 

  try {
    const res = await fetch("/generate-image", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt }), signal: currentAbortController.signal });
    const data = await res.json(); 
    
    if (imageGenIndicatorElement) hideImageGenerationIndicator();

    const imageOptions = { originalUserQuery: userMessageForImage }; 
    targetDivToPopulate.className = 'message message-bot'; // Ensure the container is styled as a bot message

    if (!res.ok || data.error) { 
        populateErrorIntoDiv(targetDivToPopulate, data.error || `Server error: ${res.status}`, data.model || "Image Gen Error", imageOptions);
        chats[currentChatId].messages.push({ role: 'assistant', content: data.error || `Server error: ${res.status}`, model: data.model || "Image Gen Error" });
        currentChatBeingActivelyMessaged = currentChatId; await saveChats(); currentChatBeingActivelyMessaged = null;
        return; 
    }
    
    populateGeneratedImageIntoDiv(targetDivToPopulate, `Image for: "${prompt}"`, data.image_url, data.model || modelForThisRequest, imageOptions);
    
    if (!chats[currentChatId]) { 
        console.warn("generateImage: currentChatId not found in chats. Creating new entry.");
        chats[currentChatId] = { title: "Image Generation", messages: [], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() };
    }
    chats[currentChatId].messages.push({ role: 'assistant', content: `Image for: "${prompt}"`, image: data.image_url, model: data.model || modelForThisRequest });
    currentChatBeingActivelyMessaged = currentChatId; await saveChats(); currentChatBeingActivelyMessaged = null;

  } catch (err) {
    if (imageGenIndicatorElement) hideImageGenerationIndicator();

    const stopMessageContent = "Image generation stopped.";
    targetDivToPopulate.className = 'message message-bot'; // Ensure styling if error occurs

    if (err.name === 'AbortError') { 
        console.log('Image gen aborted.'); 
        populateBotTextIntoDiv(targetDivToPopulate, stopMessageContent, modelForThisRequest, { isStoppedResponse: true, originalUserQuery: lastUserSentMessage });
        chats[currentChatId].messages.push({ role: 'assistant', content: stopMessageContent, model: modelForThisRequest }); 
        currentChatBeingActivelyMessaged = currentChatId; await saveChats(); currentChatBeingActivelyMessaged = null;
    } else { 
        const networkErrorOptions = { originalUserQuery: lastUserSentMessage };
        populateErrorIntoDiv(targetDivToPopulate, "⚠️ Image gen failed: " + err.message, "Network Error", networkErrorOptions);
        chats[currentChatId].messages.push({ role: 'assistant', content: "⚠️ Image gen failed: " + err.message, model: "Network Error" });
        currentChatBeingActivelyMessaged = currentChatId; await saveChats(); currentChatBeingActivelyMessaged = null;
    }
  } finally { 
      currentAbortController = null; 
      setUIInteractionState(true);   
      resetSendButtonToNormal();     
      handleInputChange();           
      if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
        chatMessages.dispatchEvent(new Event('scroll'));
      }
  }
}

function addMessageToUI(content, type, modelOrFileName = null, options = {}) {
  const { isStoppedResponse = false, originalUserQuery = "", originalFileName = null, originalExtractedText = null } = options;
  if (settingsView && settingsView.classList.contains('visible')) closeSettingsView();
  clearHomeScreenIfExists(); 
  
  const messageDiv = document.createElement('div'); 

  if (type === 'user') {
    populateUserMessageIntoDiv(messageDiv, content, modelOrFileName);
  } else if (type === 'bot') { 
    populateBotTextIntoDiv(messageDiv, content, modelOrFileName, options);
  } else if (type === 'error') {
    populateErrorIntoDiv(messageDiv, content, modelOrFileName, options);
  }
  
  if (chatMessages) {
    chatMessages.appendChild(messageDiv); 
    chatMessages.scrollTop = chatMessages.scrollHeight;
    chatMessages.dispatchEvent(new Event('scroll'));
  }
  return messageDiv; 
}


function addFeedbackButtons(messageDiv, userPrompt, botResponseText) {
    let feedbackActionsDiv = messageDiv.querySelector('.message-actions-feedback');
    if (!feedbackActionsDiv) { 
        feedbackActionsDiv = document.createElement('div'); 
        feedbackActionsDiv.className = 'message-actions-feedback'; 
        const insertBeforeEl = messageDiv.querySelector('.message-model') || messageDiv.querySelector('.message-time') || null;
        messageDiv.insertBefore(feedbackActionsDiv, insertBeforeEl);
    }
    feedbackActionsDiv.innerHTML = ''; 

    if (userPrompt && userPrompt.trim() !== "") { 
        const retryBtn = document.createElement('button'); 
        retryBtn.className = 'action-btn retry-feedback-btn'; 
        retryBtn.innerHTML = '<i class="fas fa-redo"></i> Retry'; 
        retryBtn.title = 'Retry prompt'; 
        retryBtn.addEventListener('click', () => {
            const userMsgEntry = chats[currentChatId]?.messages.findLast(m => m.role === 'user' && m.content === userPrompt);
            const fileName = userMsgEntry?.fileName || null;
            const extractedTextContent = userMsgEntry?.extractedText || ""; 
            
            executeApiCall(userPrompt, fileName, extractedTextContent, null); 
        }); 
        feedbackActionsDiv.appendChild(retryBtn); 
    }

    const likeBtn = document.createElement('button'); 
    likeBtn.className = 'action-btn like-feedback-btn'; 
    likeBtn.innerHTML = '<i class="fas fa-thumbs-up"></i> Like'; 
    likeBtn.title = 'Like'; 
    likeBtn.addEventListener('click', (e) => { 
        e.currentTarget.classList.toggle('liked'); 
        console.log("Liked:", { userPrompt, botResponseText }); 
        const dislikeBtn = e.currentTarget.parentNode.querySelector('.dislike-feedback-btn'); 
        if(e.currentTarget.classList.contains('liked') && dislikeBtn) dislikeBtn.classList.remove('disliked'); 
    }); 
    feedbackActionsDiv.appendChild(likeBtn);

    const dislikeBtn = document.createElement('button'); 
    dislikeBtn.className = 'action-btn dislike-feedback-btn'; 
    dislikeBtn.innerHTML = '<i class="fas fa-thumbs-down"></i> Dislike'; 
    dislikeBtn.title = 'Dislike / Report'; 
    dislikeBtn.addEventListener('click', () => showFeedbackModal(userPrompt, botResponseText)); 
    feedbackActionsDiv.appendChild(dislikeBtn);
}


function displaySearchResults(messageDiv, results) {
  const resultsContainer = document.createElement('div'); resultsContainer.className = 'search-results';
  results.forEach(result => { const link = document.createElement('a'); link.href = result; link.className = 'search-result'; link.target = '_blank'; link.rel = 'noopener noreferrer'; link.innerHTML = `<i class="fas fa-external-link-alt"></i> <span>${result}</span>`; resultsContainer.appendChild(link); });
  if (results.length > 0) { const openAllBtn = document.createElement('button'); openAllBtn.className = 'open-all-btn'; openAllBtn.innerHTML = '<i class="fas fa-external-link-square-alt"></i> Open All'; openAllBtn.onclick = () => results.forEach(r => window.open(r, '_blank')); resultsContainer.appendChild(openAllBtn); }
  
  const insertBeforeEl = messageDiv.querySelector('.message-model') || messageDiv.querySelector('.message-time') || messageDiv.querySelector('.message-actions-feedback') || null;
  messageDiv.insertBefore(resultsContainer, insertBeforeEl);

  if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeContinueButtons() { document.querySelectorAll('.continue-btn').forEach(btn => btn.remove()); }

function handleContinueGeneration(originalUserMessage, originalFileName = null, originalExtractedText = "") { 
    if (originalUserMessage) { 
        removeContinueButtons(); 
        executeApiCall(originalUserMessage, originalFileName, originalExtractedText, null); 
    } 
}
function showFeedbackModal(userPrompt, botResponseText) { currentFeedbackData = { userPrompt, botResponseText }; feedbackTextEl.value = ''; feedbackModal.classList.add('visible'); feedbackTextEl.focus(); }

async function submitFeedback() {
    const reportDescription = feedbackTextEl.value.trim(); if (!reportDescription) { showError("Please describe the issue."); return; }
    submitFeedbackBtn.disabled = true; submitFeedbackBtn.innerHTML = '<i class="fas fa-spinner fa-pulse"></i> Submitting...';
    let reporterEmail = "anonymous@jarvis.app", reporterName = "Anonymous";
    if (auth && auth.currentUser) { reporterEmail = auth.currentUser.email || reporterEmail; reporterName = auth.currentUser.displayName || reporterName; }
    try {
        const response = await fetch('/report-issue', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ userPrompt: currentFeedbackData.userPrompt, botResponse: currentFeedbackData.botResponseText, description: reportDescription, chatId: currentChatId, timestamp: new Date().toISOString(), reporterEmail, reporterName }) });
        if (response.ok) { 
            showSuccess("Feedback submitted. Thank you!"); 
            feedbackModal.classList.remove('visible'); 
        } else { 
            const errData = await response.json().catch(()=>({ message: "Unknown server error. Please try again." })); 
            showError(`Feedback submission failed: ${errData.error || errData.message || response.statusText}`);
        }
    } catch (error) { 
        showError(`Feedback error: ${error.message}`); 
    }
    finally { 
        submitFeedbackBtn.disabled = false; 
        submitFeedbackBtn.textContent = 'Submit Report'; 
    }
}

document.addEventListener('DOMContentLoaded', init);
</script>

</body>
</html>

"""
PLAN_HTML =  """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JARVIS 5.0 - Subscription Plans</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg: #0f172a;
            --card: #1e293b;
            --border: #334155;
            --primary: #6366f1;
            --primary-rgb: 99, 102, 241; /* Added for rgba() usage */
            --primary-hover: #4f46e5;
            --secondary: #8b5cf6;
            --success: #10b981;
            --danger: #f43f5e; 
            --text: #f8fafc;
            --subtext: #94a3b8;
            --highlight: rgba(99, 102, 241, 0.1);
        }

        *,
        *::before,
        *::after {
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            line-height: 1.6;
        }

        .plans-header {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 2rem 1rem;
            text-align: center;
            width: 100%;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .plans-header h1 {
            margin: 0 0 0.5rem 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .plans-header h1 .fas { /* Minor spacing for icon in header */
            margin-right: 0.5rem;
        }

        .plans-header p {
            margin: 0;
            font-size: 1.1rem;
            color: rgba(255,255,255,0.85);
        }

        .plans-container {
            display: flex;
            justify-content: center;
            align-items: stretch; /* Makes cards same height if in a row */
            flex-wrap: wrap;
            gap: 2rem;
            padding: 3rem 1rem;
            max-width: 1200px;
            width: 100%;
        }

        .plan-card {
            background-color: var(--card);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 2rem;
            width: 100%;
            max-width: 350px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            display: flex;
            flex-direction: column;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .plan-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 35px rgba(var(--primary-rgb), 0.2);
        }

        .plan-card.popular {
            border-top: 5px solid var(--success);
            position: relative;
        }
        .plan-card.popular::before {
            content: "Most Popular";
            position: absolute;
            top: -1px; 
            left: 50%;
            transform: translateX(-50%) translateY(-100%);
            background-color: var(--success);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 0.3rem 0.3rem 0 0;
            font-size: 0.8rem;
            font-weight: 600;
            white-space: nowrap; /* Ensure "Most Popular" stays on one line */
        }


        .plan-card h2 {
            font-size: 1.75rem;
            color: var(--text);
            margin-top: 0; /* Reset margin if any from browser defaults */
            margin-bottom: 0.5rem;
            text-align: center;
        }

        .plan-price {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 0.25rem;
            text-align: center;
        }

        .plan-price span {
            font-size: 1rem;
            font-weight: 400;
            color: var(--subtext);
        }

        .plan-description {
            font-size: 0.9rem;
            color: var(--subtext);
            margin-bottom: 1.5rem;
            text-align: center;
            min-height: 40px; /* Ensure space even if description is short */
        }

        .plan-features {
            list-style: none;
            padding: 0;
            margin: 0 0 1.5rem 0;
            flex-grow: 1; /* Pushes button to bottom */
        }

        .plan-features li {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
            font-size: 0.95rem;
        }

        .plan-features li i {
            font-size: 1rem;
            width: 20px; /* Fixed width for icon container */
            text-align: center;
            flex-shrink: 0; /* Prevent icon from shrinking */
        }
        .plan-features li i.fa-check-circle {
            color: var(--success);
        }
        .plan-features li i.fa-times-circle {
            color: var(--danger); 
        }
         .plan-features li .feature-note {
            font-size: 0.8em;
            color: var(--subtext);
            margin-left: auto; 
            padding-left: 0.5em;
            white-space: nowrap; /* Keep notes on one line if possible */
        }


        .plan-button {
            display: block;
            width: 100%;
            background: linear-gradient(45deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            padding: 0.9rem 1.5rem;
            border-radius: 0.5rem;
            font-size: 1rem;
            font-weight: 600;
            text-align: center;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 10px rgba(var(--primary-rgb), 0.25);
            margin-top: auto; /* Pushes button to bottom */
        }

        .plan-button:hover {
            background: linear-gradient(45deg, var(--primary-hover), color-mix(in srgb, var(--secondary) 90%, black 10%));
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(var(--primary-rgb), 0.35);
        }
        
        .plan-button.disabled {
            background: var(--border);
            cursor: not-allowed;
            box-shadow: none;
        }
        .plan-button.disabled:hover {
            background: var(--border);
            transform: none;
            box-shadow: none;
        }


        .footer-note {
            text-align: center;
            margin-top: 2rem;
            margin-bottom: 2rem;
            padding: 0 1rem; /* Add padding for smaller screens */
            color: var(--subtext);
            font-size: 0.9rem;
        }
        .footer-note a {
            color: var(--primary);
            text-decoration: none;
        }
        .footer-note a:hover {
            text-decoration: underline;
        }

        @media (max-width: 992px) { 
            .plan-card {
                /* Aim for 3 cards in a row if space allows, flex-basis is more robust */
                flex-grow: 1;
                flex-basis: calc(33.333% - 2rem); /* (100% / 3) - gap */
                max-width: calc(33.333% - 2rem); /* Ensure it doesn't exceed this */
            }
             .plans-container {
                justify-content: space-around; /* Still useful for alignment */
            }
        }


        @media (max-width: 768px) {
            .plan-card {
                 /* Aim for 2 cards in a row */
                flex-basis: calc(50% - 1rem); /* (100% / 2) - (gap / 2 because gap is between items) */
                max-width: calc(50% - 1rem);
                /* If you want a fixed max-width too, you can keep it, e.g. max-width: 350px; */
            }
        }
        @media (max-width: 620px) { /* Adjusted for single card stack sooner if 2 cards + gap don't fit well */
            .plans-header h1 {
                font-size: 2rem;
            }
            .plans-header p {
                font-size: 1rem;
            }
            .plan-card {
                padding: 1.5rem;
                flex-basis: 100%; /* Full width on small screens */
                max-width: 100%; 
            }
            .plan-price {
                font-size: 2rem;
            }
        }

    </style>
</head>
<body>

    <header class="plans-header">
        <h1><i class="fas fa-gem"></i>Upgrade Your JARVIS Experience</h1>
        <p>Choose a plan that best fits your needs and unlock premium features.</p>
    </header>

    <main class="plans-container">
        <div class="plan-card">
            <h2>Free Plan</h2>
            <div class="plan-price">₹0<span>/month</span></div>
            <p class="plan-description">Essential features to get you started at no cost.</p>
            <ul class="plan-features">
                <li><i class="fas fa-check-circle"></i> 100 Text Requests / Day</li>
                <li><i class="fas fa-check-circle"></i> 3 Image Generations / Day</li>
                <li><i class="fas fa-check-circle"></i> TTS Support</li>
                <li><i class="fas fa-times-circle"></i> Voice Assistant Support</li>
                <li><i class="fas fa-times-circle"></i> Ad-Free Experience</li>
                <li><i class="fas fa-times-circle"></i> Priority Support</li>
            </ul>
            <a href="#" class="plan-button">Start with Free</a>
        </div>

        <div class="plan-card">
            <h2>Basic Plan</h2>
            <div class="plan-price">₹59<span>/month</span></div>
            <p class="plan-description">More power and higher limits for regular users.</p>
            <ul class="plan-features">
                <li><i class="fas fa-check-circle"></i> 500 Text Requests / Day</li>
                <li><i class="fas fa-check-circle"></i> 50 Image Generations / Day <span class="feature-note">(2/min)</span></li>
                <li><i class="fas fa-check-circle"></i> TTS Support</li>
                <li><i class="fas fa-times-circle"></i> Voice Assistant Support</li>
                <li><i class="fas fa-check-circle"></i> Ad-Free Experience</li>
                <li><i class="fas fa-check-circle"></i> Priority Email Support</li>
            </ul>
            <a href="#" class="plan-button">Upgrade to Basic</a>
        </div>

        <div class="plan-card popular"> <!-- "Most Popular" tag on this plan -->
            <h2>Pro Plan</h2>
            <div class="plan-price">₹99<span>/month</span></div>
            <p class="plan-description">The ultimate JARVIS experience for professionals.</p>
            <ul class="plan-features">
                <li><i class="fas fa-check-circle"></i> 750 Text Requests / Day</li>
                <li><i class="fas fa-check-circle"></i> 150 Image Generations / Day <span class="feature-note">(3/min)</span></li>
                <li><i class="fas fa-check-circle"></i> TTS Support</li>
                <li><i class="fas fa-check-circle"></i> Voice Assistant Support</li>
                <li><i class="fas fa-check-circle"></i> All AI Models Access</li>
                <li><i class="fas fa-check-circle"></i> Unlimited Chat History</li>
                <li><i class="fas fa-check-circle"></i> Larger File Uploads & OCR</li>
                <li><i class="fas fa-star"></i> Early Access to New Features</li> <!-- Note: fa-star is used here, ensure it's intended or change to fa-check-circle if it's a standard inclusion -->
            </ul>
            <a href="#" class="plan-button">Upgrade to Pro</a>
        </div>
    </main>

<footer class="footer-note">
    <p>Prices are illustrative. For actual services and pricing, please refer to official documentation or contact support.
    <br>© 2024 JARVIS 5.0. All rights reserved. <a href="/term-of-policy">Terms of Service</a> | <a href="/privacy-policy">Privacy Policy</a></p>
</footer>
</body>
</html>
"""
PAYMENT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Complete Your Purchase - JARVIS 5.0</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg: #0f172a;
            --card: #1e293b;
            --border: #334155;
            --primary: #6366f1;
            --primary-rgb: 99, 102, 241;
            --primary-hover: #4f46e5;
            --secondary: #8b5cf6;
            --success: #10b981;
            --danger: #f43f5e;
            --text: #f8fafc;
            --subtext: #94a3b8;
        }

        *, *::before, *::after { box-sizing: border-box; }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            line-height: 1.6;
        }

        .payment-header {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 2rem 1rem;
            text-align: center;
            width: 100%;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }

        .payment-header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
        }

        .payment-container {
            background-color: var(--card);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 2rem;
            width: 100%;
            max-width: 500px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }

        .order-summary {
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }

        .order-summary h2 {
            font-size: 1.5rem;
            color: var(--text);
            margin-top: 0;
            margin-bottom: 1rem;
            text-align: center;
        }

        .summary-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }
        .summary-item .label { color: var(--subtext); }
        .summary-item .value { font-weight: 600; }

        .payment-form h2 {
            font-size: 1.5rem;
            color: var(--text);
            margin-top: 0;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .form-group {
            margin-bottom: 1.25rem;
        }

        .form-group label {
            display: block;
            color: var(--subtext);
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }

        .form-group input {
            width: 100%;
            padding: 0.75rem 1rem;
            background-color: #2c3a4f; /* Slightly lighter than card for input bg */
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            color: var(--text);
            font-size: 1rem;
            transition: border-color 0.2s ease;
        }
        .form-group input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(var(--primary-rgb), 0.2);
        }

        .form-row {
            display: flex;
            gap: 1rem;
        }
        .form-row .form-group {
            flex: 1;
        }

        .pay-button {
            display: block;
            width: 100%;
            background: linear-gradient(45deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            padding: 0.9rem 1.5rem;
            border-radius: 0.5rem;
            font-size: 1.1rem;
            font-weight: 600;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 10px rgba(var(--primary-rgb), 0.25);
            margin-top: 1.5rem;
        }
        .pay-button:hover:not(:disabled) {
            background: linear-gradient(45deg, var(--primary-hover), color-mix(in srgb, var(--secondary) 90%, black 10%));
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(var(--primary-rgb), 0.35);
        }
        .pay-button:disabled {
            background: var(--border);
            cursor: not-allowed;
            box-shadow: none;
        }

        .footer-secure {
            text-align: center;
            margin-top: 2rem;
            color: var(--subtext);
            font-size: 0.9rem;
        }
        .footer-secure i {
            color: var(--success);
            margin-right: 0.3rem;
        }

    </style>
</head>
<body>

    <header class="payment-header">
        <h1 id="payment-plan-title">Complete Your Purchase</h1>
    </header>

    <main class="payment-container">
        <section class="order-summary">
            <h2>Order Summary</h2>
            <div class="summary-item">
                <span class="label">Plan:</span>
                <span class="value" id="selected-plan-name">Loading...</span>
            </div>
            <div class="summary-item">
                <span class="label">Price:</span>
                <span class="value" id="selected-plan-price">Loading...</span>
            </div>
            <div class="summary-item">
                <span class="label">Billing:</span>
                <span class="value">Recurring Monthly</span>
            </div>
        </section>

        <section class="payment-form">
            <h2>Payment Details</h2>
            <form id="payment-form">
                <div class="form-group">
                    <label for="card-name">Name on Card</label>
                    <input type="text" id="card-name" placeholder="J.A.R.V.I.S." required>
                </div>
                <div class="form-group">
                    <label for="card-number">Card Number</label>
                    <input type="text" id="card-number" placeholder="•••• •••• •••• ••••" required pattern="\d{13,19}" title="Enter a valid card number">
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label for="expiry-date">Expiry Date</label>
                        <input type="text" id="expiry-date" placeholder="MM/YY" required pattern="(0[1-9]|1[0-2])\/\d{2}" title="MM/YY">
                    </div>
                    <div class="form-group">
                        <label for="cvv">CVV</label>
                        <input type="text" id="cvv" placeholder="•••" required pattern="\d{3,4}" title="3 or 4 digit CVV">
                    </div>
                </div>
                <button type="submit" class="pay-button" id="pay-now-button">Pay Securely</button>
            </form>
        </section>
    </main>

    <footer class="footer-secure">
        <p><i class="fas fa-lock"></i> Secure SSL Encrypted Payment</p>
    </footer>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const params = new URLSearchParams(window.location.search);
            const planName = params.get('plan');
            const priceString = params.get('price'); // e.g., "₹99"
            const interval = params.get('interval'); // e.g., "/month"

            const planNameElement = document.getElementById('selected-plan-name');
            const planPriceElement = document.getElementById('selected-plan-price');
            const paymentTitleElement = document.getElementById('payment-plan-title');

            if (planNameElement) planNameElement.textContent = planName || 'N/A';
            if (planPriceElement) planPriceElement.textContent = (priceString && interval) ? `${priceString}${interval}` : 'N/A';
            if (paymentTitleElement) paymentTitleElement.textContent = `Complete Payment for ${planName || 'Selected Plan'}`;

            const paymentForm = document.getElementById('payment-form');
            if (paymentForm) {
                paymentForm.addEventListener('submit', (event) => {
                    event.preventDefault();
                    
                    const payButton = document.getElementById('pay-now-button');
                    payButton.textContent = 'Processing...';
                    payButton.disabled = true;

                    // Simulate payment processing
                    setTimeout(() => {
                        alert(`Payment for ${planName} (${priceString}${interval}) submitted successfully! (This is a demo and no actual payment was processed)`);
                        payButton.textContent = 'Payment Successful!';
                        payButton.style.background = 'var(--success)';
                        // Consider disabling form fields as well
                        paymentForm.querySelectorAll('input').forEach(input => input.disabled = true);
                    }, 2000);
                });
            }
        });
    </script>

</body>
</html>
"""

TERM_POLICY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JARVIS 5.0 - Terms of Service</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f172a;
            --card: #1e293b;
            --border: #334155;
            --primary: #6366f1;
            --secondary: #8b5cf6;
            --text: #f8fafc;
            --subtext: #94a3b8;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 0;
            line-height: 1.7;
        }
        .policy-header {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 1.5rem 1rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .policy-header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 600;
        }
        .content-container {
            max-width: 800px;
            margin: 2rem auto;
            padding: 2rem;
            background-color: var(--card);
            border-radius: 0.5rem;
            border: 1px solid var(--border);
        }
        .content-container h2 {
            color: var(--primary);
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            font-size: 1.5rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.3rem;
        }
        .content-container p, .content-container ul {
            color: var(--subtext);
            margin-bottom: 1rem;
        }
        .content-container ul {
            padding-left: 1.5rem;
        }
        .content-container li {
            margin-bottom: 0.5rem;
        }
        .policy-footer {
            text-align: center;
            margin-top: 2rem;
            margin-bottom: 2rem;
            padding: 0 1rem;
            color: var(--subtext);
            font-size: 0.9rem;
        }
        .policy-footer a {
            color: var(--primary);
            text-decoration: none;
        }
        .policy-footer a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <header class="policy-header">
        <h1>Terms of Service</h1>
    </header>

    <main class="content-container">
        <p><em>Last Updated: October 26, 2023</em></p>

        <p>Welcome to JARVIS 5.0! These Terms of Service ("Terms") govern your access to and use of the JARVIS 5.0 website, applications, and services (collectively, the "Service") provided by JARVIS 5.0 ("we," "us," or "our"). By accessing or using the Service, you agree to be bound by these Terms. If you do not agree to these Terms, do not use the Service.</p>

        <h2>1. Acceptance of Terms</h2>
        <p>By creating an account, accessing, or using any part of the Service, you represent that you have read, understood, and agree to be bound by these Terms, including any future modifications. We may amend these Terms from time to time. If we make material changes, we will notify you by posting an announcement on the Service or by sending you an email.</p>

        <h2>2. Description of Service</h2>
        <p>JARVIS 5.0 provides an AI-powered assistant designed to help with various tasks, including text generation, image generation, and voice interactions. The features and capabilities of the Service may vary depending on your subscription plan.</p>

        <h2>3. User Accounts</h2>
        <p>To access certain features of the Service, you may be required to create an account. You agree to:</p>
        <ul>
            <li>Provide accurate, current, and complete information during the registration process.</li>
            <li>Maintain and promptly update your account information.</li>
            <li>Maintain the security of your password and accept all risks of unauthorized access to your account.</li>
            <li>Notify us immediately if you discover or otherwise suspect any security breaches related to the Service or your account.</li>
        </ul>

        <h2>4. User Conduct and Responsibilities</h2>
        <p>You agree not to use the Service for any unlawful purpose or in any way that interrupts, damages, or impairs the Service. Prohibited activities include, but are not limited to:</p>
        <ul>
            <li>Violating any applicable local, state, national, or international law.</li>
            <li>Generating or disseminating content that is hateful, defamatory, obscene, or otherwise objectionable.</li>
            <li>Attempting to gain unauthorized access to the Service or its related systems or networks.</li>
            <li>Using the Service to develop a competing product or service.</li>
        </ul>

        <h2>5. Subscription, Fees, and Payment</h2>
        <p>Certain features of the Service may be offered on a subscription basis. By subscribing to the Service, you agree to pay the applicable fees as described on our pricing page. Fees are non-refundable except as required by law or as explicitly stated in our refund policy. Subscriptions may auto-renew unless cancelled. You are responsible for all applicable taxes.</p>

        <h2>6. Intellectual Property</h2>
        <p>The Service and its original content (excluding content provided by users), features, and functionality are and will remain the exclusive property of JARVIS 5.0 and its licensors. Our trademarks and trade dress may not be used in connection with any product or service without our prior written consent.</p>
        <p>You retain ownership of any intellectual property rights that you hold in the content you submit to the Service. By submitting content, you grant us a worldwide, non-exclusive, royalty-free license to use, reproduce, modify, and distribute such content solely for the purpose of operating, providing, and improving the Service.</p>

        <h2>7. Termination</h2>
        <p>We may terminate or suspend your access to the Service immediately, without prior notice or liability, for any reason whatsoever, including, without limitation, if you breach these Terms. Upon termination, your right to use the Service will immediately cease. You may terminate your account at any time by following the instructions on the Service.</p>

        <h2>8. Disclaimer of Warranties</h2>
        <p>The Service is provided on an "AS IS" and "AS AVAILABLE" basis. We make no warranties, express or implied, regarding the operation or availability of the Service, or the information, content, or materials included therein. You expressly agree that your use of the Service is at your sole risk.</p>

        <h2>9. Limitation of Liability</h2>
        <p>To the maximum extent permitted by applicable law, in no event shall JARVIS 5.0, its affiliates, directors, employees, or licensors be liable for any indirect, punitive, incidental, special, consequential, or exemplary damages, including without limitation damages for loss of profits, goodwill, use, data, or other intangible losses, arising out of or relating to the use of, or inability to use, the Service.</p>

        <h2>10. Governing Law</h2>
        <p>These Terms shall be governed and construed in accordance with the laws of [Your Jurisdiction/Country, e.g., State of California, USA], without regard to its conflict of law provisions.</p>

        <h2>11. Changes to Terms</h2>
        <p>We reserve the right, at our sole discretion, to modify or replace these Terms at any time. If a revision is material, we will provide at least 30 days' notice prior to any new terms taking effect. What constitutes a material change will be determined at our sole discretion.</p>

        <h2>12. Contact Us</h2>
        <p>If you have any questions about these Terms, please contact us at: support@jarvis5.example.com</p>
    </main>

<footer class="policy-footer">
    <p><a href="/plan">Back to Subscription Plans</a></p>
    <p>© 2024 JARVIS 5.0. All rights reserved.</p>
</footer>
</body>
</html>
"""
PRIVACY_POLICY_HTML= """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JARVIS 5.0 - Privacy Policy</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0f172a;
            --card: #1e293b;
            --border: #334155;
            --primary: #6366f1;
            --secondary: #8b5cf6;
            --text: #f8fafc;
            --subtext: #94a3b8;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 0;
            line-height: 1.7;
        }
        .policy-header {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 1.5rem 1rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .policy-header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 600;
        }
        .content-container {
            max-width: 800px;
            margin: 2rem auto;
            padding: 2rem;
            background-color: var(--card);
            border-radius: 0.5rem;
            border: 1px solid var(--border);
        }
        .content-container h2 {
            color: var(--primary);
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            font-size: 1.5rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.3rem;
        }
        .content-container p, .content-container ul {
            color: var(--subtext);
            margin-bottom: 1rem;
        }
        .content-container ul {
            padding-left: 1.5rem;
        }
        .content-container li {
            margin-bottom: 0.5rem;
        }
        .policy-footer {
            text-align: center;
            margin-top: 2rem;
            margin-bottom: 2rem;
            padding: 0 1rem;
            color: var(--subtext);
            font-size: 0.9rem;
        }
        .policy-footer a {
            color: var(--primary);
            text-decoration: none;
        }
        .policy-footer a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <header class="policy-header">
        <h1>Privacy Policy</h1>
    </header>

    <main class="content-container">
        <p><em>Last Updated: October 26, 2023</em></p>

        <p>JARVIS 5.0 ("we," "us," or "our") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you use our website, applications, and services (collectively, the "Service"). Please read this privacy policy carefully. If you do not agree with the terms of this privacy policy, please do not access the Service.</p>

        <h2>1. Information We Collect</h2>
        <p>We may collect information about you in a variety of ways. The information we may collect via the Service includes:</p>
        <ul>
            <li><strong>Personal Data:</strong> Personally identifiable information, such as your name, email address, and payment information, that you voluntarily give to us when you register for the Service or when you choose to participate in various activities related to the Service.</li>
            <li><strong>Usage Data:</strong> Information our servers automatically collect when you access the Service, such as your IP address, browser type, operating system, access times, and the pages you have viewed directly before and after accessing the Service.</li>
            <li><strong>Cookies and Tracking Technologies:</strong> We may use cookies, web beacons, tracking pixels, and other tracking technologies on the Service to help customize the Service and improve your experience.</li>
            <li><strong>Content Data:</strong> Any text, images, or other content you input, upload, or generate through the Service.</li>
        </ul>

        <h2>2. How We Use Your Information</h2>
        <p>Having accurate information about you permits us to provide you with a smooth, efficient, and customized experience. Specifically, we may use information collected about you via the Service to:</p>
        <ul>
            <li>Create and manage your account.</li>
            <li>Provide, operate, and maintain our Service.</li>
            <li>Improve, personalize, and expand our Service.</li>
            <li>Understand and analyze how you use our Service.</li>
            <li>Develop new products, services, features, and functionality.</li>
            <li>Communicate with you, either directly or through one of our partners, including for customer service, to provide you with updates and other information relating to the Service, and for marketing and promotional purposes.</li>
            <li>Process your transactions and manage your subscriptions.</li>
            <li>Find and prevent fraud.</li>
            <li>Comply with legal obligations.</li>
        </ul>

        <h2>3. Disclosure of Your Information</h2>
        <p>We may share information we have collected about you in certain situations. Your information may be disclosed as follows:</p>
        <ul>
            <li><strong>By Law or to Protect Rights:</strong> If we believe the release of information about you is necessary to respond to legal process, to investigate or remedy potential violations of our policies, or to protect the rights, property, and safety of others, we may share your information as permitted or required by any applicable law, rule, or regulation.</li>
            <li><strong>Third-Party Service Providers:</strong> We may share your information with third parties that perform services for us or on our behalf, including payment processing, data analysis, email delivery, hosting services, customer service, and marketing assistance.</li>
            <li><strong>Business Transfers:</strong> We may share or transfer your information in connection with, or during negotiations of, any merger, sale of company assets, financing, or acquisition of all or a portion of our business to another company.</li>
            <li><strong>With Your Consent:</strong> We may disclose your personal information for any other purpose with your consent.</li>
        </ul>
        <p>We do not sell your personal information to third parties.</p>

        <h2>4. Data Security</h2>
        <p>We use administrative, technical, and physical security measures to help protect your personal information. While we have taken reasonable steps to secure the personal information you provide to us, please be aware that despite our efforts, no security measures are perfect or impenetrable, and no method of data transmission can be guaranteed against any interception or other type of misuse.</p>

        <h2>5. Your Data Protection Rights</h2>
        <p>Depending on your location, you may have the following rights regarding your personal information:</p>
        <ul>
            <li>The right to access – You have the right to request copies of your personal data.</li>
            <li>The right to rectification – You have the right to request that we correct any information you believe is inaccurate or complete information you believe is incomplete.</li>
            <li>The right to erasure – You have the right to request that we erase your personal data, under certain conditions.</li>
            <li>The right to restrict processing – You have the right to request that we restrict the processing of your personal data, under certain conditions.</li>
            <li>The right to object to processing – You have the right to object to our processing of your personal data, under certain conditions.</li>
            <li>The right to data portability – You have the right to request that we transfer the data that we have collected to another organization, or directly to you, under certain conditions.</li>
        </ul>
        <p>If you wish to exercise any of these rights, please contact us.</p>

        <h2>6. Children's Privacy</h2>
        <p>Our Service is not intended for use by children under the age of 13 (or the equivalent minimum age in the relevant jurisdiction). We do not knowingly collect personal information from children under 13. If we become aware that a child under 13 has provided us with personal information, we will take steps to delete such information.</p>

        <h2>7. Changes to This Privacy Policy</h2>
        <p>We may update this Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page and updating the "Last Updated" date. You are advised to review this Privacy Policy periodically for any changes.</p>

        <h2>8. Contact Us</h2>
        <p>If you have questions or comments about this Privacy Policy, please contact us at: support@jarvis5.example.com</p>
    </main>

<footer class="policy-footer">
    <p><a href="/plan">Back to Subscription Plans</a></p>
    <p>© 2024 JARVIS 5.0. All rights reserved.</p>
</footer>
</body>
</html>
"""
GUI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JARVIS 5.0 - Voice Interface</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    :root {
      --primary: #6366f1;
      --primary-rgb: 99, 102, 241;
      --primary-hover: #4f46e5;
      --primary-active: #3f3aed; /* For active button press */
      --bg: #0f172a;
      --card: #1e293b;
      --text: #f8fafc;
      --subtext: #94a3b8;
      --success: #10b981;
      --success-hover: #0d9e6e;
      --warning: #f59e0b;
      --warning-rgb: 245, 158, 11;
      --error: #ef4444;
      --danger: #ef4444;
      --danger-rgb: 239, 68, 68;
      --danger-hover: #dc2626;

      --listening-wave1: #f0f0f0;
      --listening-wave2: #e0e0e0;
      --listening-glow-rgb: 230, 230, 230;
      --listening-btn-bg: #f0f0f0; /* Light gray for listening button bg */
      --listening-btn-text: #1e293b; /* Dark text for light button */
      --border: #334155;
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', sans-serif;
      min-height: 100vh; /* Use min-height for better flexibility */
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      padding: 1rem; /* Add some padding to body for smaller screens */
    }
    .container {
      width: 100%;
      max-width: 800px;
      padding: 2rem 1rem; /* Adjusted padding */
      text-align: center;
    }
    .logo {
      font-size: 3rem;
      margin-bottom: 1rem;
      color: var(--primary);
      animation: pulseLogo 2s infinite;
    }
    @keyframes pulseLogo {
      0% { transform: scale(1); }
      50% { transform: scale(1.08); }
      100% { transform: scale(1); }
    }
    h1 {
      font-size: clamp(1.5rem, 5vw, 2rem); /* Responsive font size */
      margin-bottom: 2rem;
    }
    .response-container {
      background: var(--card);
      border-radius: 1rem;
      padding: 1.5rem; /* Adjusted padding */
      min-height: 150px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: space-between; /* Pushes content and waves apart */
      margin-bottom: 2rem;
      position: relative;
      overflow: hidden;
      width: 100%;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    .response-text {
      font-size: 1.1rem; /* Slightly smaller for more content */
      line-height: 1.6;
      text-align: left;
      width: 100%;
      z-index: 2;
      position: relative;
      min-height: 2.5em; /* Ensure some space */
      padding-bottom: 0.5rem; /* Space before TTS controls if any */
    }
    .voice-btn {
      width: 72px; /* Slightly smaller */
      height: 72px;
      border-radius: 50%;
      background: linear-gradient(145deg, var(--primary), var(--primary-hover));
      color: white;
      border: none;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.8rem; /* Adjusted icon size */
      cursor: pointer;
      transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1);
      margin: 0 auto;
      box-shadow: 0 5px 15px rgba(var(--primary-rgb), 0.2),
                  0 2px 5px rgba(var(--primary-rgb), 0.15);
      position: relative;
      overflow: hidden;
    }
    .voice-btn:not(.listening):hover {
      background: linear-gradient(145deg, var(--primary-hover), var(--primary-active));
      transform: translateY(-2px) scale(1.03);
      box-shadow: 0 8px 20px rgba(var(--primary-rgb), 0.25),
                  0 3px 7px rgba(var(--primary-rgb), 0.2);
    }
    .voice-btn:not(.listening):active {
      background: var(--primary-active);
      transform: translateY(0px) scale(0.97);
      box-shadow: 0 2px 8px rgba(var(--primary-rgb), 0.2);
    }
    .voice-btn.listening {
      animation: pulseButtonListening 1.5s infinite;
      background: var(--listening-btn-bg);
      color: var(--listening-btn-text);
      box-shadow: 0 4px 15px rgba(var(--listening-glow-rgb), 0.2); /* Base shadow for listening */
    }
    @keyframes pulseButtonListening {
      0% { transform: scale(1); box-shadow: 0 4px 15px rgba(var(--listening-glow-rgb), 0.2); }
      50% { transform: scale(1.08); box-shadow: 0 6px 20px rgba(var(--listening-glow-rgb), 0.35); } /* Brighter glow */
      100% { transform: scale(1); box-shadow: 0 4px 15px rgba(var(--listening-glow-rgb), 0.2); }
    }
    .status {
      margin-top: 1.25rem; /* More space */
      color: var(--subtext);
      font-size: 0.9rem;
      min-height: 1.2em;
      line-height: 1.4;
    }
    .back-btn, .language-btn {
      position: absolute;
      top: 1rem;
      background: transparent;
      color: var(--subtext);
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 0.5rem 1rem;
      cursor: pointer;
      display: inline-flex; /* Changed to inline-flex */
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s;
      z-index: 10;
      font-size: 0.9rem;
    }
    .back-btn { left: 1rem; }
    .language-selector { position: absolute; top: 1rem; right: 1rem; z-index: 10; }
    .language-btn { position: static; /* No longer absolute within its parent */ }

    .back-btn:hover, .language-btn:hover {
      background: rgba(var(--primary-rgb), 0.08);
      color: var(--text);
      border-color: rgba(var(--primary-rgb), 0.25);
    }

    /* Water Wave Animation - (No changes from original, seems fine) */
    .water-wave-container {
      position: absolute;
      bottom: 0;
      left: 0;
      width: 100%;
      height: 60px;
      overflow: hidden;
      opacity: 0;
      transition: opacity 0.5s ease-in-out;
      z-index: 1;
    }
    .response-container.wave-active .water-wave-container { opacity: 1; }
    .water-wave-svg { width: 100%; height: 100%; }
    .water-wave-svg path {
      --wave-offset-x: 0px; --wave-offset-y: 0px; --wave-scale-y: 1;
      transform: translateX(var(--wave-offset-x)) translateY(var(--wave-offset-y)) scaleY(var(--wave-scale-y));
      transform-origin: 50% 100%;
    }
    @keyframes wave-flow { 0% { --wave-offset-x: 0px; } 100% { --wave-offset-x: -1200px; } }
    @keyframes wave-amplitude-idle {  0%, 100% { --wave-offset-y: 0px; --wave-scale-y: 1; } 50% { --wave-offset-y: -3px; --wave-scale-y: 1.03; } }
    .response-container.state-idle .water-wave-svg #wavePath1 { fill: var(--primary); fill-opacity: 0.5; animation: wave-flow 25s linear infinite, wave-amplitude-idle 10s ease-in-out infinite alternate;  }
    .response-container.state-idle .water-wave-svg #wavePath2 { fill: var(--primary-hover); fill-opacity: 0.35; animation: wave-flow 30s linear infinite reverse 0.5s, wave-amplitude-idle 10s ease-in-out infinite alternate 0.5s;  }
    .response-container.state-idle .water-wave-svg #wavePath3 { fill: var(--primary); fill-opacity: 0.25; animation: wave-flow 35s linear infinite 1s, wave-amplitude-idle 10s ease-in-out infinite alternate 1s; }
    @keyframes wave-amplitude-listening {  0%, 100% { --wave-offset-y: 0px; --wave-scale-y: 1; } 50% { --wave-offset-y: -10px; --wave-scale-y: 1.15; } }
    .response-container.state-listening .water-wave-svg #wavePath1 { fill: var(--listening-wave1); fill-opacity: 0.6; animation: wave-flow 7s linear infinite, wave-amplitude-listening 1.5s ease-in-out infinite alternate; }
    .response-container.state-listening .water-wave-svg #wavePath2 { fill: var(--listening-wave2); fill-opacity: 0.45; animation: wave-flow 9s linear infinite reverse 0.3s, wave-amplitude-listening 1.7s ease-in-out infinite alternate 0.2s; }
    .response-container.state-listening .water-wave-svg #wavePath3 { fill: var(--listening-wave1); fill-opacity: 0.35; animation: wave-flow 11s linear infinite 0.6s, wave-amplitude-listening 1.9s ease-in-out infinite alternate 0.4s; }
    @keyframes wave-amplitude-processing { 0% { --wave-offset-y: 2px; --wave-scale-y: 0.9; } 50% { --wave-offset-y: -15px; --wave-scale-y: 1.35; } 100% { --wave-offset-y: 2px; --wave-scale-y: 0.9; } }
    .response-container.state-processing .water-wave-svg #wavePath1 { fill: var(--warning); fill-opacity: 0.5; animation: wave-flow 3s linear infinite, wave-amplitude-processing 0.8s ease-in-out infinite alternate; }
    .response-container.state-processing .water-wave-svg #wavePath2 { fill: #ffc107; fill-opacity: 0.35; animation: wave-flow 3.5s linear infinite reverse 0.2s, wave-amplitude-processing 0.9s ease-in-out infinite alternate 0.1s; }
    .response-container.state-processing .water-wave-svg #wavePath3 { fill: var(--warning); fill-opacity: 0.25; animation: wave-flow 4s linear infinite 0.4s, wave-amplitude-processing 1s ease-in-out infinite alternate 0.2s; }
    @keyframes wave-amplitude-speaking {  0%, 100% { --wave-offset-y: 0px; --wave-scale-y: 1; } 50% { --wave-offset-y: -7px; --wave-scale-y: 1.08; } }
    .response-container.state-speaking .water-wave-svg #wavePath1 { fill: var(--danger); fill-opacity: 0.5; animation: wave-flow 15s linear infinite, wave-amplitude-speaking 2.5s ease-in-out infinite alternate; }
    .response-container.state-speaking .water-wave-svg #wavePath2 { fill: var(--danger-hover); fill-opacity: 0.35; animation: wave-flow 18s linear infinite reverse 0.5s, wave-amplitude-speaking 2.7s ease-in-out infinite alternate 0.3s; }
    .response-container.state-speaking .water-wave-svg #wavePath3 { fill: var(--danger); fill-opacity: 0.25; animation: wave-flow 22s linear infinite 1s, wave-amplitude-speaking 2.9s ease-in-out infinite alternate 0.6s; }

    .tts-controls {
      display: flex;
      flex-wrap: wrap; /* Allow wrapping on small screens */
      gap: 0.75rem; /* Increased gap */
      margin-top: 1rem;
      justify-content: center;
      z-index: 2;
      position: relative; /* Ensure it's above waves if absolutely positioned, but it's in flow */
      width: 100%; /* Take full width for centering */
    }
    .play-tts-btn, .tts-download-btn {
      color: white;
      border: none;
      padding: 0.6rem 1.2rem; /* Slightly larger padding */
      border-radius: 0.5rem; /* Consistent radius */
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.85rem; /* Slightly larger font */
      transition: background-color 0.2s, transform 0.1s;
      text-decoration: none; /* For download link */
    }
    .play-tts-btn { background: var(--primary); }
    .play-tts-btn:hover { background: var(--primary-hover); }
    .play-tts-btn.playing { background: var(--warning); color: var(--bg); }
    .play-tts-btn.playing:hover { background: #fcae1e; }

    .tts-download-btn { background: var(--success); }
    .tts-download-btn:hover { background: var(--success-hover); }
    
    .play-tts-btn:active, .tts-download-btn:active {
        transform: scale(0.97);
    }

    /* Response Animation Glows (No changes from original) */
    .response-animation { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(135deg, rgba(var(--primary-rgb), 0.1) 0%, rgba(var(--primary-rgb), 0) 100%); opacity: 0; transition: opacity 0.3s ease, background 0.5s ease; z-index: 0; pointer-events: none; }
    .response-container.active .response-animation { opacity: 1; }
    .response-container.state-processing .response-animation { background: radial-gradient(circle, rgba(var(--warning-rgb), 0.25) 0%, rgba(var(--warning-rgb), 0.05) 60%, transparent 80%); animation: processingGlowPulse 1.2s infinite ease-in-out; }
    .response-container.state-listening .response-animation { background: radial-gradient(circle, rgba(var(--listening-glow-rgb), 0.15) 0%, rgba(var(--listening-glow-rgb), 0.02) 60%, transparent 80%); animation: processingGlowPulse 1.8s infinite ease-in-out; }
    .response-container.state-speaking .response-animation { background: radial-gradient(circle, rgba(var(--danger-rgb), 0.20) 0%, rgba(var(--danger-rgb), 0.03) 60%, transparent 80%); animation: processingGlowPulse 1.5s infinite ease-in-out; }
    @keyframes processingGlowPulse { 0% { transform: scale(1); opacity: 0.7; } 50% { transform: scale(1.03); opacity: 1; } 100% { transform: scale(1); opacity: 0.7; } }
    
    /* Language selector dropdown (No changes from original) */
    .language-dropdown { position: absolute; top: calc(100% + 0.5rem); right: 0; background: var(--card); border: 1px solid var(--border); border-radius: 0.5rem; padding: 0.5rem; display: none; z-index: 20; min-width: 160px; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
    .language-dropdown.visible { display: block; }
    .language-option { padding: 0.6rem 1rem; border-radius: 0.25rem; cursor: pointer; display: flex; align-items: center; gap: 0.75rem; font-size: 0.9rem;}
    .language-option:hover { background: rgba(var(--primary-rgb), 0.1); }
    .language-option.active { background: rgba(var(--primary-rgb), 0.2); color: var(--primary); }
    .language-option i.fa-check { width: 1em; color: var(--primary); }

    /* Accessibility: Focus Visible */
    button:focus-visible, a:focus-visible, div[data-lang]:focus-visible {
      outline: 2px solid var(--primary);
      outline-offset: 2px;
    }
    .voice-btn:focus-visible { outline-offset: 4px; }
    .language-dropdown:focus-visible { outline: none; /* Handled by child elements */ }

  </style>
</head>
<body>
  <button class="back-btn" id="backBtn" title="Back to Chat">
    <i class="fas fa-arrow-left"></i> Back to Chat
  </button>
  
  <div class="language-selector" id="languageSelector">
    <button class="language-btn" id="languageBtn" aria-haspopup="true" aria-expanded="false" title="Change language">
      <i class="fas fa-language"></i>
      <span id="currentLanguage">English</span>
      <i class="fas fa-chevron-down" style="font-size: 0.7em; margin-left: 0.3em;"></i>
    </button>
    <div class="language-dropdown" id="languageDropdown" role="menu">
      <div class="language-option" data-lang="en" role="menuitemradio" aria-checked="false">
        <i class="fas fa-check" style="visibility: hidden;"></i>
        <span>English</span>
      </div>
      <div class="language-option" data-lang="hi" role="menuitemradio" aria-checked="false">
        <i class="fas fa-check" style="visibility: hidden;"></i>
        <span>हिंदी (Hindi)</span>
      </div>
    </div>
  </div>
  
  <div class="container">
    <div class="logo">
      <i class="fas fa-robot"></i>
    </div>
    <h1 id="title">JARVIS 5.0 Voice Interface</h1>
    
    <div class="response-container" id="responseContainer">
      <div class="response-animation"></div>
      
      <div class="response-text" id="responseText" aria-live="polite">
        <!-- Responses are primarily spoken. This area can show transcripts or key info if needed. -->
      </div>
      <!-- TTS controls will be appended here by JavaScript -->
      <div class="water-wave-container" id="waterWaveContainer" aria-hidden="true">
        <svg class="water-wave-svg" viewBox="0 0 1200 60" preserveAspectRatio="none">
            <path id="wavePath1" d="M-1200,37.5 Q-900,52.5 -600,37.5 T0,37.5 Q300,22.5 600,37.5 T1200,37.5 Q1500,52.5 1800,37.5 T2400,37.5 V60 H-1200 Z"></path>
            <path id="wavePath2" d="M-1200,41.25 Q-900,45 -600,41.25 T0,41.25 Q300,22.5 600,41.25 T1200,41.25 Q1500,45 1800,41.25 T2400,41.25 V60 H-1200 Z"></path>
            <path id="wavePath3" d="M-1200,45 Q-900,56.25 -600,45 T0,45 Q300,33.75 600,45 T1200,45 Q1500,56.25 1800,45 T2400,45 V60 H-1200 Z"></path>
        </svg>
      </div>
    </div>
    
    <button class="voice-btn" id="voiceBtn" title="Activate Voice Input">
      <i class="fas fa-microphone"></i>
    </button>
    <div class="status" id="statusText" aria-live="assertive">
      Click the microphone to speak
    </div>
  </div>

<script>
// Global state variables
let isProcessing = false;
let currentLanguage = localStorage.getItem('jarvis_language') || 'en';
const languageMap = {
  'en': 'English',
  'hi': 'हिंदी (Hindi)'
};
const greetings = {
  'en': "Hello! I'm JARVIS. Click the microphone to speak with me.",
  'hi': "नमस्ते! मैं JARVIS हूँ। माइक्रोफ़ोन पर क्लिक करके बात करें।"
};
const statusMessages = {
    clickToSpeak: {
        'en': "Click the microphone to speak",
        'hi': "बोलने के लिए माइक्रोफ़ोन पर क्लिक करें"
    },
    listening: {
        'en': "Listening... Speak now",
        'hi': "सुन रहा हूँ... बोलें"
    },
    recognized: {
        'en': "Recognized: ",
        'hi': "पहचाना गया: "
    },
    processing: {
        'en': "Processing your request...",
        'hi': "आपका अनुरोध प्रसंस्करण..."
    },
    errorGeneral: {
        'en': "Error: ",
        'hi': "त्रुटि: "
    },
    errorNoSpeech: {
        'en': "No speech detected",
        'hi': "कोई आवाज़ नहीं मिली"
    },
    errorAudioCapture: {
        'en': "Microphone issue",
        'hi': "माइक्रोफ़ोन समस्या"
    },
    errorServer: {
        'en': "Server error: ",
        'hi': "सर्वर त्रुटि: "
    },
    errorNetwork: {
        'en': "Network error: ",
        'hi': "नेटवर्क त्रुटि: "
    },
    jarvisSpeaking: {
        'en': "JARVIS is speaking...",
        'hi': "JARVIS बोल रहा है..."
    },
    speechRecognitionNotSupported: {
        'en': "Speech recognition not supported in your browser",
        'hi': "आपके ब्राउज़र में स्पीच रिकग्निशन सपोर्ट नहीं है"
    },
    speechSynthesisNotSupported: {
        'en': "Speech synthesis not supported in your browser",
        'hi': "आपके ब्राउज़र में स्पीच सिंथेसिस सपोर्ट नहीं है"
    }
};


// DOM elements
const backBtn = document.getElementById('backBtn');
const responseText = document.getElementById('responseText');
const voiceBtn = document.getElementById('voiceBtn');
const statusText = document.getElementById('statusText');
const responseContainer = document.getElementById('responseContainer');
const waterWaveContainer = document.getElementById('waterWaveContainer');
const titleElement = document.getElementById('title');
const currentLanguageElement = document.getElementById('currentLanguage');
const languageBtn = document.getElementById('languageBtn');
const languageDropdown = document.getElementById('languageDropdown');
const languageOptions = document.querySelectorAll('.language-option');

// Speech recognition and synthesis
let recognition;
let isListening = false;
let currentAudio = null;
let currentTTSFile = null;
let greetingAttempted = false; // Flag for initial greeting

function updateWaveState(state) {
    responseContainer.classList.remove('state-idle', 'state-listening', 'state-processing', 'state-speaking');
    if (state === 'idle') responseContainer.classList.add('state-idle');
    else if (state === 'listening') responseContainer.classList.add('state-listening');
    else if (state === 'processing') responseContainer.classList.add('state-processing');
    else if (state === 'speaking') responseContainer.classList.add('state-speaking');

    if (['idle', 'listening', 'processing', 'speaking'].includes(state)) {
        responseContainer.classList.add('wave-active', 'active');
    } else {
        responseContainer.classList.remove('wave-active', 'active');
    }
}

function setLanguage(lang) {
    currentLanguage = lang;
    localStorage.setItem('jarvis_language', lang);
    currentLanguageElement.textContent = languageMap[lang];
    languageBtn.setAttribute('aria-expanded', 'false');
    
    languageOptions.forEach(option => {
        const isActive = option.dataset.lang === lang;
        option.classList.toggle('active', isActive);
        option.setAttribute('aria-checked', isActive.toString());
        option.querySelector('i.fa-check').style.visibility = isActive ? 'visible' : 'hidden';
    });
    
    titleElement.textContent = lang === 'hi' ? "JARVIS 5.0 वॉइस इंटरफ़ेस" : "JARVIS 5.0 Voice Interface";
    if (!isListening && !isProcessing && (!currentAudio || currentAudio.paused) && !window.speechSynthesis.speaking) {
        statusText.textContent = statusMessages.clickToSpeak[lang];
    }
    
    if (recognition) {
        recognition.lang = lang === 'hi' ? 'hi-IN' : 'en-US';
    }
    
    languageDropdown.classList.remove('visible');
}

function initializeSpeechRecognition() {
    try {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            throw new Error("SpeechRecognition API not found.");
        }
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = currentLanguage === 'hi' ? 'hi-IN' : 'en-US';

        recognition.onstart = () => {
            isListening = true;
            voiceBtn.classList.add('listening');
            voiceBtn.title = currentLanguage === 'hi' ? "सुनना बंद करें" : "Stop Listening";
            statusText.textContent = statusMessages.listening[currentLanguage];
            updateWaveState('listening');
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            statusText.textContent = `${statusMessages.recognized[currentLanguage]}"${transcript}"`;
            processInput(transcript);
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error, event.message);
            let errorKey;
            switch(event.error) {
                case 'no-speech': errorKey = 'errorNoSpeech'; break;
                case 'audio-capture': errorKey = 'errorAudioCapture'; break;
                default: errorKey = 'errorGeneral';
            }
            const specificErrorMsg = statusMessages[errorKey] ? statusMessages[errorKey][currentLanguage] : event.error;
            const errorMsg = errorKey === 'errorGeneral' ? `${specificErrorMsg}${event.error}` : specificErrorMsg;
            
            statusText.textContent = errorMsg;
            if (event.error !== 'no-speech' && event.error !== 'aborted') { // Don't speak for "no-speech"
                speakResponse(currentLanguage === 'hi' ? `भाषण पहचान त्रुटि: ${event.error}` : `Speech recognition error: ${event.error}`);
            }
        };

        recognition.onend = () => {
            stopListening();
        };

    } catch (e) {
        console.error('Speech recognition not supported or failed to initialize:', e);
        voiceBtn.style.display = 'none'; // Hide button if STT not available
        statusText.textContent = statusMessages.speechRecognitionNotSupported[currentLanguage];
        recognition = null; // Ensure recognition is null if setup failed
    }
}

function toggleListening() {
    if (!recognition) {
        statusText.textContent = statusMessages.speechRecognitionNotSupported[currentLanguage];
        console.warn("Attempted to use recognition, but it's not initialized.");
        return;
    }

    if (isListening) {
        recognition.stop();
    } else {
        try {
            // Clear previous TTS/audio
            const existingTTSControls = responseContainer.querySelector('.tts-controls');
            if (existingTTSControls) existingTTSControls.remove();
            currentTTSFile = null;
            if (currentAudio) {
                currentAudio.pause();
                currentAudio = null;
            }
            window.speechSynthesis.cancel(); // Stop any ongoing browser TTS
            
            // recognition.lang is set by setLanguage and initializeSpeechRecognition
            recognition.start();
        } catch (e) {
            console.error('Error starting recognition:', e);
            statusText.textContent = (currentLanguage === 'hi' ? "माइक्रोफ़ोन शुरू करने में त्रुटि: " : "Error starting microphone: ") + e.message;
            stopListening(); 
            updateWaveState('idle'); 
        }
    }
}

function stopListening() {
    isListening = false;
    voiceBtn.classList.remove('listening');
    voiceBtn.title = currentLanguage === 'hi' ? "बोलने के लिए दबाएं" : "Activate Voice Input";

    // Update status text only if not processing, speaking, or playing custom audio
    if (!isProcessing && !window.speechSynthesis.speaking && (!currentAudio || currentAudio.paused)) {
        statusText.textContent = statusMessages.clickToSpeak[currentLanguage];
        updateWaveState('idle');
    }
}

async function processInput(input) {
    if (!input.trim()) {
        updateWaveState(isListening ? 'listening' : 'idle');
        return;
    }
    
    isProcessing = true;
    updateWaveState('processing');
    statusText.textContent = statusMessages.processing[currentLanguage];

    const existingTTSControls = responseContainer.querySelector('.tts-controls');
    if (existingTTSControls) existingTTSControls.remove();
    
    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                messages: [{ role: 'user', content: input }],
                language: currentLanguage
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: `HTTP error ${response.status}` }));
            throw new Error(errorData.error || `HTTP error ${response.status}`);
        }
        
        const data = await response.json();
        isProcessing = false; // Set before speakResponse
        
        if (data.error) {
            const errorMsg = `${statusMessages.errorServer[currentLanguage]}${data.error}`;
            speakResponse(errorMsg); // Speak the server error
            statusText.textContent = errorMsg;
            updateWaveState('idle'); // Reset to idle after error
            return; 
        }
        
        // If there's a text response from LLM, it will be in data.response
        // This will be spoken by speakResponse.
        // responseText.textContent = data.response; // Optionally display transcript

        speakResponse(data.response); // This will set state to 'speaking'

        if (data.action === 'tts' && data.tts_file) {
            currentTTSFile = data.tts_file;
            showTTSControls();
        }
        
    } catch (err) {
        isProcessing = false;
        const errorMsg = `${statusMessages.errorNetwork[currentLanguage]}${err.message}`;
        console.error("Processing error:", err);
        speakResponse(errorMsg); // Speak the network error
        statusText.textContent = errorMsg;
        updateWaveState('idle'); // Reset to idle after error
    } 
    // `finally` block removed as state updates are handled within try/catch and speakResponse/playTTS
}

function showTTSControls() {
    const controlsDiv = document.createElement('div');
    controlsDiv.className = 'tts-controls';
    
    const playBtn = document.createElement('button');
    playBtn.className = 'play-tts-btn';
    playBtn.innerHTML = `${currentLanguage === 'hi' ? '<i class="fas fa-play"></i> ऑडियो चलाएं' : '<i class="fas fa-play"></i> Play Audio'}`;
    playBtn.type = 'button';
    playBtn.addEventListener('click', () => playTTS(playBtn));
    
    const downloadBtn = document.createElement('a');
    downloadBtn.href = `/download-tts?file=${encodeURIComponent(currentTTSFile)}`;
    downloadBtn.className = 'tts-download-btn';
    downloadBtn.innerHTML = `${currentLanguage === 'hi' ? '<i class="fas fa-download"></i> ऑडियो डाउनलोड करें' : '<i class="fas fa-download"></i> Download Audio'}`;
    downloadBtn.download = 'tts-audio.mp3'; // Suggest filename
    
    controlsDiv.appendChild(playBtn);
    controlsDiv.appendChild(downloadBtn);
    
    responseText.insertAdjacentElement('afterend', controlsDiv);
}

function playTTS(button) {
    if (!currentTTSFile) {
        statusText.textContent = currentLanguage === 'hi' ? "चलाने के लिए कोई ऑडियो फ़ाइल नहीं" : "No audio file to play.";
        return;
    }

    if (currentAudio && currentAudio.src.includes(encodeURIComponent(currentTTSFile)) && !currentAudio.paused) {
        currentAudio.pause(); // Pause will trigger onpause handler to update UI
        return;
    }
    
    if (currentAudio) currentAudio.pause(); // Pause any existing audio
    window.speechSynthesis.cancel(); // Cancel any browser TTS
    
    currentAudio = new Audio(`/download-tts?file=${encodeURIComponent(currentTTSFile)}`);
    
    currentAudio.onplay = () => {
        updateWaveState('speaking'); // Use 'speaking' state for custom audio too
        button.innerHTML = `${currentLanguage === 'hi' ? '<i class="fas fa-stop"></i> रोकें' : '<i class="fas fa-stop"></i> Stop'}`;
        button.classList.add('playing');
        statusText.textContent = currentLanguage === 'hi' ? "ऑडियो प्रतिक्रिया चल रही है..." : "Playing audio response...";
    };

    currentAudio.play().catch(error => {
        console.error("Error playing audio:", error);
        statusText.textContent = currentLanguage === 'hi' ? "ऑडियो चलाने में विफल" : "Failed to play audio";
        button.innerHTML = `${currentLanguage === 'hi' ? '<i class="fas fa-play"></i> ऑडियो चलाएं' : '<i class="fas fa-play"></i> Play Audio'}`;
        button.classList.remove('playing');
        if (!isProcessing && !isListening) updateWaveState('idle'); // Revert to idle if nothing else is active
        currentAudio = null;
    });
    
    const resetToIdleIfNeeded = () => {
        if (isProcessing) updateWaveState('processing');
        else if (isListening) {
            updateWaveState('listening');
            statusText.textContent = statusMessages.listening[currentLanguage];
        } else {
            updateWaveState('idle');
            statusText.textContent = statusMessages.clickToSpeak[currentLanguage];
        }
    };

    currentAudio.onended = () => {
        button.innerHTML = `${currentLanguage === 'hi' ? '<i class="fas fa-play"></i> ऑडियो चलाएं' : '<i class="fas fa-play"></i> Play Audio'}`;
        button.classList.remove('playing');
        resetToIdleIfNeeded();
    };
    
    currentAudio.onpause = () => { // Handles both explicit pause and pause at end of playback before onended
        if (currentAudio && currentAudio.currentTime === currentAudio.duration && currentAudio.paused) return; // Ignore pause at the very end
        button.innerHTML = `${currentLanguage === 'hi' ? '<i class="fas fa-play"></i> ऑडियो चलाएं' : '<i class="fas fa-play"></i> Play Audio'}`;
        button.classList.remove('playing');
        resetToIdleIfNeeded();
    };

    currentAudio.onerror = (e) => {
        console.error("Audio playback error:", e);
        statusText.textContent = currentLanguage === 'hi' ? "ऑडियो फ़ाइल चलाने में त्रुटि" : "Error playing audio file.";
        button.innerHTML = `${currentLanguage === 'hi' ? '<i class="fas fa-play"></i> ऑडियो चलाएं' : '<i class="fas fa-play"></i> Play Audio'}`;
        button.classList.remove('playing');
        resetToIdleIfNeeded();
        currentAudio = null;
    };
}

function speakResponse(text) {
    if (!('speechSynthesis' in window)) {
        console.warn("Speech synthesis not supported.");
        statusText.textContent = statusMessages.speechSynthesisNotSupported[currentLanguage];
        if (!isProcessing && !isListening && (!currentAudio || currentAudio.paused)) updateWaveState('idle');
        return;
    }

    window.speechSynthesis.cancel(); // Cancel previous speech
    if (currentAudio) currentAudio.pause(); // Pause custom audio if any
        
    const cleanedText = text.replace(/[^\p{L}\p{N}\s.,?!'"`-।॥]/gu, ' ').replace(/\s+/g, ' ').trim();

    if (!cleanedText) {
        console.warn("Cleaned text for speech is empty, not speaking.");
        if (!isProcessing && !isListening && (!currentAudio || currentAudio.paused)) updateWaveState('idle');
        return;
    }

    const utterance = new SpeechSynthesisUtterance(cleanedText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    const voices = window.speechSynthesis.getVoices();
    let selectedVoice = null;

    if (currentLanguage === 'hi') {
        selectedVoice = voices.find(v => v.lang.toLowerCase().startsWith('hi') && (v.name.toLowerCase().includes('male') || v.name.toLowerCase().includes('पुरुष'))) ||
                        voices.find(v => v.lang.toLowerCase().startsWith('hi') && !(v.name.toLowerCase().includes('female') || v.name.toLowerCase().includes('महिला'))) ||
                        voices.find(v => v.lang.toLowerCase().startsWith('hi'));
    } else { // Default 'en'
        selectedVoice = voices.find(v => v.lang.toLowerCase().startsWith('en') && v.name.toLowerCase().includes('male')) ||
                        voices.find(v => v.lang.toLowerCase().startsWith('en') && !v.name.toLowerCase().includes('female')) ||
                        voices.find(v => v.lang.toLowerCase() === 'en-us') ||
                        voices.find(v => v.lang.toLowerCase() === 'en-gb') ||
                        voices.find(v => v.lang.toLowerCase().startsWith('en'));
    }

    if (selectedVoice) utterance.voice = selectedVoice;
    utterance.lang = currentLanguage === 'hi' ? 'hi-IN' : 'en-US'; // Crucial fallback

    utterance.onstart = () => {
        updateWaveState('speaking');
        statusText.textContent = statusMessages.jarvisSpeaking[currentLanguage];
    };

    const resetToIdleIfNeeded = () => {
        if (isProcessing) {
            updateWaveState('processing'); // Stay in processing if backend is still working (though unlikely here)
        } else if (isListening) {
            updateWaveState('listening');
            statusText.textContent = statusMessages.listening[currentLanguage];
        } else {
            updateWaveState('idle');
            statusText.textContent = statusMessages.clickToSpeak[currentLanguage];
        }
    };
    
    utterance.onend = resetToIdleIfNeeded;
    utterance.onerror = (event) => {
        console.error("Speech synthesis error:", event);
        statusText.textContent = (currentLanguage === 'hi' ? "भाषण आउटपुट के दौरान त्रुटि" : "Error during speech output.") + ` (${event.error})`;
        resetToIdleIfNeeded();
    };
    window.speechSynthesis.speak(utterance);
}

// Event Listeners
backBtn.addEventListener('click', () => { window.location.href = '/'; });
voiceBtn.addEventListener('click', toggleListening);

languageBtn.addEventListener('click', () => {
    const isVisible = languageDropdown.classList.toggle('visible');
    languageBtn.setAttribute('aria-expanded', isVisible.toString());
});

languageOptions.forEach(option => {
    option.addEventListener('click', () => {
        setLanguage(option.dataset.lang);
    });
    option.addEventListener('keydown', (e) => { // Keyboard accessibility for dropdown
        if (e.key === 'Enter' || e.key === ' ') {
            setLanguage(option.dataset.lang);
        }
    });
});

document.addEventListener('click', (e) => { // Close dropdown on outside click
    if (!languageSelector.contains(e.target)) {
        languageDropdown.classList.remove('visible');
        languageBtn.setAttribute('aria-expanded', 'false');
    }
});


// Initialization
function initializeApp() {
    const savedLang = localStorage.getItem('jarvis_language') || 'en';
    setLanguage(savedLang); // Set language first
    initializeSpeechRecognition(); // Then initialize STT with the correct language
    updateWaveState('idle');

    const attemptInitialGreeting = () => {
        if (window.speechSynthesis.getVoices().length > 0 && !greetingAttempted) {
            speakResponse(greetings[currentLanguage]);
            greetingAttempted = true;
        } else if (window.speechSynthesis.getVoices().length === 0 && !greetingAttempted) {
            // This means onvoiceschanged might not have fired yet or no voices.
            // Retry slightly later if onvoiceschanged doesn't pick it up.
            console.warn("Initial voices not ready, will retry greeting if onvoiceschanged fires.");
        }
    };

    if ('speechSynthesis' in window) {
        if (window.speechSynthesis.getVoices().length > 0) {
            attemptInitialGreeting();
        } else {
            window.speechSynthesis.onvoiceschanged = () => {
                // console.log("Voices loaded (onvoiceschanged). Attempting greeting.");
                attemptInitialGreeting();
            };
        }
    } else {
        console.warn("Speech synthesis not supported.");
        statusText.textContent = statusMessages.speechSynthesisNotSupported[currentLanguage];
    }
}

// Start the application
initializeApp();

</script>
</body>
</html>
"""
# Create a directory for feedback logs if it doesn't exist
FEEDBACK_DIR = os.path.join(os.path.expanduser("~"), "jarvis_feedback")
if not os.path.exists(FEEDBACK_DIR):
    os.makedirs(FEEDBACK_DIR)

@app.route("/")
def index():
    return render_template_string(HTML) # HTML variable defined in your full code

@app.route("/gui")
def gui():
    return render_template_string(GUI_HTML) # GUI_HTML variable defined in your full code

@app.route("/plan")
def plan_page():
    return render_template_string(PLAN_HTML)
    
# ... (HTML string definitions from step 1) ...

@app.route("/term-of-policy")
def term_of_policy_page():
    return render_template_string(TERM_POLICY_HTML)

@app.route("/privacy-policy")
def privacy_policy_page():
    return render_template_string(PRIVACY_POLICY_HTML)

# --- Your existing Flask routes follow ---
# @app.route("/")
# def index():
# ... and so on for /gui, /plan, /ask, /upload, etc.
    
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    messages = data.get("messages", [])
    extracted_text = data.get("extracted_text", "")
    model_preference = data.get("model_preference", "auto")
    language = data.get("language", "en") # For TTS in voice GUI
    
    # Only use extracted text if it's not empty
    if not extracted_text or not extracted_text.strip():
        extracted_text = ""
        
    # Set model preference
    if model_preference == "openrouter":
        jarvis.force_openrouter = True
    elif model_preference == "groq":
        jarvis.force_openrouter = False
        jarvis.groq_fail_count = 0  # Reset failure count
    # For 'auto', it will use the default logic in get_response
        
    # Get the actual prompt content from the last message
    user_prompt = ""
    if messages and isinstance(messages, list) and len(messages) > 0:
        last_message = messages[-1]
        if isinstance(last_message, dict) and 'content' in last_message:
            user_prompt = last_message['content']

    response = jarvis.get_response(user_prompt, extracted_text)
    
    # If the response includes TTS and a language is specified (likely from GUI call)
    # Regenerate TTS file with the specified language if the action is TTS
    if response.get('action') == 'tts' and language in ['en', 'hi']:
        # The text for TTS is usually in response['response'] from commands like /tts
        # For general AI responses to be spoken by the GUI, the GUI itself handles TTS.
        # This part is specifically if the command itself was /tts and we want to honor the GUI language.
        text_for_tts_action = response.get('response', '') 
        # Strip any "🔊 TTS for:" prefix if present
        if "🔊 TTS for:" in text_for_tts_action:
             text_for_tts_action = text_for_tts_action.split(":",1)[1].split("(Lang:")[0].replace("'","").strip()

        new_tts_file = jarvis._generate_tts_file(text_for_tts_action, language)
        if new_tts_file:
            response['tts_file'] = new_tts_file
    elif response.get('action') != 'tts' and language and response.get('response'):
        # If it's a regular response (not a /tts command) and language is provided (from GUI)
        # We can optionally generate a TTS file here if the GUI expects it.
        # The current GUI JS calls speakResponse which uses browser TTS.
        # If GUI needs a file, this is where you'd add it. For now, assume GUI handles speaking for non-/tts commands.
        pass

    return jsonify(response)

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        extracted_text = jarvis.process_uploaded_file(file)
        return jsonify({'text': extracted_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        # Use the JARVIS instance to generate the image with fallback models
        response = jarvis.generate_image(prompt)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/download-tts", methods=["GET"])
def download_tts():
    file_path_arg = request.args.get('file') # Renamed to avoid conflict with 'file' from request.files
    if not file_path_arg:
        return "File not specified", 400
    
    # Security check: Ensure the file path is relative to the tts_dir and doesn't escape it.
    # os.path.abspath will resolve '..'
    abs_tts_dir = os.path.abspath(jarvis.tts_dir)
    requested_abs_path = os.path.abspath(os.path.join(jarvis.tts_dir, os.path.basename(file_path_arg)))

    if not requested_abs_path.startswith(abs_tts_dir):
        return "Access denied: Path traversal attempt", 403
    
    if not os.path.exists(requested_abs_path): # Check the absolute path
        # Fallback check for original file_path_arg if it was already absolute (less likely for tts_file from _generate_tts_file)
        if os.path.exists(file_path_arg) and file_path_arg.startswith(abs_tts_dir):
             requested_abs_path = file_path_arg # Use original if it was already absolute and valid
        else:
            return "File not found", 404
            
    try:
        return send_file(
            requested_abs_path,
            as_attachment=True,
            download_name="tts-audio.mp3", # Keep generic download name
            mimetype="audio/mpeg"
        )
    except Exception as e:
        return str(e), 500

@app.route("/download-image", methods=["GET"])
def download_image():
    filename = request.args.get('file')
    if not filename:
        return "File not specified", 400

            
    
    # Security check for filename (e.g., prevent path traversal if filename could contain '..')
    # os.path.basename will extract just the filename part
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(jarvis.images_dir, safe_filename)
    
    # Ensure the file path is within the allowed directory (double check with abspath)
    abs_images_dir = os.path.abspath(jarvis.images_dir)
    requested_abs_path = os.path.abspath(file_path)

    if not requested_abs_path.startswith(abs_images_dir):
        return "Access denied", 403
    
    if not os.path.exists(requested_abs_path):
        return "File not found", 404
    
    try:
        return send_file(
            requested_abs_path,
            as_attachment=True,
            download_name=safe_filename, # Use safe_filename for download
            mimetype="image/png" # Assuming PNG, adjust if other types are possible
        )
    except Exception as e:
        return str(e), 500

@app.route("/report-issue", methods=["POST"])
def report_issue():
    data = request.get_json()
    try:
        # Ensure timestamp is present, default to now if not
        timestamp_str = data.get("timestamp", datetime.now().isoformat())
        # Create a filename from the timestamp to avoid issues with special chars in other fields
        # Replace colons and periods in timestamp for filename compatibility
        safe_timestamp_fn = timestamp_str.replace(":", "-").replace(".", "_")
        feedback_filename = f"feedback_{safe_timestamp_fn}.json"
        feedback_filepath = os.path.join(FEEDBACK_DIR, feedback_filename)
        
        with open(feedback_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Feedback received and saved to {feedback_filepath}") # Log to console
        # You can also print data directly: print(data)
        return jsonify({"message": "Report received and saved on server."}), 200
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return jsonify({"error": "Failed to save report on server."}), 500

# Placeholder for HTML and GUI_HTML variables - ensure they are defined above this point in your full script

if __name__ == "__main__":
    # Create directories if they don't exist (already in user's code, good)
    if not os.path.exists(jarvis.uploads_dir):
        os.makedirs(jarvis.uploads_dir)
    if not os.path.exists(jarvis.tts_dir):
        os.makedirs(jarvis.tts_dir)
    if not os.path.exists(jarvis.images_dir):
        os.makedirs(jarvis.images_dir)
    # FEEDBACK_DIR is created at the top of this "last part"
    
    app.run(debug=True)