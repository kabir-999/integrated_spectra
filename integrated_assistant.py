import json
import sys
import os
import uuid
import re
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO

# Voice assistant imports
from components.SpeechToText import SpeechToText
from components.LocalLLM import LocalLLM
from components.TextToSpeech import TextToSpeech
from components.SmartGlassesAudio import SmartGlassesAudio
from components.VectorDB import RAGChatbot

# Command definitions
COMMAND_KEYWORDS = {
    "play_youtube_video": ["play video", "youtube", "show video", "watch video"],
    "navigate": ["navigate", "direction", "take me to", "go to", "route to", "how to reach"],
    "play_song": ["play song", "play music", "play track", "song"],
    "play_local_song": ["play local", "local song", "play file"],
    "call": ["call", "phone", "dial"],
    "send_sms": ["send sms", "text message", "send text", "sms"],
    "send_whatsapp": ["whatsapp", "send whatsapp", "whatsapp message"],
    "toggle_flashlight": ["flashlight", "torch", "turn on light", "turn off light"],
    "open_app": ["open app", "launch app", "start app", "open"]
}

SIMPLE_COMMAND_PROMPT = """You are analyzing if text is a command.

User said: "{user_input}"

Is this a COMMAND or CONVERSATION?
- COMMAND: User wants to DO something (play, navigate, call, send, open, turn on/off)
- CONVERSATION: User is asking a question or chatting (what, who, how, why, tell me)

Rules:
- Questions starting with "what", "who", "how", "why", "when" = CONVERSATION
- "tell me", "give me", "show me information" = CONVERSATION
- "play", "navigate", "call", "open", "turn on/off" = COMMAND

Answer with ONE WORD only: COMMAND or CONVERSATION"""

CHATBOT_SYSTEM_PROMPT = """You are a helpful, friendly voice assistant. 
Keep responses concise (2-3 sentences) and conversational.
Your response will be spoken aloud."""


class ImprovedCommandDetector:
    """Rule-based + LLM hybrid command detection"""
    
    def __init__(self):
        self.command_patterns = self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns for command detection"""
        patterns = {
            "play_youtube_video": [
                r"play.*video",
                r"youtube.*video",
                r"show.*video"
            ],
            "navigate": [
                r"navigate\s+to\s+(.+)",
                r"take\s+me\s+to\s+(.+)",
                r"go\s+to\s+(.+)",
                r"directions?\s+to\s+(.+)",
                r"how\s+to\s+reach\s+(.+)",
                r"route\s+to\s+(.+)"
            ],
            "play_song": [
                r"play\s+(?:song|music|track)?\s*(.+)",
                r"play\s+me\s+(.+)"
            ],
            "play_local_song": [
                r"play\s+local\s+(?:song|music|file)?\s*(.+)"
            ],
            "call": [
                r"call\s+(.+)",
                r"phone\s+(.+)",
                r"dial\s+(.+)"
            ],
            "send_sms": [
                r"send\s+sms\s+to\s+([+\d]+)\s+saying\s+(.+)",
                r"text\s+([+\d]+)\s+saying\s+(.+)",
                r"send\s+text\s+to\s+([+\d]+)\s+(.+)"
            ],
            "send_whatsapp": [
                r"send\s+whatsapp\s+to\s+([+\d]+)\s+saying\s+(.+)",
                r"whatsapp\s+([+\d]+)\s+saying\s+(.+)"
            ],
            "toggle_flashlight": [
                r"turn\s+(on|off)\s+(?:the\s+)?(?:flash)?light",
                r"(?:flash)?light\s+(on|off)",
                r"torch\s+(on|off)"
            ],
            "open_app": [
                r"open\s+(.+?)(?:\s+app)?$",
                r"launch\s+(.+?)(?:\s+app)?$",
                r"start\s+(.+?)(?:\s+app)?$"
            ]
        }
        return patterns
    
    def detect_command_type(self, text):
        """Use keyword matching to detect command type"""
        text_lower = text.lower().strip()
        
        conversation_starters = ["what", "who", "how", "why", "when", "where", "tell me", "give me information", "explain"]
        for starter in conversation_starters:
            if text_lower.startswith(starter):
                return None, None
        
        if "?" in text and not any(kw in text_lower for kw in ["play", "navigate", "call", "open", "turn"]):
            return None, None
        
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    return command_type, match.groups() if match.groups() else None
        
        for command_type, keywords in COMMAND_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return command_type, None
        
        return None, None
    
    def extract_parameters(self, text, command_type, matched_groups):
        """Extract parameters based on command type"""
        text = text.strip()
        
        if command_type == "play_youtube_video":
            query = re.sub(r"^(play|youtube|show|watch)\s+(video)?\s*", "", text, flags=re.IGNORECASE).strip()
            return {"query": query}
        
        elif command_type == "navigate":
            if matched_groups:
                return {"destination": matched_groups[0].strip()}
            destination = re.sub(r"^(navigate|take me|go|direction|route)\s+(to)?\s*", "", text, flags=re.IGNORECASE).strip()
            return {"destination": destination}
        
        elif command_type == "play_song":
            if matched_groups:
                query = matched_groups[0].strip()
            else:
                query = re.sub(r"^play\s+(song|music|track|me)?\s*", "", text, flags=re.IGNORECASE).strip()
            return {"query": query, "platform": "ytmusic"}
        
        elif command_type == "play_local_song":
            if matched_groups:
                return {"song_name": matched_groups[0].strip()}
            song_name = re.sub(r"^play\s+local\s+(song|music|file)?\s*", "", text, flags=re.IGNORECASE).strip()
            return {"song_name": song_name}
        
        elif command_type == "call":
            if matched_groups:
                return {"phone_number": matched_groups[0].strip()}
            phone = re.sub(r"^(call|phone|dial)\s+", "", text, flags=re.IGNORECASE).strip()
            return {"phone_number": phone}
        
        elif command_type == "send_sms":
            if matched_groups and len(matched_groups) >= 2:
                return {"phone_number": matched_groups[0].strip(), "message": matched_groups[1].strip()}
            return {"phone_number": "", "message": text}
        
        elif command_type == "send_whatsapp":
            if matched_groups and len(matched_groups) >= 2:
                return {"phone_number": matched_groups[0].strip(), "message": matched_groups[1].strip()}
            return {"phone_number": "", "message": text}
        
        elif command_type == "toggle_flashlight":
            if matched_groups:
                state = matched_groups[0].lower()
            else:
                state = "on" if "on" in text.lower() else "off"
            return {"state": state}
        
        elif command_type == "open_app":
            if matched_groups:
                app_name = matched_groups[0].strip()
            else:
                app_name = re.sub(r"^(open|launch|start)\s+(app)?\s*", "", text, flags=re.IGNORECASE).strip()
            return {"app_name": app_name}
        
        return {}


class ObjectTracker:
    """Track objects and their positions to detect changes"""
    
    def __init__(self, position_threshold=100):
        self.previous_objects = {}
        self.position_threshold = position_threshold  # Pixel distance threshold
    
    def calculate_center(self, bbox):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def has_changed(self, current_objects):
        """
        Check if objects have changed significantly
        Returns: (has_changed: bool, change_description: str)
        """
        # If no previous data, it's a change
        if not self.previous_objects:
            self.previous_objects = current_objects
            return True, "initial_detection"
        
        # Check if object counts have changed
        prev_counts = {}
        curr_counts = {}
        
        for obj in self.previous_objects:
            label = obj['label']
            prev_counts[label] = prev_counts.get(label, 0) + 1
        
        for obj in current_objects:
            label = obj['label']
            curr_counts[label] = curr_counts.get(label, 0) + 1
        
        # If different object types or counts
        if prev_counts != curr_counts:
            self.previous_objects = current_objects
            return True, "object_count_changed"
        
        # Check if positions have changed significantly
        position_changed = False
        
        for curr_obj in current_objects:
            curr_label = curr_obj['label']
            curr_center = self.calculate_center(curr_obj['bbox'])
            
            # Find matching object in previous frame
            min_distance = float('inf')
            for prev_obj in self.previous_objects:
                if prev_obj['label'] == curr_label:
                    prev_center = self.calculate_center(prev_obj['bbox'])
                    distance = self.calculate_distance(curr_center, prev_center)
                    if distance < min_distance:
                        min_distance = distance
            
            # If object moved significantly
            if min_distance > self.position_threshold:
                position_changed = True
                break
        
        if position_changed:
            self.previous_objects = current_objects
            return True, "position_changed"
        
        # No significant change
        return False, "no_change"
    
    def reset(self):
        """Reset the tracker"""
        self.previous_objects = {}


class VisionRecognitionSystem:
    """BLIP-based vision recognition system"""
    
    def __init__(self, camera_id=0, tts=None):
        self.camera_id = camera_id
        self.yolo_model = None
        self.blip_processor = None
        self.blip_model = None
        self.cap = None
        self.is_running = False
        self.tts = tts
        self.glasses = SmartGlassesAudio(None) if tts else None
        self.object_tracker = ObjectTracker(position_threshold=100)
        print("🔧 Initializing Vision Recognition System...")
    
    def load_models(self):
        """Load YOLO and BLIP models"""
        try:
            print("📦 Loading YOLO model...")
            self.yolo_model = YOLO('yolov8n.pt')  # Using nano model for speed
            print("✅ YOLO model loaded")
            
            print("📦 Loading BLIP model...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.blip_model = self.blip_model.cuda()
                print("✅ BLIP model loaded (GPU)")
            else:
                print("✅ BLIP model loaded (CPU)")
            
            self.blip_model.eval()
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def speak_text(self, text):
        """Convert text to speech and play it"""
        if not self.tts or not self.glasses:
            print(f"🔊 [TTS]: {text}")
            return
        
        try:
            print(f"🔊 Speaking: {text}")
            audio_filename = f"vision_{uuid.uuid4().hex}.mp3"
            audio_file = self.tts.synthesize_speech(text, output_file=audio_filename)
            self.glasses.play_audio_to_glasses(audio_file)
            
            try:
                os.remove(audio_file)
            except Exception:
                pass
        except Exception as e:
            print(f"❌ TTS Error: {e}")
    
    def analyze_with_blip(self, frame):
        """Analyze frame using BLIP model for scene description"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Process image
            inputs = self.blip_processor(pil_image, return_tensors="pt")
            
            # Move to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                outputs = self.blip_model.generate(**inputs, max_new_tokens=50)
            
            caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            return caption if caption else "Scene analysis in progress..."
            
        except Exception as e:
            print(f"❌ Error in BLIP analysis: {e}")
            return "Scene analysis unavailable"
    
    def capture_and_caption(self):
        """Capture a single image, caption it with BLIP, and speak the caption"""
        print("\n" + "="*60)
        print("📸 IMAGE CAPTURE & CAPTIONING")
        print("="*60)
        
        # Load models if not already loaded
        if not self.blip_model:
            self.load_models()
        
        # Open camera
        print(f"\n📷 Opening camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            self.speak_text("Failed to open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Allow camera to warm up
        time.sleep(0.5)
        
        print("✅ Camera opened successfully")
        print("\n📋 Instructions:")
        print("  • Position yourself/object in frame")
        print("  • Press SPACE to capture")
        print("  • Press 'q' to return to main menu")
        print("="*60)
        print("\n⚠️  IMPORTANT: Click on the camera window to focus it!")
        
        captured = False
        window_name = 'Capture Image - Press SPACE or Q'
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        frame_count = 0
        
        try:
            while not captured:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame_count += 1
                
                # Add capture instruction overlay with blinking effect
                display_frame = frame.copy()
                
                # Blinking text effect
                if (frame_count // 15) % 2 == 0:
                    cv2.rectangle(display_frame, (30, 30), (850, 100), (0, 0, 0), -1)
                    cv2.rectangle(display_frame, (30, 30), (850, 100), (0, 255, 0), 3)
                    cv2.putText(display_frame, "Press SPACE to capture, Q to return", 
                               (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                cv2.imshow(window_name, display_frame)
                
                # Wait for key press with longer delay for better detection
                key = cv2.waitKey(30) & 0xFF
                
                if key == 32:  # Space bar (ASCII code 32)
                    print("\n📸 Capturing image...")
                    
                    # Capture and analyze
                    print("🔍 Analyzing with BLIP...")
                    caption = self.analyze_with_blip(frame)
                    
                    print(f"\n📝 Caption: {caption}")
                    print("="*60)
                    
                    # Speak the caption
                    self.speak_text(caption)
                    
                    # Save the captured image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"capture_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Image saved as: {filename}")
                    
                    captured = True
                    
                elif key == ord('q') or key == ord('Q'):
                    print("\n↩️ Returning to main menu...")
                    break
                elif key == 27:  # ESC key
                    print("\n↩️ Returning to main menu (ESC pressed)...")
                    break
        
        except Exception as e:
            print(f"\n❌ Error during capture: {str(e)}")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Ensure all windows are closed
            cv2.waitKey(1)
            
            if captured:
                print("\n✅ Image capture and captioning completed!")
            else:
                print("\n✅ Capture cancelled")
            
            # Brief pause before returning to menu
            time.sleep(1)
    
    def detect_objects(self, frame):
        """Detect objects using YOLO model"""
        try:
            results = self.yolo_model.predict(frame, conf=0.45, verbose=False)
            
            detected_objects = []
            annotated_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    label = result.names[cls]
                    
                    detected_objects.append({
                        'label': label,
                        'confidence': f"{conf:.2f}",
                        'bbox': [x1, y1, x2, y2]
                    })
                    
                    # Color coding
                    if label == 'person':
                        color = (0, 255, 255)  # Yellow
                    elif label in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                        color = (255, 0, 0)  # Blue
                    else:
                        color = (0, 255, 0)  # Green
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    text = f"{label} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(annotated_frame, text, (x1 + 3, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            return detected_objects, annotated_frame
            
        except Exception as e:
            print(f"❌ Error in object detection: {e}")
            return [], frame
    
    def realtime_object_detection(self):
        """Run real-time object detection with intelligent voice announcements"""
        print("\n" + "="*60)
        print("🎯 REAL-TIME OBJECT DETECTION (INTELLIGENT)")
        print("="*60)
        
        # Load models if not already loaded
        if not self.yolo_model:
            self.load_models()
        
        # Open camera
        print(f"\n📷 Opening camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            self.speak_text("Failed to open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✅ Camera opened successfully")
        print("\n📋 Instructions:")
        print("  • Real-time object detection active")
        print("  • Analysis every 5 seconds")
        print("  • Announces only when objects/positions change")
        print("  • Press 'q' to return to main menu")
        print("="*60)
        
        self.speak_text("Object detection started")
        
        # Reset tracker
        self.object_tracker.reset()
        
        last_analysis_time = 0
        analysis_interval = 5.0  # Analyze every 5 seconds
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Detect objects (but don't announce yet)
                detected_objects, annotated_frame = self.detect_objects(frame)
                
                # Display object count on frame
                obj_count = len(detected_objects)
                
                # Add status bar
                cv2.rectangle(annotated_frame, (10, 10), (600, 100), (0, 0, 0), -1)
                cv2.putText(annotated_frame, f"Objects Detected: {obj_count}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show next analysis countdown
                current_time = time.time()
                time_until_next = max(0, analysis_interval - (current_time - last_analysis_time))
                cv2.putText(annotated_frame, f"Next analysis: {time_until_next:.1f}s", 
                           (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Periodic analysis with change detection
                if current_time - last_analysis_time >= analysis_interval:
                    if detected_objects:
                        # Check if scene has changed
                        has_changed, change_type = self.object_tracker.has_changed(detected_objects)
                        
                        if has_changed:
                            # Count objects by type
                            object_counts = {}
                            for obj in detected_objects:
                                label = obj['label']
                                object_counts[label] = object_counts.get(label, 0) + 1
                            
                            # Create announcement text
                            announcement_parts = []
                            for label, count in object_counts.items():
                                if count == 1:
                                    announcement_parts.append(f"one {label}")
                                else:
                                    announcement_parts.append(f"{count} {label}s")
                            
                            announcement = "I see " + ", ".join(announcement_parts)
                            
                            print(f"\n🔊 Change detected ({change_type})")
                            print(f"🔊 Announcing: {announcement}")
                            
                            # Speak in background thread to avoid blocking
                            threading.Thread(target=self.speak_text, args=(announcement,), daemon=True).start()
                        else:
                            print(f"\n✓ No significant changes detected")
                    else:
                        # Check if we had objects before
                        if self.object_tracker.previous_objects:
                            print("\n🔊 Objects disappeared")
                            threading.Thread(target=self.speak_text, args=("No objects detected",), daemon=True).start()
                            self.object_tracker.reset()
                        else:
                            print("\n✓ Still no objects")
                    
                    last_analysis_time = current_time
                
                cv2.imshow('Real-time Object Detection', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n↩️ Returning to main menu...")
                    self.speak_text("Returning to main menu")
                    break
                
        except Exception as e:
            print(f"\n❌ Error during detection: {str(e)}")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            # Ensure all windows are closed
            cv2.waitKey(1)
            print("✅ Object detection stopped")
            
            # Brief pause before returning to menu
            time.sleep(1)


class CommandVoiceAssistant:
    def __init__(self, whisper_model="base", llm_model="llama3.2:1b", glasses_device=None):
        """Initialize the command-based voice assistant with conversational capabilities"""
        print("🔧 Initializing Command Voice Assistant...")
        self.stt = SpeechToText(whisper_model)
        self.llm = LocalLLM(llm_model)
        self.tts = TextToSpeech()
        self.glasses = SmartGlassesAudio(glasses_device)
        self.vectordb = RAGChatbot()
        self.detector = ImprovedCommandDetector()
        print("✅ Voice Assistant initialized successfully!")
    
    def record_and_transcribe(self, duration=5):
        """Record audio and convert to text"""
        print(f"\n🎤 Recording for {duration} seconds... Speak now!")
        
        try:
            audio_data = self.stt.record_audio(duration=duration)
            print("🔄 Converting speech to text...")
            user_input = self.stt.transcribe_realtime(audio_data)
            
            if user_input.strip():
                print(f"📝 You said: \"{user_input}\"")
                return user_input.strip()
            else:
                print("❌ No speech detected. Please try again.")
                return None
                
        except Exception as e:
            print(f"❌ Error during recording/transcription: {str(e)}")
            return None
    
    def analyze_with_llm_fallback(self, user_input):
        """Use LLM as fallback for ambiguous cases"""
        try:
            prompt = SIMPLE_COMMAND_PROMPT.format(user_input=user_input)
            response = self.llm.generate_response(prompt, max_tokens=50)
            
            response_clean = response.strip().upper()
            print(f"🔍 LLM Classification: {response_clean}")
            
            if "COMMAND" in response_clean:
                return True
            elif "CONVERSATION" in response_clean:
                return False
            else:
                return False
                
        except Exception as e:
            print(f"⚠️ LLM fallback error: {e}")
            return False
    
    def detect_command(self, user_input):
        """Main command detection logic"""
        print("\n🔍 Analyzing input...")
        
        command_type, matched_groups = self.detector.detect_command_type(user_input)
        
        if command_type:
            print(f"✅ Rule-based detection: {command_type}")
            parameters = self.detector.extract_parameters(user_input, command_type, matched_groups)
            
            return {
                "is_command": True,
                "action": command_type,
                "inputs": parameters
            }
        
        print("🤔 Ambiguous input, checking with LLM...")
        is_command = self.analyze_with_llm_fallback(user_input)
        
        if is_command:
            print("⚠️ LLM detected command but couldn't extract details")
            return {"is_command": False}
        
        print("💬 Detected as: Conversation")
        return {"is_command": False}
    
    def generate_conversational_response(self, user_input):
        """Generate a conversational response using LLM with RAG context"""
        print("\n💬 Generating conversational response...")
        
        try:
            prompt, contexts = self.vectordb.build_prompt_with_context(
                user_input, 
                CHATBOT_SYSTEM_PROMPT
            )
            
            if contexts:
                print(f"📚 Using {len(contexts)} relevant context(s) from past conversations")
            
            response = self.llm.generate_response(prompt, max_tokens=150)
            self.vectordb.store_conversation(user_input, response)
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error processing your request."
    
    def speak_response(self, text):
        """Convert text to speech and play it"""
        print("\n🔊 Converting text to speech...")
        
        try:
            audio_filename = f"response_{uuid.uuid4().hex}.mp3"
            audio_file = self.tts.synthesize_speech(text, output_file=audio_filename)
            
            print("▶️ Playing audio response...")
            self.glasses.play_audio_to_glasses(audio_file)
            
            try:
                os.remove(audio_file)
            except Exception:
                pass
                
        except Exception as e:
            print(f"❌ Error in text-to-speech: {str(e)}")
    
    def process_input(self, user_input):
        """Process user input - detect command or generate conversational response"""
        if not user_input:
            return None
        
        result = self.detect_command(user_input)
        
        print("\n" + "="*60)
        if result.get("is_command"):
            print("✅ COMMAND DETECTED")
            print(f"📋 Action: {result.get('action')}")
            print(f"📥 Inputs: {json.dumps(result.get('inputs', {}), indent=2)}")
            print("\n📄 JSON Output:")
            print(json.dumps(result, indent=2))
            print("="*60)
            
            return result
            
        else:
            print("💬 CONVERSATIONAL MODE")
            print("="*60)
            
            response = self.generate_conversational_response(user_input)
            
            print(f"\n🤖 Assistant: {response}")
            print("="*60)
            
            self.speak_response(response)
            
            return {
                "is_command": False,
                "conversation": True,
                "response": response
            }
    
    def run_voice_assistant(self):
        """Run voice assistant in loop"""
        print("\n" + "="*60)
        print("🎙️ VOICE ASSISTANT MODE")
        print("="*60)
        
        while True:
            print("\n📍 VOICE ASSISTANT MENU")
            print("-" * 40)
            print("1️⃣  Record (5 seconds)")
            print("0️⃣  Back to Main Menu")
            print("-" * 40)
            
            try:
                choice = input("\nEnter your choice: ").strip()
                
                if choice == "1":
                    user_input = self.record_and_transcribe(duration=5)
                    if user_input:
                        self.process_input(user_input)
                        
                elif choice == "0":
                    print("\n↩️ Returning to main menu...")
                    break
                    
                else:
                    print("⚠️ Invalid choice. Please enter 1 or 0.")
                    
            except KeyboardInterrupt:
                print("\n\n↩️ Returning to main menu...")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue


class IntegratedAssistant:
    """Integrated system with voice assistant and vision recognition"""
    
    def __init__(self):
        self.voice_assistant = None
        self.vision_system = None
        self.tts = None
    
    def initialize_tts(self):
        """Initialize TTS for vision system"""
        if not self.tts:
            print("🔧 Initializing Text-to-Speech...")
            self.tts = TextToSpeech()
            print("✅ TTS initialized")
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*70)
        print("🤖 INTEGRATED AI ASSISTANT SYSTEM")
        print("="*70)
        print("\n✨ Features:")
        print("  • Voice Assistant with Command Detection")
        print("  • Image Capture with BLIP Captioning + Speech")
        print("  • Real-time Object Detection with Intelligent Announcements")
        print("  • Change Detection (Position & Objects)")
        print("  • Conversational AI with Memory")
        print("="*70)
        
        while True:
            print("\n\n📍 MAIN MENU")
            print("="*70)
            print("1️⃣  Voice Assistant (Commands + Conversation)")
            print("2️⃣  Capture & Caption Image (BLIP + TTS)")
            print("3️⃣  Real-time Object Detection (YOLO + Smart Voice)")
            print("0️⃣  Exit")
            print("="*70)
            
            try:
                choice = input("\nEnter your choice: ").strip()
                
                if choice == "1":
                    print("\n🔄 Loading Voice Assistant...")
                    try:
                        if not self.voice_assistant:
                            self.voice_assistant = CommandVoiceAssistant(
                                whisper_model="base",
                                llm_model="llama3.2:1b",
                                glasses_device=None
                            )
                        self.voice_assistant.run_voice_assistant()
                    except Exception as e:
                        print(f"\n❌ Voice Assistant Error: {str(e)}")
                        print("↩️ Returning to main menu...")
                        time.sleep(1)
                    
                elif choice == "2":
                    print("\n🔄 Loading Image Capture System...")
                    try:
                        self.initialize_tts()
                        if not self.vision_system:
                            self.vision_system = VisionRecognitionSystem(
                                camera_id=0,
                                tts=self.tts
                            )
                        self.vision_system.capture_and_caption()
                    except Exception as e:
                        print(f"\n❌ Image Capture Error: {str(e)}")
                        print("↩️ Returning to main menu...")
                        # Ensure camera is released
                        cv2.destroyAllWindows()
                        time.sleep(1)
                    
                elif choice == "3":
                    print("\n🔄 Loading Object Detection System...")
                    try:
                        self.initialize_tts()
                        if not self.vision_system:
                            self.vision_system = VisionRecognitionSystem(
                                camera_id=0,
                                tts=self.tts
                            )
                        self.vision_system.realtime_object_detection()
                    except Exception as e:
                        print(f"\n❌ Object Detection Error: {str(e)}")
                        print("↩️ Returning to main menu...")
                        # Ensure camera is released
                        cv2.destroyAllWindows()
                        time.sleep(1)
                    
                elif choice == "0":
                    print("\n" + "="*70)
                    print("👋 Thank you for using the Integrated AI Assistant!")
                    print("="*70)
                    break
                    
                else:
                    print("⚠️ Invalid choice. Please enter 1, 2, 3, or 0.")
                    
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully - return to menu instead of exiting
                print("\n\n⚠️ Interrupted! Returning to main menu...")
                cv2.destroyAllWindows()
                time.sleep(1)
                continue
                
            except Exception as e:
                print(f"\n❌ Unexpected Error: {str(e)}")
                print("↩️ Returning to main menu...")
                cv2.destroyAllWindows()
                time.sleep(1)
                continue


def main():
    """Main entry point"""
    try:
        app = IntegratedAssistant()
        app.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
