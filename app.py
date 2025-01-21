from flask import Flask, render_template, request, send_file, send_from_directory, jsonify, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import tensorflow as tf
from transformers import pipeline, CLIPProcessor, CLIPModel
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import logging
import torch
import torchvision.models as models
from dotenv import load_dotenv
import re
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import tensorflow_hub as hub
import secrets

# Load environment variables
load_dotenv()

# Suppress TensorFlow and CUDA warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Initialize TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

if not client.api_key:
    raise ValueError("OpenAI API key not found in environment variables")

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.secret_key = secrets.token_hex(16)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.debug('Upload endpoint called')
        
        # Check if a file was uploaded
        if 'image' not in request.files:
            logger.error('No file part')
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            logger.error('No selected file')
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            logger.error('Invalid file type')
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
            
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.debug(f'File saved to {filepath}')
            
            # Process the image and generate description
            description = process_image(filepath)
            logger.debug(f'Generated description: {description}')
            
            # Generate captions based on the description
            captions = generate_captions(description)
            logger.debug(f'Generated captions: {captions}')
            
            return jsonify({
                'success': True,
                'filename': filename,
                'description': description,
                'captions': captions
            })
            
    except Exception as e:
        logger.error(f'Error processing upload: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/generate-meme', methods=['POST'])
def create_meme():
    try:
        data = request.get_json()
        filename = data.get('filename')
        caption = data.get('caption')
        
        if not filename or not caption:
            logger.error('Missing filename or caption')
            return jsonify({'error': 'Missing filename or caption'}), 400
            
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(input_path):
            logger.error(f'Image file not found: {input_path}')
            return jsonify({'error': 'Image file not found'}), 404
            
        # Generate meme
        try:
            # Open the image
            with Image.open(input_path) as img:
                # Convert RGBA to RGB if necessary
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # Create a copy to draw on
                meme = img.copy()
                draw = ImageDraw.Draw(meme)
                
                # Adjust font size based on image dimensions
                W, H = img.size
                font_size = min(W, H) // 15  # Dynamic font size based on image size
                max_font_size = 40  # Maximum font size cap
                font_size = min(font_size, max_font_size)  # Use the smaller value
                
                # Try multiple font paths for different OS support
                font_paths = [
                    "arial.ttf",
                    "C:/Windows/Fonts/arial.ttf",  # Windows path
                    "C:/Windows/Fonts/Arial.ttf",  # Alternative Windows path
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux path
                    "/System/Library/Fonts/Helvetica.ttc"  # Mac path
                ]
                
                font = None
                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
                
                if font is None:
                    logger.warning("Using default font as no system fonts were found")
                    font = ImageFont.load_default()
                    font_size = 16  # Smaller size for default font
                
                # Get image size
                W, H = meme.size
                
                # Wrap text
                lines = []
                words = caption.split()
                current_line = []
                
                for word in words:
                    current_line.append(word)
                    # Use getbbox() instead of textsize for better compatibility
                    bbox = draw.textbbox((0, 0), ' '.join(current_line), font=font)
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    
                    if w > W * 0.8:  # Reduced from 0.9 to 0.8 for better margins
                        if len(current_line) > 1:
                            lines.append(' '.join(current_line[:-1]))
                            current_line = [word]
                        else:
                            lines.append(' '.join(current_line))
                            current_line = []
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                # Calculate text position
                line_height = h * 1.2  # Reduced from 1.5 to 1.2 for tighter spacing
                total_text_height = line_height * len(lines)
                
                # Position text at bottom with padding
                bottom_padding = H * 0.05  # 5% padding from bottom
                y = H - total_text_height - bottom_padding  # Start from bottom and move up
                
                # Create semi-transparent background for better text readability
                # Calculate background rectangle
                bg_padding = font_size * 0.5  # Padding around text
                bg_top = y - bg_padding
                bg_bottom = H - bottom_padding + bg_padding
                bg_rect = [(0, bg_top), (W, bg_bottom)]
                
                # Draw semi-transparent black background
                overlay = Image.new('RGBA', meme.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(bg_rect, fill=(0, 0, 0, 128))  # Black with 50% transparency
                
                # Composite the overlay onto the main image
                meme = Image.alpha_composite(meme.convert('RGBA'), overlay)
                meme = meme.convert('RGB')  # Convert back to RGB for saving
                draw = ImageDraw.Draw(meme)  # Create new draw object for the composited image
                
                # Draw text with outline
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    x = (W - w) / 2  # Center text
                    
                    # Draw outline
                    outline_color = 'black'
                    outline_width = max(1, font_size // 15)  # Dynamic outline width
                    for adj in range(-outline_width, outline_width+1):
                        for adj2 in range(-outline_width, outline_width+1):
                            draw.text((x+adj, y+adj2), line, font=font, fill=outline_color)
                    
                    # Draw text
                    draw.text((x, y), line, font=font, fill='white')
                    y += line_height
                
                # Save the meme
                meme_filename = f"meme_{os.path.splitext(filename)[0]}.jpg"
                meme_path = os.path.join(app.config['UPLOAD_FOLDER'], meme_filename)
                
                # Save with high quality
                meme.save(meme_path, 'JPEG', quality=95)
                logger.info(f"Meme saved successfully at: {meme_path}")
                
                # Return the URL for the meme
                return jsonify({
                    'success': True,
                    'meme_url': f'/uploads/{meme_filename}'
                })
                
        except Exception as e:
            logger.error(f'Error generating meme: {str(e)}')
            return jsonify({'error': 'Failed to generate meme'}), 500
            
    except Exception as e:
        logger.error(f'Error in create_meme: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/regenerate', methods=['POST'])
def regenerate():
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Missing filename'}), 400
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image file not found'}), 404
            
        # Generate new description
        description = process_image(filepath)
        logger.info(f"Regenerated description: {description}")
        
        # Generate new captions based on the description
        captions = generate_captions(description)
        logger.info(f"Regenerated captions: {captions}")
        
        return jsonify({
            'success': True,
            'description': description,
            'captions': captions
        })
        
    except Exception as e:
        logger.error(f'Error regenerating content: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/auth')
def auth():
    return render_template('auth.html')

@app.route('/signin', methods=['POST'])
def signin():
    email = request.form.get('email')
    password = request.form.get('password')
    
    # TODO: Add your user authentication logic here
    # For demo, we'll just create a session
    session['user'] = {
        'email': email,
        'name': email.split('@')[0]  # Use part before @ as name
    }
    return redirect(url_for('index'))

@app.route('/signup', methods=['POST'])
def signup():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    
    if password != confirm_password:
        # TODO: Add proper error handling
        return redirect(url_for('auth'))
    
    # TODO: Add your user registration logic here
    # For demo, we'll just create a session
    session['user'] = {
        'email': email,
        'name': name
    }
    return redirect(url_for('index'))

@app.route('/signout')
def signout():
    # Clear the user session
    session.pop('user', None)
    # Flash a message (optional)
    flash('You have been signed out successfully')
    # Redirect to home page
    return redirect(url_for('index'))

def analyze_pose_and_emotion(image):
    try:
        # Pose Detection
        with mp_pose.Pose(min_detection_confidence=0.5) as pose:
            results = pose.process(image)
            pose_landmarks = results.pose_landmarks
            
            if pose_landmarks:
                # Basic pose analysis
                nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                
                # Determine if person is standing, sitting, or other pose
                pose_state = "standing"
                if nose.y > (left_shoulder.y + right_shoulder.y) / 2:
                    pose_state = "looking down"
                elif abs(left_shoulder.y - right_shoulder.y) > 0.1:
                    pose_state = "tilted"
                
                # Emotion Analysis using DeepFace
                try:
                    emotion_analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
                    if isinstance(emotion_analysis, list):
                        emotion_analysis = emotion_analysis[0]
                    dominant_emotion = emotion_analysis['dominant_emotion']
                except Exception as e:
                    logger.warning(f"Emotion detection failed: {str(e)}")
                    dominant_emotion = None
                
                return {
                    'pose_detected': True,
                    'pose_state': pose_state,
                    'emotion': dominant_emotion
                }
    except Exception as e:
        logger.warning(f"Pose/emotion analysis failed: {str(e)}")
    
    return {
        'pose_detected': False,
        'pose_state': None,
        'emotion': None
    }

def analyze_scene(image_rgb):
    try:
        # Resize image for MobileNet
        input_image = tf.image.resize(image_rgb, (224, 224))
        input_image = input_image / 255.0  # Normalize
        input_image = tf.expand_dims(input_image, 0)
        
        # Get scene predictions
        predictions = scene_model(input_image)
        scene_score = tf.nn.softmax(predictions[0])
        predicted_class = tf.argmax(scene_score)
        confidence = float(scene_score[predicted_class])
        
        return {
            'scene_detected': True,
            'scene_confidence': confidence
        }
    except Exception as e:
        logger.warning(f"Scene analysis failed: {str(e)}")
        return {
            'scene_detected': False,
            'scene_confidence': 0
        }

def process_image(filepath):
    try:
        # Load YOLOv5 model with optimized settings
        model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
        model.conf = 0.5   # Increased base confidence threshold
        model.iou = 0.45   # IOU threshold for NMS
        model.classes = None  # Detect all classes
        model.max_det = 20  # Maximum number of detections per image
        
        # Read image
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError("Failed to load image")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get predictions with augmentation
        results = model(image_rgb, augment=True)  # Enable test-time augmentation
        
        # Extract detected objects with confidence scores
        detected_objects = []
        unique_objects = {}  # Use dict to track highest confidence per class
        
        # Process detections
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            class_name = results.names[int(cls)].replace('_', ' ')
            confidence = float(conf)
            
            # Update unique_objects with highest confidence for each class
            if class_name not in unique_objects or confidence > unique_objects[class_name]['confidence']:
                unique_objects[class_name] = {
                    'confidence': confidence,
                    'box': box
                }
        
        # Convert to list and sort by confidence
        for class_name, data in unique_objects.items():
            detected_objects.append({
                'name': class_name,
                'confidence': data['confidence'],
                'box': data['box']
            })
        
        # Sort by confidence
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Analyze image properties
        height, width = image.shape[:2]
        is_dark = cv2.mean(image)[0] < 127
        is_colorful = np.std(image_rgb) > 50
        
        # Calculate size of detected objects relative to image
        for obj in detected_objects:
            box = obj['box']
            obj_width = box[2] - box[0]
            obj_height = box[3] - box[1]
            obj['size_ratio'] = (obj_width * obj_height) / (width * height)
        
        # Analyze pose and emotion
        pose_emotion = analyze_pose_and_emotion(image_rgb)
        
        # Analyze scene
        scene_info = analyze_scene(image_rgb)
        
        # Filter and categorize detections
        primary_objects = []    # Large objects with high confidence
        secondary_objects = []  # Smaller objects or lower confidence
        background_objects = [] # Very small or low confidence objects
        
        for obj in detected_objects:
            if obj['confidence'] >= 0.7 and obj['size_ratio'] > 0.1:
                primary_objects.append(obj['name'])
            elif obj['confidence'] >= 0.5 or obj['size_ratio'] > 0.05:
                secondary_objects.append(obj['name'])
            else:
                background_objects.append(obj['name'])
        
        # Build context for GPT
        context = []
        
        # Add image properties
        base_desc = "an image"
        if is_dark:
            base_desc = "a dark-toned image"
        elif is_colorful:
            base_desc = "a colorful image"
            
        if width > height * 1.5:
            base_desc += " in landscape format"
        elif height > width * 1.5:
            base_desc += " in portrait format"
        
        context.append(base_desc)
        
        # Add pose and emotion context if detected
        if pose_emotion['pose_detected']:
            pose_desc = f"A person {pose_emotion['pose_state']}"
            if pose_emotion['emotion']:
                pose_desc += f" with a {pose_emotion['emotion']} expression"
            context.append(pose_desc)
        
        # Add detected objects with context
        if primary_objects:
            context.append(f"Main focus: {', '.join(primary_objects)}")
        if secondary_objects:
            context.append(f"Also featuring: {', '.join(secondary_objects)}")
        if background_objects:
            context.append(f"In the background: {', '.join(background_objects[:3])}")  # Limit background objects
        
        # Generate description
        description = generate_gpt_description(context)
        
        logger.info(f"Image properties: {base_desc}")
        logger.info(f"Pose/Emotion: {pose_emotion}")
        logger.info(f"Scene confidence: {scene_info['scene_confidence']}")
        logger.info(f"Primary objects: {primary_objects}")
        logger.info(f"Secondary objects: {secondary_objects}")
        logger.info(f"Background objects: {background_objects[:3]}")
        logger.info(f"Generated description: {description}")
        
        return description
        
    except Exception as e:
        logger.error(f'Error in process_image: {str(e)}')
        return "An image that could make a great meme"

def clean_text(text):
    """Aggressively clean text to ensure single line without any numbering."""
    # First, split by common list separators and take first item
    text = re.split(r'\d+[\.\)]\s*|\n|;|\||\.(?=\s*\d+[\.\)])', text)[0]
    
    # Remove any leading numbers, including those in the middle of text
    text = re.sub(r'^\s*\d+[\.\)]\s*|\s+\d+[\.\)]\s+', ' ', text)
    
    # Remove any quotes or extra whitespace
    text = re.sub(r'^["\']|["\']$', '', text.strip())
    
    # Remove any "Caption:" or similar prefixes
    text = re.sub(r'^(Caption|Option|Response|Answer|Suggested caption):\s*', '', text, flags=re.IGNORECASE)
    
    # Remove any remaining numbers at start
    text = re.sub(r'^\d+[\.\)]\s*', '', text)
    
    # Final cleanup of extra whitespace
    text = ' '.join(text.split())
    
    return text

def generate_gpt_description(objects):
    try:
        messages = [
            {
                "role": "system",
                "content": """You are an expert at analyzing images and providing engaging context for meme creation. Follow these rules:
                1. Create a detailed but concise description that captures the scene, mood, and potential humor
                2. Focus on elements that could make good memes (expressions, actions, situations)
                3. Use natural, conversational language
                4. Keep it to 1-2 sentences maximum
                5. Include visual details that add character to the scene
                6. NEVER use bullet points or numbering"""
            },
            {
                "role": "user",
                "content": f"Describe this image containing: {', '.join(objects)}. Make it engaging and suitable for meme creation."
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for better descriptions
            messages=messages,
            max_tokens=150,
            temperature=0.7,  # Higher temperature for more creative descriptions
            presence_penalty=1.0,
            frequency_penalty=1.0
        )
        
        description = response.choices[0].message.content.strip()
        description = clean_text(description)
        logger.info(f"Generated description: {description}")
        
        return description
        
    except Exception as e:
        logger.error(f'Error generating description: {str(e)}')
        # Return a creative default description
        return f"A scene featuring {', '.join(objects[:3])} with potential for epic meme creation"

def generate_captions(image_description):
    try:
        messages = [
            {
                "role": "system",
                "content": """You are an expert meme caption generator known for creating viral, hilarious captions. Your output MUST follow these rules:
                1. Generate EXACTLY THREE extremely witty and funny captions
                2. Each caption should be meme-worthy and have viral potential
                3. Mix different meme styles: reaction captions, relatable moments, unexpected twists
                4. Use contemporary internet humor and meme culture references
                5. Keep each caption punchy and impactful
                6. Make sure captions are genuinely funny, not just descriptive
                7. Consider popular meme formats and trends
                8. Avoid offensive or inappropriate content
                9. Each caption should be unique in style and approach
                10. Keep each caption to one concise, impactful sentence

                Format your response as:
                1. [First caption]
                2. [Second caption]
                3. [Third caption]

                Breaking any of these rules is not allowed."""
            },
            {
                "role": "user",
                "content": "Generate THREE hilarious, meme-worthy captions that would make this image go viral. Number them 1, 2, 3."
            },
            {
                "role": "assistant",
                "content": "I will create three numbered captions that are extremely funny and perfect for memes."
            },
            {
                "role": "user",
                "content": f"Here's the image context to caption: {image_description}"
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4",  # Upgraded to GPT-4 for wittier captions
            messages=messages,
            max_tokens=200,
            temperature=0.9,  # Increased temperature for more creative and funny responses
            presence_penalty=1.0,
            frequency_penalty=1.0,
            n=1
        )
        
        caption_text = response.choices[0].message.content.strip()
        
        # Split the response into individual captions
        captions = []
        for line in caption_text.split('\n'):
            line = line.strip()
            if line and any(line.startswith(str(i) + '.') for i in range(1, 4)):
                # Remove the number and clean the caption
                caption = clean_text(line[2:].strip())
                if caption:
                    captions.append(caption)
        
        # Ensure we have exactly 3 captions
        while len(captions) < 3:
            captions.append("Caption generation error. Please try again.")
        
        logger.info(f"Generated captions: {captions}")
        return captions[:3]  # Return exactly 3 captions
        
    except Exception as e:
        logger.error(f'Error generating captions: {str(e)}')
        return [
            "Error generating captions. Please try again.",
            "Our meme lords are taking a coffee break.",
            "The humor algorithm needs a reboot."
        ]

def generate_meme(image_path, caption):
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Create a copy of the image to draw on
            meme = img.copy()
            draw = ImageDraw.Draw(meme)
            
            # Get image dimensions
            width, height = meme.size
            
            # Clean up the caption
            caption = clean_text(caption)
            
            # Start with a large font size and reduce until it fits
            font_size = 80
            font = ImageFont.truetype("arial.ttf", font_size)
            
            # Get the size of the text
            bbox = draw.textbbox((0, 0), caption, font=font)
            text_width = bbox[2] - bbox[0]
            
            # Reduce font size until the text fits within 90% of the image width
            while text_width > width * 0.9 and font_size > 20:
                font_size -= 5
                font = ImageFont.truetype("arial.ttf", font_size)
                bbox = draw.textbbox((0, 0), caption, font=font)
                text_width = bbox[2] - bbox[0]
            
            text_height = bbox[3] - bbox[1]
            
            # Calculate position to center the text
            x = (width - text_width) / 2
            y = height - text_height - 20  # 20 pixels from bottom
            
            # Draw text outline
            outline_color = 'black'
            outline_width = 2
            for adj in range(-outline_width, outline_width+1):
                for adj2 in range(-outline_width, outline_width+1):
                    draw.text((x+adj, y+adj2), caption, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((x, y), caption, font=font, fill='white')
            
            # Save the meme
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'meme.jpg')
            meme.save(output_path, 'JPEG')
            return output_path
            
    except Exception as e:
        logger.error(f'Error generating meme: {str(e)}')
        raise

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
