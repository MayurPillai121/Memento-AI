from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Generate simple description and captions
            description = "An image ready for meme creation"
            captions = [
                "When you finally deploy your app on Replit",
                "Debug mode: activated",
                "It ain't much, but it's honest work"
            ]
            
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
            return jsonify({'error': 'Missing filename or caption'}), 400
            
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(input_path):
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
                
                # Get image dimensions
                W, H = img.size
                
                # Use default font since we're in Replit
                font = ImageFont.load_default()
                font_size = 16
                
                # Wrap text
                words = caption.split()
                lines = []
                current_line = []
                
                for word in words:
                    current_line.append(word)
                    w = draw.textlength(' '.join(current_line), font=font)
                    if w > W * 0.9:
                        if len(current_line) > 1:
                            lines.append(' '.join(current_line[:-1]))
                            current_line = [word]
                        else:
                            lines.append(' '.join(current_line))
                            current_line = []
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                # Calculate text position
                total_height = len(lines) * font_size * 1.5
                y = H - total_height - 10
                
                # Draw each line
                for line in lines:
                    w = draw.textlength(line, font=font)
                    x = (W - w) / 2
                    
                    # Draw text outline
                    for adj in range(-1, 2):
                        for adj2 in range(-1, 2):
                            draw.text((x+adj, y+adj2), line, font=font, fill='black')
                    
                    # Draw text
                    draw.text((x, y), line, font=font, fill='white')
                    y += font_size * 1.5
                
                # Save the meme
                meme_filename = f"meme_{filename}"
                meme_path = os.path.join(app.config['UPLOAD_FOLDER'], meme_filename)
                meme.save(meme_path, 'JPEG', quality=95)
                
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
