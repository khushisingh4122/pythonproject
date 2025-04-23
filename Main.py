import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime, date
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from functools import wraps

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a secure secret key
app.config['UPLOAD_FOLDER'] = 'images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Admin credentials (in production, use a proper user management system)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

class FaceRecognitionAttendance:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """Load face encodings and names from the CSV file."""
        try:
            with open("faces.csv", mode="r", newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue  # Skip invalid rows
                    name = row[0]
                    encoding = np.fromstring(row[1], sep=',')
                    if encoding.shape[0] == 128:
                        self.known_face_names.append(name)
                        self.known_face_encodings.append(encoding)
                    else:
                        print(f"[!] Skipping invalid encoding for {name}")
            print(f"[✓] Loaded {len(self.known_face_encodings)} known face(s).")
        except FileNotFoundError:
            print("No known faces found. Add faces first.")

    def save_face(self, name, encoding):
        """Save the new face encoding to the CSV file."""
        with open("faces.csv", mode="a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, ','.join(map(str, encoding))])

    def log_attendance(self, name):
        """Log the attendance with a timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("attendance_log.csv", mode="a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, timestamp])

    def recognize_face(self, image):
        """Recognize faces in the image."""
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        logged_names = set()

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            name = "Unknown"
            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            if name not in logged_names:
                self.log_attendance(name)
                logged_names.add(name)
                print(f"Attendance logged for {name}")

    def capture_from_webcam(self):
        """Capture an image from the webcam."""
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Display the resulting image
            cv2.imshow('Webcam', frame)

            # Recognize faces and log attendance when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.recognize_face(frame)
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def capture_from_image_file(self, image_path):
        """Capture and process image from a file."""
        image = cv2.imread(image_path)
        self.recognize_face(image)

    def add_new_face(self, image, name):
        """Add a new face to the system."""
        if image is None:
            print("Error: Could not load the image. Please check if the file exists and the path is correct.")
            return
            
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            print("No face detected in the image! Please use a clear front-facing photo.")
            return
            
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if face_encodings:
            encoding = face_encodings[0]  # Assuming one face per image
            self.save_face(name, encoding)
            print(f"New face {name} added successfully.")
            self.known_face_names.append(name)
            self.known_face_encodings.append(encoding)
        else:
            print("Could not encode the face. Please use a different photo.")

    def process_image_data(self, image_data):
        """Process image data and return recognized faces"""
        face_locations = face_recognition.face_locations(image_data)
        face_encodings = face_recognition.face_encodings(image_data, face_locations)
        
        recognized_faces = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            name = "Unknown"
            if len(matches) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    self.log_attendance(name)
                    recognized_faces.append({
                        'name': name,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        return recognized_faces

    def get_all_students(self):
        """Get list of all registered students"""
        return self.known_face_names

    def remove_student(self, name):
        """Remove a student from the system"""
        try:
            # Find the index of the student
            index = self.known_face_names.index(name)
            # Remove from lists
            self.known_face_names.pop(index)
            self.known_face_encodings.pop(index)
            
            # Update the CSV file
            with open("faces.csv", mode="w", newline='') as f:
                writer = csv.writer(f)
                for i, name in enumerate(self.known_face_names):
                    writer.writerow([name, ','.join(map(str, self.known_face_encodings[i]))])
            
            return True
        except ValueError:
            return False

    def get_statistics(self):
        """Get attendance statistics"""
        total_students = len(self.known_face_names)
        today = date.today().strftime("%Y-%m-%d")
        today_attendance = 0
        total_attendance = 0
        
        try:
            with open("attendance_log.csv", mode="r", newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue
                    name, timestamp = row
                    total_attendance += 1
                    if timestamp.startswith(today):
                        today_attendance += 1
        except FileNotFoundError:
            pass
        
        return {
            "total_students": total_students,
            "today_attendance": today_attendance,
            "total_attendance": total_attendance
        }

    def get_student_attendance(self, student_name):
        """Get attendance records for a specific student"""
        attendance_records = []
        try:
            with open("attendance_log.csv", mode="r", newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue
                    name, timestamp = row
                    if name == student_name:
                        attendance_records.append({
                            "timestamp": timestamp
                        })
        except FileNotFoundError:
            pass
        
        return attendance_records

fr_system = FaceRecognitionAttendance()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    # If user is already logged in, redirect to dashboard
    if 'admin' in session:
        return redirect(url_for('admin_dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error='Invalid username or password')
    
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/admin/statistics')
@login_required
def get_statistics():
    stats = fr_system.get_statistics()
    return jsonify(stats)

@app.route('/admin/students')
@login_required
def get_students():
    students = fr_system.get_all_students()
    return jsonify({'students': [{'name': name} for name in students]})

@app.route('/admin/add_face', methods=['POST'])
@login_required
def admin_add_face():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'Image and name are required'}), 400
    
    file = request.files['image']
    name = request.form['name'].strip()
    
    if not name:
        return jsonify({'error': 'Name cannot be empty'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Check if face is detected
            face_locations = face_recognition.face_locations(rgb_image)
            if not face_locations:
                os.remove(filepath)  # Clean up the uploaded file
                return jsonify({'error': 'No face detected in the image'}), 400
            
            fr_system.add_new_face(rgb_image, name)
            return jsonify({'message': f'Successfully added face for {name}'})
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)  # Clean up on error
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/admin/remove_student', methods=['POST'])
@login_required
def remove_student():
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({'error': 'Student name is required'}), 400
    
    name = data['name']
    if fr_system.remove_student(name):
        return jsonify({'message': f'Successfully removed {name}'})
    else:
        return jsonify({'error': 'Student not found'}), 404

@app.route('/admin/student_attendance/<string:name>')
@login_required
def get_student_attendance(name):
    try:
        records = fr_system.get_student_attendance(name)
        return jsonify({"attendance": records})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    # Read image file into numpy array
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    recognized_faces = fr_system.process_image_data(rgb_image)
    
    return jsonify({
        'recognized_faces': recognized_faces
    })

if __name__ == "__main__":
    app.run(debug=True)
