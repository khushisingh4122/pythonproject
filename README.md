# Face Recognition Attendance System

This project uses face recognition to mark attendance from a webcam or image. It stores face encodings and logs attendance into CSV files.

## Features

- Add new faces from image files
- Capture faces from webcam and mark attendance
- Log attendance with timestamp
- Save face encodings for future use

## How to Use

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the program:**
   ```bash
   python Main.py
   ```

3. **Choose an option:**
   - Press `1` to use webcam for attendance
   - Press `2` to recognize face from an image
   - Press `3` to add a new face

## File Structure

- `faces.csv` – stores known face encodings and names
- `attendance_log.csv` – logs attendance records
- `Main.py` – main Python script
- `requirements.txt` – required Python libraries

## Libraries Used

- face_recognition
- OpenCV (cv2)
- NumPy
- CSV
- datetime

## Notes

- Make sure your camera is accessible.
- Save clear, front-facing images for best results.

## Author

khushi singh,Yagya Bhardwaj, Utkarsh Sharma, Umair Elahi

B.Tech CSE (AI & ML) — COER University
