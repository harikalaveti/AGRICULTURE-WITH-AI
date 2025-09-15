# Agri AI Assistant Flutter App

This Flutter app connects to the FastAPI backend for agriculture AI predictions.

## Setup Instructions

### 1. Install Flutter
Make sure Flutter is installed and added to your PATH:
- Download from: https://flutter.dev/docs/get-started/install
- Add Flutter to your system PATH

### 2. Get Dependencies
```bash
flutter pub get
```

### 3. Configure API URL
Edit `lib/main.dart` and update the `apiUrl` variable:

- **For Android Emulator**: `http://10.0.2.2:8000`
- **For Physical Device**: `http://YOUR_PC_IP:8000` (e.g., `http://192.168.1.20:8000`)
- **For ngrok**: `https://your-ngrok-id.ngrok.io`

### 4. Run the App
```bash
# List available devices
flutter devices

# Run on specific device
flutter run -d <device-id>

# Or run on default device
flutter run
```

## Features

- **Disease Prediction**: Upload crop images to get disease severity predictions
- **Weather Prediction**: Get weather risk assessments
- **Pesticide Recommendations**: Get personalized pesticide dose recommendations

## Backend Requirements

Make sure your FastAPI backend is running on the configured URL with these endpoints:
- `POST /predict_disease/` - Image upload for disease prediction
- `POST /predict_weather/` - Weather risk prediction
- `POST /recommend_pesticide/` - Pesticide dose recommendation

## Troubleshooting

1. **Connection Refused**: Check if the backend server is running
2. **API URL Issues**: Verify the correct IP address for your setup
3. **Image Upload Issues**: Check camera/storage permissions
