# check_app_status.py
import requests
import json

def check_app_status():
    print('🎉 Your Agriculture AI App is Running!')
    print('=' * 50)

    try:
        # Test the root endpoint
        r = requests.get('http://127.0.0.1:8000/')
        print(f'✅ App Status: {r.json()}')
        
        print('\n📱 Available Endpoints:')
        print('   🌐 Web Interface: http://127.0.0.1:8000/docs')
        print('   📖 Documentation: http://127.0.0.1:8000/redoc')
        print('   🔗 API Root: http://127.0.0.1:8000/')

        print('\n🧪 Test a Quick API Call:')
        # Test weather prediction
        data = {'temp': 28, 'humidity': 65, 'rainfall': 3}
        r = requests.post('http://127.0.0.1:8000/predict_weather/', json=data)
        weather_data = r.json()
        print(f'   🌤️  Weather Risk: {weather_data["risk_level"]} ({weather_data["weather_risk"]})')
        
        # Test pesticide recommendation
        data = {'crop': 'wheat', 'disease_severity': 2, 'humidity': 70, 'rainfall': 5}
        r = requests.post('http://127.0.0.1:8000/recommend_pesticide/', json=data)
        pesticide_data = r.json()
        print(f'   🌾 Natural Language: {pesticide_data["natural_language"]}')

        print('\n🎯 Your app is ready to use!')
        print('\n📋 Next Steps:')
        print('   1. Open http://127.0.0.1:8000/docs in your browser')
        print('   2. Try uploading an image for disease prediction')
        print('   3. Test weather and pesticide recommendations')
        print('   4. Use the Flutter app (if Flutter is installed)')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        print('Make sure the server is running with: py -3.12 -m uvicorn app:app --reload --host 0.0.0.0 --port 8000')

if __name__ == "__main__":
    check_app_status()

