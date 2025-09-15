import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const AgriAIApp());
}

class AgriAIApp extends StatelessWidget {
  const AgriAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Agri AI Assistant',
      theme: ThemeData(primarySwatch: Colors.green),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? _image;
  final picker = ImagePicker();
  String result = "";
  bool isLoading = false;

  // API Base URL configuration
  // For Android emulator: use http://10.0.2.2:8000
  // For physical device: use your PC's IP address (e.g., http://192.168.1.20:8000)
  // For ngrok: use https://your-ngrok-id.ngrok.io
  final String apiUrl = "http://10.0.2.2:8000"; // Android emulator localhost

  // Pick image from gallery/camera
  Future pickImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        result = ""; // Clear previous result
      });
    }
  }

  // Upload image and get disease prediction
  Future predictDisease() async {
    if (_image == null) {
      setState(() {
        result = "Please select an image first!";
      });
      return;
    }
    
    setState(() {
      isLoading = true;
      result = "Analyzing image...";
    });

    try {
      var request = http.MultipartRequest(
          'POST', Uri.parse('$apiUrl/predict_disease/'));
      request.files.add(await http.MultipartFile.fromPath('file', _image!.path));
      var res = await request.send();
      var response = await http.Response.fromStream(res);
      
      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        setState(() {
          result = "Disease Prediction:\n"
              "Severity: ${data['severity_score']}\n"
              "Prediction: ${data['prediction']}\n"
              "Confidence: ${data['probabilities']}";
        });
      } else {
        setState(() {
          result = "Error: ${response.statusCode} - ${response.body}";
        });
      }
    } catch (e) {
      setState(() {
        result = "Error: $e";
      });
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  // Get weather prediction
  Future predictWeather() async {
    setState(() {
      isLoading = true;
      result = "Getting weather prediction...";
    });

    try {
      var response = await http.post(
        Uri.parse('$apiUrl/predict_weather/'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"temp": 30, "humidity": 60, "rainfall": 5}),
      );
      
      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        setState(() {
          result = "Weather Prediction:\n"
              "Risk Level: ${data['risk_level']}\n"
              "Weather Risk: ${data['weather_risk']}\n"
              "Temperature: ${data['input_data']['temperature']}°C\n"
              "Humidity: ${data['input_data']['humidity']}%";
        });
      } else {
        setState(() {
          result = "Error: ${response.statusCode} - ${response.body}";
        });
      }
    } catch (e) {
      setState(() {
        result = "Error: $e";
      });
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  // Get pesticide recommendation
  Future recommendPesticide() async {
    setState(() {
      isLoading = true;
      result = "Getting pesticide recommendation...";
    });

    try {
      var response = await http.post(
        Uri.parse('$apiUrl/recommend_pesticide/'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "crop": "wheat",
          "disease_severity": 2,
          "humidity": 70,
          "rainfall": 3
        }),
      );
      
      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        var rec = data['recommendation'];
        setState(() {
          result = "Pesticide Recommendation:\n"
              "Crop: ${data['input_data']['crop']}\n"
              "Dose: ${rec['recommended_dose_ml_per_ha']} ml/ha\n"
              "Interval: ${rec['interval_days']} days\n"
              "Severity: ${data['severity_score']}\n"
              "Weather Risk: ${data['weather_risk']}";
        });
      } else {
        setState(() {
          result = "Error: ${response.statusCode} - ${response.body}";
        });
      }
    } catch (e) {
      setState(() {
        result = "Error: $e";
      });
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Agri AI Assistant"),
        backgroundColor: Colors.green[700],
        foregroundColor: Colors.white,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const SizedBox(height: 20),
              _image == null
                  ? Container(
                      height: 200,
                      width: double.infinity,
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Center(
                        child: Text(
                          "No Image Selected",
                          style: TextStyle(fontSize: 16, color: Colors.grey),
                        ),
                      ),
                    )
                  : Container(
                      height: 200,
                      width: double.infinity,
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.file(_image!, fit: BoxFit.cover),
                      ),
                    ),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton.icon(
                    onPressed: isLoading ? null : pickImage,
                    icon: const Icon(Icons.image),
                    label: const Text("Pick Image"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                      foregroundColor: Colors.white,
                    ),
                  ),
                  ElevatedButton.icon(
                    onPressed: isLoading ? null : predictDisease,
                    icon: const Icon(Icons.analytics),
                    label: const Text("Predict Disease"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.orange,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton.icon(
                    onPressed: isLoading ? null : predictWeather,
                    icon: const Icon(Icons.wb_sunny),
                    label: const Text("Weather"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.cyan,
                      foregroundColor: Colors.white,
                    ),
                  ),
                  ElevatedButton.icon(
                    onPressed: isLoading ? null : recommendPesticide,
                    icon: const Icon(Icons.eco),
                    label: const Text("Pesticide"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              if (isLoading)
                const CircularProgressIndicator(),
              const SizedBox(height: 20),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.grey[100],
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.grey[300]!),
                ),
                child: Text(
                  result.isEmpty ? "Select an image and choose an action above" : result,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
              const SizedBox(height: 20),
              Text(
                "API URL: $apiUrl",
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey[600],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
