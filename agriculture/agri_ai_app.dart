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

  // API Base URL (replace with your FastAPI backend URL)
  final String apiUrl = "http://10.0.2.2:8000"; // Android emulator localhost

  // Pick image from gallery/camera
  Future pickImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
    }
  }

  // Upload image and get disease prediction
  Future predictDisease() async {
    if (_image == null) return;
    var request = http.MultipartRequest(
        'POST', Uri.parse('$apiUrl/predict_disease/'));
    request.files.add(await http.MultipartFile.fromPath('file', _image!.path));
    var res = await request.send();
    var response = await http.Response.fromStream(res);
    setState(() {
      result = "Disease Prediction: ${response.body}";
    });
  }

  // Get weather prediction
  Future predictWeather() async {
    var response = await http.post(
      Uri.parse('$apiUrl/predict_weather/'),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"temp": 30, "humidity": 60, "rainfall": 5}),
    );
    setState(() {
      result = "Weather Prediction: ${response.body}";
    });
  }

  // Get pesticide recommendation
  Future recommendPesticide() async {
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
    setState(() {
      result = "Pesticide Recommendation: ${response.body}";
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Agri AI Assistant")),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _image == null
                ? const Text("No Image Selected")
                : Image.file(_image!, height: 150),
            const SizedBox(height: 20),
            ElevatedButton(
                onPressed: pickImage, child: const Text("Pick Crop Image")),
            ElevatedButton(
                onPressed: predictDisease,
                child: const Text("Predict Disease")),
            ElevatedButton(
                onPressed: predictWeather,
                child: const Text("Check Weather Prediction")),
            ElevatedButton(
                onPressed: recommendPesticide,
                child: const Text("Get Pesticide Recommendation")),
            const SizedBox(height: 20),
            Text(result,
                style: const TextStyle(
                    fontSize: 16, fontWeight: FontWeight.bold)),
          ],
        ),
      ),
    );
  }
}
