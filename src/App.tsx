import React, { useState , useEffect} from 'react';
import * as tf from '@tensorflow/tfjs';

const labels = ['Cumulus', 'Altocumulus', 'Cirrus', 'Clear Sky', 'Stratocumulus', 'Cumulonimbus']; // Replace with your actual labels


// Simple icon components as fallback
const IconCloud = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>
  </svg>
);

const IconUpload = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="54" height="54" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17 8 12 3 7 8"/>
    <line x1="12" y1="3" x2="12" y2="15"/>
  </svg>
);

const IconUmbrella = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22 12a10 10 0 1 0-20 0Z"/>
    <path d="M12 12v8a2 2 0 0 0 4 0"/>
  </svg>
);

const IconSun = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="4"/>
    <path d="M12 2v2"/>
    <path d="M12 20v2"/>
    <path d="M4.93 4.93l1.41 1.41"/>
    <path d="M17.66 17.66l1.41 1.41"/>
    <path d="M2 12h2"/>
    <path d="M20 12h2"/>
    <path d="M6.34 17.66l-1.41 1.41"/>
    <path d="M19.07 4.93l-1.41 1.41"/>
  </svg>
);

const WeatherClassification = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [isDragging, setIsDragging] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [fileDetails, setFileDetails] = useState<{
    name: string;
    size: string;
    type: string;
  } | null>(null);
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [predictions, setPredictions] = useState<{ label: string; value: number; }[] | null>(null);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleFileSelect = (file: File) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setFileDetails({
        name: file.name,
        size: formatFileSize(file.size),
        type: file.type
      });
      
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
      setUploadStatus('success');
    } else {
      setSelectedFile(null);
      setFileDetails(null);
      setUploadStatus('error');
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const loadModel = async () => {
    try {
      const model = await tf.loadGraphModel('/tfjs_model/model.json');
      console.log('Model loaded:', model);
      setModel(model);
    } catch (error) {
      console.error('Error loading the model:', error);
    }
  };

  useEffect(() => {
    loadModel();
  }, []);

  const handleSubmit = async () => {
    if (!selectedFile || !model) return;

    try {
      const img = new Image();
      img.src = URL.createObjectURL(selectedFile);
      img.onload = async () => {
        const tensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([256, 256]) // Adjust size as needed
          .toFloat()
          .expandDims();
        
        const predictionTensor = model.predict(tensor) as tf.Tensor;
        console.log('Predictions:', predictionTensor);
        const predictionArray = predictionTensor.dataSync();
        console.log('Prediction Array:', predictionArray);
        const predictionResults = Array.from(predictionArray)
          .map((value, index) => ({
            label: labels[index],
            value: value < 0 ? 0 : value // Set negative values to zero
          }))
          .sort((a, b) => b.value - a.value); // Sort by value in descending order
        
        // Sum the values
        const totalValue = predictionResults.reduce((sum, prediction) => sum + prediction.value, 0);
        
        // Convert to percentages
        const predictionResultsWithPercentages = predictionResults.map(prediction => ({
          label: prediction.label,
          value: totalValue > 0 ? (prediction.value / totalValue) * 100 : 0 // Normalize to percentage
        }));
        
        console.log('Predictions:', predictionResultsWithPercentages);
        setPredictions(predictionResultsWithPercentages);
      };
    } catch (error) {
      console.error('Error processing the image:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-cyan-50 via-blue-50 to-sky-50 p-8">
      <div className="max-w-3xl mx-auto space-y-8">
        {/* Header Section */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-3">
            <div className="h-12 w-12 text-cyan-500">
              <IconUmbrella />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-500 to-blue-400 bg-clip-text text-transparent">
              CloudAI
            </h1>
          </div>
          <p className="text-lg text-gray-600">
            Intelligent Cloud Classification at Your Fingertips
          </p>
        </div>

        <div className="bg-white rounded-lg border-2 border-cyan-100 shadow-lg overflow-hidden">
          <div className="bg-gradient-to-r from-cyan-50 to-sky-50 p-6">
            <div className="flex items-center gap-2 text-2xl font-semibold text-gray-800">
              <div className="h-8 w-8 text-cyan-500">
                <IconCloud />
              </div>
              Cloud Analysis
            </div>
            <p className="text-gray-600 text-lg mt-2">
              Upload your cloud image and let <span className='font-bold bg-gradient-to-r from-cyan-500 to-blue-400 bg-clip-text text-transparent'>CloudAI</span> do the magic
            </p>
          </div>
          
          <div className="p-6">
            <div
              className={`border-3 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
                isDragging 
                  ? 'border-cyan-500 bg-cyan-50' 
                  : 'border-gray-300 hover:border-cyan-300 hover:bg-cyan-50/30'
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {!preview ? (
                <div className="space-y-6">
                  <div className="flex justify-center">
                    <div className="h-16 w-16 text-cyan-400">
                      <IconUpload />
                    </div>
                  </div>
                  <div>
                    <p className="text-gray-600 text-lg">Drag and drop your image here, or</p>
                    <label className="mt-5 inline-block cursor-pointer">
                      <span className="px-4 py-2 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 transition-colors duration-300">
                        Browse Files
                      </span>
                      <input
                        type="file"
                        className="hidden"
                        accept="image/*"
                        onChange={handleFileInput}
                      />
                    </label>
                  </div>
                  <p className="text-sm text-gray-500">Supports: JPG, PNG, GIF (Max 5MB)</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative">
                    <img
                      src={preview}
                      alt="Preview"
                      className="max-h-80 mx-auto rounded-lg shadow-md"
                    />
                    <button
                      className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition-colors duration-300"
                      onClick={() => {
                        setSelectedFile(null);
                        setPreview('');
                        setFileDetails(null);
                        setUploadStatus('idle');
                        setPredictions(null);
                      }}
                    >
                      ✕
                    </button>
                  </div>
                  {fileDetails && (
                    <div className="text-sm text-gray-600 space-y-1">
                      <p><span className="font-medium">File name:</span> {fileDetails.name}</p>
                      <p><span className="font-medium">Size:</span> {fileDetails.size}</p>
                      <p><span className="font-medium">Type:</span> {fileDetails.type}</p>
                    </div>
                  )}
                  <button
                    onClick={handleSubmit}
                    className="mt-4 px-6 py-2 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 transition-colors duration-300"
                  >
                    Analyze Cloud
                  </button>
                </div>
              )}
            </div>

            {predictions && (
              <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="text-green-800 flex flex-col gap-4 text-xl">
                  <ul>
                    {predictions.map((prediction, index) => (
                      <li key={index} className="flex justify-between">
                        <span>{prediction.label}</span>
                        <span>{prediction.value.toFixed(2)}%</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            {uploadStatus === 'success' && (
              <div className="mt-6 p-4 bg-cyan-50 border border-cyan-200 rounded-lg">
                <div className="text-cyan-800 flex items-center gap-4 text-xl justify-center">
                  <div className="h-5 w-fit">
                    <IconSun />
                  </div>
                  Image uploaded successfully!
                </div>
              </div>
            )}

            {uploadStatus === 'error' && (
              <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                <div className="text-red-800 flex items-center gap-2 text-lg">
                  <div className="h-5 w-5">
                    <IconCloud />
                  </div>
                  Please upload a valid image file.
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-gray-500 text-sm">
          © 2024 CloudAI - Advanced Cloud Classification System
        </div>
      </div>
    </div>
  );
};

export default WeatherClassification;