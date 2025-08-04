"use client"; // This is a client component, as it uses browser-only APIs and hooks

import { useState, ChangeEvent } from 'react';
import Image from "next/image";

interface PredictionResult {
  predicted_class: string;
  confidence: number;
}

export default function Home() {
  // State for the selected file and its preview URL
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  
  // State for the prediction result, loading status, and errors
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handle file selection from the input
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      // Reset previous prediction and error
      setPrediction(null);
      setError(null);
    }
  };

  // Handle the prediction process
  const handlePrediction = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Send the image to the backend API
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction request failed');
      }

      const result = await response.json();
      setPrediction(result);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-12 bg-gray-50">
      <div className="w-full max-w-xl bg-white rounded-lg shadow-xl p-8">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-6">
          🌿 Leaf Disease Detection
        </h1>
        
        {/* File Input */}
        <div className="mb-6">
          <label htmlFor="file-upload" className="cursor-pointer bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition-colors duration-300 w-full text-center inline-block">
            Choose an Image
          </label>
          <input id="file-upload" type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
        </div>

        {/* Image Preview */}
        {previewUrl && (
          <div className="mb-6 text-center">
            <h2 className="text-xl font-semibold text-gray-700 mb-2">Image Preview</h2>
            <Image 
              src={previewUrl} 
              alt="Image preview" 
              width={240} 
              height={240} 
              className="mx-auto rounded-lg shadow-md h-auto w-auto max-h-60" 
              />
          </div>
        )}

        {/* Predict Button */}
        {file && (
          <div className="text-center mb-6">
            <button
              onClick={handlePrediction}
              disabled={isLoading}
              className="bg-green-500 hover:bg-green-600 text-white font-bold py-3 px-8 rounded-lg transition-colors duration-300 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Predicting...' : 'Predict Disease'}
            </button>
          </div>
        )}

        {/* Results Display */}
        {error && <p className="text-red-500 text-center font-semibold">{`Error: ${error}`}</p>}
        {prediction && (
          <div className="text-center bg-gray-100 p-6 rounded-lg">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Prediction Result</h2>
            <p className="text-lg text-indigo-600 font-semibold">
              {prediction.predicted_class.replace(/_/g, ' ')}
            </p>
            <p className="text-md text-gray-600 mt-1">
              Confidence: <span className="font-bold">{(prediction.confidence * 100).toFixed(2)}%</span>
            </p>
          </div>
        )}
      </div>
    </main>
  );
}