"use client"

import { useState } from "react"
import {
  Activity,
  Heart,
  Thermometer,
  Wind,
  User,
  Droplets,
  TrendingUp,
  Brain,
  Pill,
  Shield,
  Stethoscope,
} from "lucide-react"

const App = () => {
  const [formData, setFormData] = useState({
    fever: 0,
    cough: 0,
    fatigue: 0,
    difficulty_breathing: 0,
    age: "",
    gender: 0,
    blood_pressure: 1,
    cholesterol_level: 1,
  })

  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const remediesDatabase = {
    "Common Cold": {
      remedies: [
        "Get plenty of rest and sleep",
        "Stay hydrated with water, herbal teas, and warm broths",
        "Use a humidifier or breathe steam from hot shower",
        "Gargle with warm salt water for sore throat",
        "Take over-the-counter pain relievers if needed",
      ],
      prevention: [
        "Wash hands frequently",
        "Avoid close contact with sick people",
        "Don't touch face with unwashed hands",
        "Maintain a healthy immune system",
      ],
    },
    Flu: {
      remedies: [
        "Rest and get plenty of sleep",
        "Drink lots of fluids to prevent dehydration",
        "Take antiviral medications if prescribed early",
        "Use fever reducers and pain relievers as needed",
        "Stay home to avoid spreading the virus",
      ],
      prevention: [
        "Get annual flu vaccination",
        "Practice good hand hygiene",
        "Avoid crowded places during flu season",
        "Maintain healthy lifestyle habits",
      ],
    },
    "COVID-19": {
      remedies: [
        "Isolate immediately and follow health guidelines",
        "Monitor symptoms and oxygen levels",
        "Stay hydrated and rest",
        "Contact healthcare provider for guidance",
        "Take prescribed medications if available",
      ],
      prevention: [
        "Get vaccinated and boosted",
        "Wear masks in crowded indoor spaces",
        "Practice social distancing",
        "Improve indoor ventilation",
      ],
    },
    Pneumonia: {
      remedies: [
        "Seek immediate medical attention",
        "Take prescribed antibiotics completely",
        "Get plenty of rest and fluids",
        "Use a humidifier for easier breathing",
        "Follow up with healthcare provider",
      ],
      prevention: [
        "Get pneumonia and flu vaccines",
        "Practice good hygiene",
        "Don't smoke and avoid secondhand smoke",
        "Maintain a healthy immune system",
      ],
    },
    Bronchitis: {
      remedies: [
        "Rest and avoid strenuous activities",
        "Drink warm liquids to soothe throat",
        "Use a humidifier or inhale steam",
        "Take cough suppressants if recommended",
        "Avoid smoke and air pollutants",
      ],
      prevention: [
        "Don't smoke and avoid secondhand smoke",
        "Get annual flu vaccine",
        "Wash hands frequently",
        "Wear mask in polluted environments",
      ],
    },
    Healthy: {
      remedies: [
        "Continue maintaining healthy lifestyle",
        "Regular exercise and balanced diet",
        "Adequate sleep and stress management",
        "Regular health check-ups",
        "Stay hydrated and practice good hygiene",
      ],
      prevention: [
        "Maintain current healthy habits",
        "Stay up to date with vaccinations",
        "Regular medical screenings",
        "Practice preventive care",
      ],
    },
  }

  const handleInputChange = (name, value) => {
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }))
    // Clear previous results when form changes
    if (prediction) setPrediction(null)
    if (error) setError(null)
  }

  const handleSubmit = async () => {
    if (!formData.age) {
      setError("Please enter your age")
      return
    }

    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      // Simulate API call delay
      await new Promise((resolve) => setTimeout(resolve, 2000))

      // Mock prediction based on symptoms
      const mockPrediction = {
        "Random Forest": {
          disease: formData.fever || formData.cough ? "Common Cold" : "Healthy",
          confidence: 0.85,
        },
        SVM: {
          disease: formData.difficulty_breathing ? "Bronchitis" : "Healthy",
          confidence: 0.78,
        },
        "Neural Network": {
          disease: formData.fatigue && formData.fever ? "Flu" : "Healthy",
          confidence: 0.82,
        },
      }

      setPrediction(mockPrediction)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setFormData({
      fever: 0,
      cough: 0,
      fatigue: 0,
      difficulty_breathing: 0,
      age: "",
      gender: 0,
      blood_pressure: 1,
      cholesterol_level: 1,
    })
    setPrediction(null)
    setError(null)
  }

  const SymptomCard = ({ icon: Icon, title, name, value, onChange }) => (
    <div className="backdrop-blur-md bg-white/80 rounded-xl p-4 border border-gray-300/50 hover:border-gray-400/70 transition-all duration-300 hover:shadow-lg hover:shadow-gray-300/20 group">
      <div className="flex items-center gap-2 mb-3">
        <div className="p-1.5 rounded-lg bg-gray-100/90 backdrop-blur-sm">
          <Icon className={`w-4 h-4 ${value ? "text-gray-700" : "text-gray-500"} transition-colors duration-300`} />
        </div>
        <span className="font-medium text-gray-800 text-sm">{title}</span>
      </div>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => onChange(name, 0)}
          className={`flex-1 px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
            value === 0
              ? "bg-gray-600 text-white shadow-md shadow-gray-600/30"
              : "bg-gray-100/70 text-gray-700 hover:bg-gray-200/70 backdrop-blur-sm"
          }`}
        >
          No
        </button>
        <button
          type="button"
          onClick={() => onChange(name, 1)}
          className={`flex-1 px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
            value === 1
              ? "bg-gray-700 text-white shadow-md shadow-gray-700/30"
              : "bg-gray-100/70 text-gray-700 hover:bg-gray-200/70 backdrop-blur-sm"
          }`}
        >
          Yes
        </button>
      </div>
    </div>
  )

  const SelectCard = ({ icon: Icon, title, name, value, options, onChange }) => (
    <div className="backdrop-blur-md bg-white/80 rounded-xl p-4 border border-gray-300/50 hover:border-gray-400/70 transition-all duration-300 hover:shadow-lg hover:shadow-gray-300/20">
      <div className="flex items-center gap-2 mb-3">
        <div className="p-1.5 rounded-lg bg-gray-100/90 backdrop-blur-sm">
          <Icon className="w-4 h-4 text-gray-700" />
        </div>
        <span className="font-medium text-gray-800 text-sm">{title}</span>
      </div>
      <select
        value={value}
        onChange={(e) => onChange(name, Number.parseInt(e.target.value))}
        className="w-full p-2.5 bg-gray-50/90 backdrop-blur-sm border border-gray-300/50 rounded-lg focus:ring-2 focus:ring-gray-500/50 focus:border-gray-500/50 text-gray-800 font-medium placeholder-gray-500 text-sm transition-all duration-300"
      >
        {options.map((option, index) => (
          <option key={index} value={index} className="bg-white text-gray-800">
            {option}
          </option>
        ))}
      </select>
    </div>
  )

  const PredictionCard = ({ modelName, result }) => (
    <div className="backdrop-blur-md bg-white/85 rounded-xl p-4 border border-gray-300/50 hover:shadow-lg hover:shadow-gray-300/20 transition-all duration-300 group">
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium text-gray-800 text-sm">{modelName}</span>
        <div className="p-1.5 rounded-lg bg-gray-100/90 backdrop-blur-sm group-hover:scale-110 transition-transform duration-300">
          <Brain className="w-4 h-4 text-gray-700" />
        </div>
      </div>
      <div className="text-lg font-medium text-gray-900 mb-2">{result.disease}</div>
      {result.confidence && (
        <div className="flex items-center gap-2">
          <div className="flex-1 bg-gray-200/70 rounded-full h-1.5 backdrop-blur-sm">
            <div
              className="bg-gray-600 h-1.5 rounded-full transition-all duration-1000 ease-out"
              style={{ width: `${result.confidence * 100}%` }}
            ></div>
          </div>
          <span className="text-xs font-medium text-gray-700">{(result.confidence * 100).toFixed(1)}%</span>
        </div>
      )}
    </div>
  )

  const RemediesSection = ({ predictions }) => {
    const diseases = Object.values(predictions).map((p) => p.disease)
    const uniqueDiseases = [...new Set(diseases)]

    return (
      <div className="mt-6 backdrop-blur-md bg-white/75 rounded-2xl p-6 border border-gray-300/50 shadow-lg">
        <h2 className="text-2xl font-medium text-gray-900 mb-6 flex items-center gap-2">
          <div className="p-2 rounded-xl bg-gray-100/90 backdrop-blur-sm">
            <Pill className="w-6 h-6 text-gray-700" />
          </div>
          Recommended Care & Prevention
        </h2>

        {uniqueDiseases.map((disease, index) => {
          const remedyData = remediesDatabase[disease] || remediesDatabase["Healthy"]
          return (
            <div key={index} className="mb-6 last:mb-0">
              <h3 className="text-xl font-medium text-gray-800 mb-4 flex items-center gap-2">
                <Stethoscope className="w-5 h-5 text-gray-700" />
                {disease}
              </h3>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="backdrop-blur-sm bg-gray-50/90 rounded-xl p-4 border border-gray-200/50">
                  <h4 className="text-base font-medium text-gray-800 mb-3 flex items-center gap-2">
                    <Pill className="w-4 h-4" />
                    Treatment & Care
                  </h4>
                  <ul className="space-y-2">
                    {remedyData.remedies.map((remedy, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-gray-700 text-sm">
                        <div className="w-1.5 h-1.5 rounded-full bg-gray-600 mt-1.5 flex-shrink-0"></div>
                        <span className="font-medium">{remedy}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="backdrop-blur-sm bg-gray-50/90 rounded-xl p-4 border border-gray-200/50">
                  <h4 className="text-base font-medium text-gray-800 mb-3 flex items-center gap-2">
                    <Shield className="w-4 h-4" />
                    Prevention Tips
                  </h4>
                  <ul className="space-y-2">
                    {remedyData.prevention.map((tip, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-gray-700 text-sm">
                        <div className="w-1.5 h-1.5 rounded-full bg-gray-600 mt-1.5 flex-shrink-0"></div>
                        <span className="font-medium">{tip}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )
        })}

        <div className="mt-6 p-4 backdrop-blur-sm bg-gray-100/70 border border-gray-200/50 rounded-xl">
          <p className="text-gray-800 font-medium flex items-start gap-2 text-sm">
            <div className="w-5 h-5 rounded-full bg-gray-600 flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-white text-xs font-medium">!</span>
            </div>
            <span>
              <strong>Medical Disclaimer:</strong> This AI prediction system is for educational and informational
              purposes only. These recommendations are general guidelines and should not replace professional medical
              advice. Always consult with qualified healthcare professionals for proper diagnosis and treatment.
            </span>
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#FAFAFA] via-[#D1D1D1] to-[#929292] relative overflow-hidden">
      <div className="absolute inset-0">
        {/* Geometric shapes for visual interest */}
        <div className="absolute top-20 left-20 w-32 h-32 bg-gradient-to-br from-white/40 to-[#D1D1D1]/30 rounded-full blur-xl"></div>
        <div className="absolute bottom-32 right-32 w-48 h-48 bg-gradient-to-tl from-[#929292]/20 to-white/30 rounded-full blur-2xl"></div>
        <div className="absolute top-1/2 left-1/4 w-24 h-24 bg-gradient-to-r from-[#D1D1D1]/25 to-[#929292]/20 rounded-full blur-lg"></div>

        {/* Subtle dot pattern */}
        <div className="absolute top-10 left-10 w-2 h-2 bg-[#929292]/40 rounded-full"></div>
        <div className="absolute top-20 right-20 w-1 h-1 bg-[#222222]/30 rounded-full"></div>
        <div className="absolute bottom-20 left-20 w-3 h-3 bg-[#D1D1D1]/50 rounded-full"></div>
        <div className="absolute bottom-40 right-40 w-1 h-1 bg-[#929292]/35 rounded-full"></div>
        <div className="absolute top-1/3 left-1/4 w-2 h-2 bg-[#222222]/25 rounded-full"></div>
        <div className="absolute top-2/3 right-1/3 w-1 h-1 bg-[#929292]/30 rounded-full"></div>
        <div className="absolute top-1/4 right-1/4 w-1 h-1 bg-[#D1D1D1]/40 rounded-full"></div>
        <div className="absolute bottom-1/3 left-1/3 w-2 h-2 bg-[#929292]/20 rounded-full"></div>
      </div>

      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-white/30 via-transparent to-[#929292]/20"></div>
      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-tl from-[#D1D1D1]/10 via-transparent to-white/20"></div>

      <div className="relative container mx-auto px-4 py-8 max-w-6xl">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 rounded-2xl bg-white/90 backdrop-blur-md border border-gray-300/50 shadow-lg">
              <Activity className="w-10 h-10 text-gray-700" />
            </div>
            <h1 className="text-4xl md:text-5xl font-medium text-gray-900">AI Health Predictor</h1>
          </div>
          <p className="text-gray-700 text-lg font-medium max-w-2xl mx-auto leading-relaxed">
            Advanced AI-powered health assessment system with personalized care recommendations
          </p>
        </div>

        <div className="space-y-6">
          <div className="backdrop-blur-md bg-white/70 rounded-2xl p-6 border border-gray-300/40 shadow-lg">
            <h2 className="text-2xl font-medium text-gray-900 mb-6 flex items-center gap-2">
              <div className="p-2 rounded-xl bg-gray-100/90 backdrop-blur-sm">
                <Thermometer className="w-6 h-6 text-gray-700" />
              </div>
              Current Symptoms
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <SymptomCard
                icon={Thermometer}
                title="Fever"
                name="fever"
                value={formData.fever}
                onChange={handleInputChange}
              />
              <SymptomCard icon={Wind} title="Cough" name="cough" value={formData.cough} onChange={handleInputChange} />
              <SymptomCard
                icon={Activity}
                title="Fatigue"
                name="fatigue"
                value={formData.fatigue}
                onChange={handleInputChange}
              />
              <SymptomCard
                icon={Wind}
                title="Difficulty Breathing"
                name="difficulty_breathing"
                value={formData.difficulty_breathing}
                onChange={handleInputChange}
              />
            </div>
          </div>

          <div className="backdrop-blur-md bg-white/70 rounded-2xl p-6 border border-gray-300/40 shadow-lg">
            <h2 className="text-2xl font-medium text-gray-900 mb-6 flex items-center gap-2">
              <div className="p-2 rounded-xl bg-gray-100/90 backdrop-blur-sm">
                <User className="w-6 h-6 text-gray-700" />
              </div>
              Patient Profile
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="backdrop-blur-md bg-white/80 rounded-xl p-4 border border-gray-300/50 hover:border-gray-400/70 transition-all duration-300">
                <div className="flex items-center gap-2 mb-3">
                  <div className="p-1.5 rounded-lg bg-gray-100/90 backdrop-blur-sm">
                    <User className="w-4 h-4 text-gray-700" />
                  </div>
                  <span className="font-medium text-gray-800 text-sm">Age</span>
                </div>
                <input
                  type="number"
                  value={formData.age}
                  onChange={(e) => handleInputChange("age", e.target.value)}
                  placeholder="Enter your age"
                  min="0"
                  max="150"
                  className="w-full p-2.5 bg-gray-50/90 backdrop-blur-sm border border-gray-300/50 rounded-lg focus:ring-2 focus:ring-gray-500/50 focus:border-gray-500/50 text-gray-800 font-medium placeholder-gray-500 text-sm transition-all duration-300"
                />
              </div>

              <SelectCard
                icon={User}
                title="Gender"
                name="gender"
                value={formData.gender}
                options={["Female", "Male"]}
                onChange={handleInputChange}
              />

              <SelectCard
                icon={Heart}
                title="Blood Pressure"
                name="blood_pressure"
                value={formData.blood_pressure}
                options={["Low", "Normal", "High"]}
                onChange={handleInputChange}
              />

              <SelectCard
                icon={Droplets}
                title="Cholesterol Level"
                name="cholesterol_level"
                value={formData.cholesterol_level}
                options={["Low", "Normal", "High"]}
                onChange={handleInputChange}
              />
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <button
              onClick={handleSubmit}
              disabled={loading || !formData.age}
              className="group px-8 py-3 bg-gray-700 text-white font-medium rounded-xl shadow-lg shadow-gray-700/30 hover:shadow-gray-700/50 transform hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center gap-2 backdrop-blur-sm border border-gray-600/20"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Analyzing Health Data...</span>
                </>
              ) : (
                <>
                  <TrendingUp className="w-4 h-4 group-hover:rotate-12 transition-transform duration-300" />
                  <span>Predict & Analyze</span>
                </>
              )}
            </button>

            <button
              onClick={resetForm}
              className="px-8 py-3 bg-gray-600 text-white font-medium rounded-xl shadow-lg shadow-gray-600/30 hover:shadow-gray-600/50 hover:bg-gray-500 transform hover:scale-105 transition-all duration-300 backdrop-blur-sm border border-gray-500/20"
            >
              <span>Reset Form</span>
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-6 backdrop-blur-md bg-red-50/80 border border-red-200/50 rounded-xl p-4 shadow-lg">
            <div className="flex items-center gap-2 text-red-700">
              <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center">
                <span className="text-white text-xs font-medium">!</span>
              </div>
              <span className="font-medium">Error:</span>
              <span className="font-medium">{error}</span>
            </div>
          </div>
        )}

        {prediction && (
          <div className="mt-6 backdrop-blur-md bg-white/70 rounded-2xl p-6 border border-gray-300/40 shadow-lg">
            <h2 className="text-2xl font-medium text-gray-900 mb-6 flex items-center gap-2">
              <div className="p-2 rounded-xl bg-gray-100/90 backdrop-blur-sm">
                <Brain className="w-6 h-6 text-gray-700" />
              </div>
              AI Analysis Results
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
              {Object.entries(prediction).map(([modelName, result]) => (
                <PredictionCard key={modelName} modelName={modelName} result={result} />
              ))}
            </div>
          </div>
        )}

        {prediction && <RemediesSection predictions={prediction} />}
      </div>
    </div>
  )
}

export default App
