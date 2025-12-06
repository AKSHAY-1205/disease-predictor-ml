"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import {
  Activity,
  Brain,
  Droplets,
  Heart,
  Pill,
  Stethoscope,
  Thermometer,
  User,
  Wind,
  Shield,
  Globe,
} from "lucide-react"

// Keep a simple brand color system for clarity: primary teal + neutrals + a warm accent
const BRAND = {
  primary: "teal", // Tailwind teal scale
}

const translations = {
  en: {
    appTitle: "AI Health Predictor",
    appSubtitle: "Random Forest-powered health assessment with personalized care recommendations",
    languagePromptTitle: "Choose your language",
    english: "English",
    tamil: "தமிழ்",
    currentSymptoms: "Current Symptoms",
    fever: "Fever",
    cough: "Cough",
    fatigue: "Fatigue",
    difficultyBreathing: "Difficulty Breathing",
    patientProfile: "Patient Profile",
    age: "Age",
    gender: "Gender",
    female: "Female",
    male: "Male",
    bloodPressure: "Blood Pressure",
    low: "Low",
    normal: "Normal",
    high: "High",
    cholesterolLevel: "Cholesterol Level",
    predictAnalyze: "Predict & Analyze",
    analyzing: "Analyzing...",
    resetForm: "Reset Form",
    aiResults: "AI Analysis Result",
    rfModel: "Random Forest",
    confidence: "Confidence",
    carePrevention: "Recommended Care & Prevention",
    treatmentCare: "Treatment & Care",
    preventionTips: "Prevention Tips",
    disclaimerTitle: "Medical Disclaimer:",
    disclaimerBody:
      "This AI prediction is for educational purposes only and must not replace professional medical advice. Always consult qualified healthcare providers for diagnosis and treatment.",
    errorPrefix: "Error:",
    ageRequired: "Please enter your age.",
    invalidResponse: "Unexpected response from server.",
    yes: "Yes",
    no: "No",
  },
  ta: {
    appTitle: "ஏஐ சுகாதார கணிப்ப��",
    appSubtitle: "ரேண்டம் ஃபாரஸ்ட் அடிப்படை சுகாதார மதிப்பீடு மற்றும் தனிப்பயன் பரிந்துரைகள்",
    languagePromptTitle: "மொழியைத் தேர்ந்தெடுக்கவும்",
    english: "English",
    tamil: "தமிழ்",
    currentSymptoms: "தற்போதைய அறிகுறிகள்",
    fever: "காய்ச்சல்",
    cough: "இருமல்",
    fatigue: "சோர்வு",
    difficultyBreathing: "சுவாசத்தில் சிரமம்",
    patientProfile: "நோயாளி விவரம்",
    age: "வயது",
    gender: "பாலினம்",
    female: "பெண்",
    male: "ஆண்",
    bloodPressure: "ரத்த அழுத்தம்",
    low: "குறைவு",
    normal: "சாதாரணம்",
    high: "உயர்",
    cholesterolLevel: "கொழுப்பு அளவு",
    predictAnalyze: "கணித்து பகுப்பாய்வு செய்",
    analyzing: "பகுப்பாய்வு செய்கிறது...",
    resetForm: "மீட்டமை",
    aiResults: "ஏஐ பகுப்பாய்வு முடிவு",
    rfModel: "ரேண்டம் ஃபாரஸ்ட்",
    confidence: "நம்பிக்கை",
    carePrevention: "சிகிச்சை மற்றும் தடுப்பு",
    treatmentCare: "சிகிச்சை",
    preventionTips: "தடுப்பு குறிப்புகள்",
    disclaimerTitle: "மருத்துவ அறிவிப்பு:",
    disclaimerBody: "இந்த ஏஐ கணிப்பு கல்வி நோக்கத்திற்காக மட்டுமே. சரியான மருத்துவர்களை அணுகவும்.",
    errorPrefix: "பிழை:",
    ageRequired: "தயவுசெய்து உங்கள் வயதை உள்ளிடவும்.",
    invalidResponse: "சர்வரிலிருந்து எதிர்பாராத பதில்.",
    yes: "ஆம்",
    no: "இல்லை",
  },
}

const diseaseDisplayMap = {
  en: {
    "Common Cold": "Common Cold",
    Flu: "Flu",
    "COVID-19": "COVID-19",
    Pneumonia: "Pneumonia",
    Bronchitis: "Bronchitis",
    Healthy: "Healthy",
  },
  ta: {
    "Common Cold": "சாதாரண குளிர்",
    Flu: "காய்ச்சல்",
    "COVID-19": "கோவிட்-19",
    Pneumonia: "நிமோனியா",
    Bronchitis: "பிராங்கைட்டிஸ்",
    Healthy: "நலமாக உள்ளீர்கள்",
  },
}

const remediesDB = {
  en: {
    "Common Cold": {
      remedies: [
        "Rest and sleep well",
        "Drink warm fluids and stay hydrated",
        "Consider steam inhalation",
        "Warm salt-water gargle for sore throat",
        "Over-the-counter pain relievers if needed",
      ],
      prevention: [
        "Frequent hand washing",
        "Avoid close contact with sick individuals",
        "Avoid touching face with unwashed hands",
        "Maintain good immunity",
      ],
    },
    Flu: {
      remedies: [
        "Plenty of rest",
        "Hydration to prevent dehydration",
        "Antiviral meds if prescribed",
        "Fever reducers if needed",
        "Stay home to limit spread",
      ],
      prevention: ["Annual flu vaccination", "Hand hygiene", "Avoid crowded places in flu season", "Healthy lifestyle"],
    },
    "COVID-19": {
      remedies: [
        "Isolate and follow local guidelines",
        "Monitor symptoms and oxygen levels",
        "Hydrate and rest",
        "Consult healthcare provider",
        "Take prescribed medication if available",
      ],
      prevention: [
        "Vaccination and boosters",
        "Mask in crowded indoor spaces",
        "Social distancing",
        "Improve ventilation",
      ],
    },
    Pneumonia: {
      remedies: [
        "Seek medical attention promptly",
        "Complete prescribed antibiotics",
        "Rest and fluids",
        "Use humidifier if needed",
        "Follow up with provider",
      ],
      prevention: ["Pneumonia and flu vaccines", "Hand hygiene", "Avoid smoking", "Strong immune health"],
    },
    Bronchitis: {
      remedies: [
        "Avoid strenuous activity",
        "Warm liquids to soothe throat",
        "Humidifier/steam inhalation",
        "Cough suppressants if advised",
        "Avoid smoke and pollutants",
      ],
      prevention: [
        "Do not smoke; avoid secondhand smoke",
        "Annual flu vaccine",
        "Hand washing",
        "Mask in polluted environments",
      ],
    },
    Healthy: {
      remedies: [
        "Maintain current healthy habits",
        "Regular exercise and balanced diet",
        "Adequate sleep and stress management",
        "Routine health checkups",
        "Good hydration",
      ],
      prevention: ["Continue healthy routines", "Stay up to date on vaccines", "Regular screenings", "Preventive care"],
    },
  },
  ta: {
    "Common Cold": {
      remedies: [
        "மிகவும் ஓய்வு எடுக்கவும்",
        "சூடான திரவங்களை குடிக்கவும்",
        "வேப்பம்/நீராவி சுவாசம் உதவலாம்",
        "உப்பு நீரால் கொப்பளி செய்யவும்",
        "தேவைப்பட்டால் வலி தணிக்கும் மருந்துகள்",
      ],
      prevention: [
        "கைகளை அடிக்கடி கழுவவும்",
        "உடல்நலம் குன்றியவர்களுடன் நெருக்கம் தவிர்க்கவும்",
        "கழுவாத கைகளால் முகத்தைத் தொட வேண்டாம்",
        "இம்யூனிட்டியை பராமரிக்கவும்",
      ],
    },
    Flu: {
      remedies: [
        "பல நேரம் ஓய்வு",
        "உடல் நீரிழப்பு தவிர்க்க திரவங்கள்",
        "மருத்துவர் பரிந்துரைத்தால் வைரஸ் எதிர்ப்பு மருந்துகள்",
        "தேவைப்பட்டால் காய்ச்சல் தணிக்கைகள்",
        "பரவலைத் தடுக்க வீட்டில் இருக்கவும்",
      ],
      prevention: [
        "ஆண்டு தோறும் தோறும் காய்ச்சல் தடுப்பூசி",
        "கைகளின் தூய்மை",
        "கூட்டம் அதிகமான இடங்களைத் தவிர்க்கவும்",
        "உடல்நல பழக்க வழக்கங்கள்",
      ],
    },
    "COVID-19": {
      remedies: [
        "தனிமைப்படுத்தி வழிகாட்டுதல்களைப் பின்பற்றவும்",
        "அறிகுறி/ஆக்சிஜன் அளவு கண்காணிக்கவும்",
        "நீர் குடித்து ஓய்வு",
        "மருத்துவரை அணுகவும்",
        "மருந்துகள் உத்தரவுப்படி எடுத்துக்கொள்ளவும்",
      ],
      prevention: [
        "தடுப்பூசி மற்றும் பூஸ்டர் டோஸ்",
        "நெரிசலான இடங்களில் முகக்கவசம்",
        "சமூக இடைவெளி",
        "வெளிச்சம்/வெண்டிலேஷன் மேம்படுத்தவும்",
      ],
    },
    Pneumonia: {
      remedies: [
        "உடனடி மருத்துவ ஆலோசனை பெறவும்",
        "மருந்துகளை முழுமையாக எடுத்துக்கொள்ளவும்",
        "ஓய்வு மற்றும் திரவங்கள்",
        "தேவையெனில் ஹ்யூமிடிஃபையர்",
        "மருத்துவர் பின்பற்றுதல்",
      ],
      prevention: [
        "நிமோனியா மற்றும் காய்ச்சல் தடுப்பூசி",
        "கைகள் சுத்தமாக வைத்துக்கொள்ளவும்",
        "புகைபிடித்தலைத் தவிர்க்கவும்",
        "இம்யூனிட்டி பராமரிக்கவும்",
      ],
    },
    Bronchitis: {
      remedies: [
        "கடுமையான செயல்கள் தவிர்க்கவும்",
        "தொண்டை நிவாரணம் தர சூடான பானங்கள்",
        "ஹ்யூமிடிஃபையர்/நீராவி",
        "மருத்துவர் கூறினால் இருமல் தணிக்கைகள்",
        "புகை/மாசு தவிர்க்கவும்",
      ],
      prevention: [
        "புகைபிடிக்க வேண்டாம்; பக்கவிளைவு புகையைத் தவிர்க்கவும்",
        "ஆண்டு தோறும் தோறும் காய்ச்சல் தடுப்பூசி",
        "கைகளைச் சுத்தமாக வைத்துக்கொள்ளவும்",
        "மாசு அதிகம் உள்ள இடங்களில் முகக்கவசம்",
      ],
    },
    Healthy: {
      remedies: [
        "தற்போதைய ஆரோக்கிய பழக்கவழக்கங்களைப் பின்பற்றவும்",
        "வழக்கமான உடற்பயிற்சி, சீரான உணவு",
        "போதிய தூக்கம், மனஅழுத்த மேலாண்மை",
        "மருத்துவ பரிசோதனைகள்",
        "நீர் உட்கொள்ளுதல்",
      ],
      prevention: ["அருகிய பழக்க வழக்கங்கள் தொடரவும்", "தடுப்பூசிகள் புதுப்பித்து கொள்ளவும்", "வழக்கமான சோதனைகள்", "தடுப்புச் சிகிச்சை"],
    },
  },
}

const API_BASE = "http://localhost:5000"

const PALETTE = {
  moonlight: "#E1E2E1",
  midnight: "#2A3A54", // slightly cooler and darker
  ocean: "#16365C",
  noir: "#0B1526",
  accent: "#17B6C9", // muted cyan/teal (less glaring)
  accentSoft: "rgba(23,182,201,.16)", // soft-fill for selections
}

const HERO_IMAGE = "/images/health-hero2.png"

function RippleButton({ children, onClick, disabled, variant = "ghost", ariaLabel, className = "" }) {
  const [ripples, setRipples] = useState([])
  const btnRef = useRef(null)

  const handlePointerDown = (e) => {
    const rect = btnRef.current?.getBoundingClientRect()
    if (!rect) return
    const size = Math.max(rect.width, rect.height)
    const x = e.clientX - rect.left - size / 2
    const y = e.clientY - rect.top - size / 2
    const id = Math.random()
    setRipples((r) => [...r, { id, x, y, size }])
    setTimeout(() => setRipples((r) => r.filter((p) => p.id !== id)), 500)
  }

  const base =
    "relative overflow-hidden rounded-xl px-6 py-3 text-sm font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 active:scale-[0.99] disabled:opacity-60 disabled:cursor-not-allowed"

  const styles =
    variant === "cta"
      ? "bg-[var(--accent)] text-[var(--noir)] hover:brightness-110 shadow-[0_8px_24px_rgba(23,182,201,.25)] focus-visible:ring-[var(--accent)]"
      : variant === "soft"
        ? "bg-[var(--accentSoft)] text-[var(--accent)] border border-[rgba(23,182,201,.40)] hover:bg-[rgba(23,182,201,.20)] hover:border-[rgba(23,182,201,.55)] focus-visible:ring-[rgba(23,182,201,.45)]"
        : "border border-[rgba(23,182,201,.30)] text-[var(--accent)] hover:bg-[rgba(23,182,201,.08)] focus-visible:ring-[rgba(23,182,201,.35)]"

  return (
    <button
      ref={btnRef}
      type="button"
      aria-label={ariaLabel}
      disabled={disabled}
      onClick={onClick}
      onPointerDown={handlePointerDown}
      className={`${base} ${styles} ${className}`}
      style={{ ["--accent"]: PALETTE.accent, ["--accentSoft"]: PALETTE.accentSoft, ["--noir"]: PALETTE.noir }}
    >
      {children}
      {ripples.map((r) => (
        <span
          key={r.id}
          className="pointer-events-none absolute rounded-full"
          style={{
            left: r.x,
            top: r.y,
            width: r.size,
            height: r.size,
            background: "rgba(23,182,201,.18)", // softer ripple
            transform: "scale(0)",
            animation: "ripple 500ms ease-out forwards",
          }}
        />
      ))}
      <style>{`@keyframes ripple { to { transform: scale(2.4); opacity: 0; } }`}</style>
    </button>
  )
}

function TiltCard({ children, className = "" }) {
  const ref = useRef(null)
  const handleMove = (e) => {
    const el = ref.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const px = (e.clientX - rect.left) / rect.width
    const py = (e.clientY - rect.top) / rect.height
    const rx = (py - 0.5) * -4
    const ry = (px - 0.5) * 4
    el.style.transform = `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg)`
  }
  const reset = () => {
    const el = ref.current
    if (!el) return
    el.style.transform = "perspective(900px) rotateX(0deg) rotateY(0deg)"
  }
  return (
    <div
      ref={ref}
      onMouseMove={handleMove}
      onMouseLeave={reset}
      className={[
        "group transition-transform duration-300 will-change-transform",
        "rounded-2xl bg-[rgba(13,24,42,.50)] backdrop-blur-xl border",
        "hover:-translate-y-0.5 hover:shadow-2xl",
        className,
      ].join(" ")}
      style={{
        borderColor: "rgba(34,211,238,.15)",
        boxShadow: "0 2px 0 rgba(34,211,238,.08), 0 10px 40px rgba(0,0,0,.30)",
      }}
    >
      {children}
    </div>
  )
}

export default function App() {
  const [lang, setLang] = useState("en")
  const t = useMemo(() => translations[lang], [lang])

  const [formData, setFormData] = useState({
    fever: 0,
    cough: 0,
    fatigue: 0,
    difficulty_breathing: 0,
    age: "",
    gender: 0, // 0: female, 1: male
    blood_pressure: 1, // 0: low, 1: normal, 2: high
    cholesterol_level: 1, // 0: low, 1: normal, 2: high
  })

  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const abortRef = useRef(null)

  const [hydrated, setHydrated] = useState(false)
  const [showLangModal, setShowLangModal] = useState(false)

  useEffect(() => {
    setShowLangModal(true)
    setHydrated(true)
  }, [])

  const selectLanguage = (code) => {
    setLang(code)
    setShowLangModal(false)
  }

  const onInputChange = (name, value) => {
    setFormData((prev) => ({ ...prev, [name]: value }))
    setPrediction(null)
    setError(null)
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

  const handleSubmit = async () => {
    if (!formData.age) {
      setError(t.ageRequired)
      return
    }

    if (abortRef.current) abortRef.current.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fever: Number(formData.fever),
          cough: Number(formData.cough),
          fatigue: Number(formData.fatigue),
          difficulty_breathing: Number(formData.difficulty_breathing),
          age: Number(formData.age),
          gender: Number(formData.gender),
          blood_pressure: Number(formData.blood_pressure),
          cholesterol_level: Number(formData.cholesterol_level),
        }),
        signal: controller.signal,
      })

      const data = await res.json()
      if (!res.ok) throw new Error(data?.error || t.invalidResponse)
      const rf = data?.prediction
      if (!rf) throw new Error(t.invalidResponse)
      setPrediction({
        model: "Random Forest",
        disease: rf.predicted_disease,
        confidence: 0.87, // Placeholder, replace with actual confidence if available from API
        risk_assessment: rf.risk_assessment, // Assuming risk_assessment is part of the API response
      })
    } catch (e) {
      if (e.name !== "AbortError") setError(e.message || String(e))
    } finally {
      setLoading(false)
      abortRef.current = null
    }
  }

  const diseaseDisplay = (d) => diseaseDisplayMap[lang]?.[d] || d
  const remediesFor = (d) => remediesDB[lang]?.[d] || remediesDB[lang]?.["Healthy"]

  // Updated SymptomToggle to use new ripple button styles and tilt card
  function SymptomToggle({ icon: Icon, title, name, value, onChange }) {
    return (
      <TiltCard className="rounded-xl">
        <div className="p-4">
          <div className="mb-3 flex items-center gap-2">
            <div
              className="rounded-md p-1.5"
              style={{
                background: "rgba(23,182,201,.10)",
                color: PALETTE.accent,
                border: `1px solid rgba(23,182,201,.20)`,
              }}
            >
              <Icon className="h-4 w-4" />
            </div>
            <span className="text-sm font-medium">{title}</span>
          </div>
          <div className="flex gap-2">
            <RippleButton
              ariaLabel={`${title} - No`}
              variant={value === 0 ? "soft" : "ghost"}
              onClick={() => onChange(name, 0)}
              className="flex-1 py-2"
            >
              {t.no}
            </RippleButton>
            <RippleButton
              ariaLabel={`${title} - Yes`}
              variant={value === 1 ? "soft" : "ghost"}
              onClick={() => onChange(name, 1)}
              className="flex-1 py-2"
            >
              {t.yes}
            </RippleButton>
          </div>
        </div>
      </TiltCard>
    )
  }

  // Updated SelectField to use new ripple button styles and tilt card
  function SelectField({ icon: Icon, title, name, value, options }) {
    return (
      <TiltCard className="rounded-xl">
        <div className="p-4">
          <div className="mb-3 flex items-center gap-2">
            <div
              className="rounded-md p-1.5"
              style={{
                background: "rgba(23,182,201,.10)",
                color: PALETTE.accent,
                border: `1px solid rgba(23,182,201,.20)`,
              }}
            >
              <Icon className="h-4 w-4" />
            </div>
            <span className="text-sm font-medium">{title}</span>
          </div>
          <select
            value={value}
            onChange={(e) => onInputChange(name, Number(e.target.value))}
            className="w-full rounded-lg p-3 text-sm font-medium focus:outline-none transition-shadow"
            style={{
              background: "rgba(255,255,255,.05)",
              color: "rgba(225,226,225,.95)",
              border: "1px solid rgba(23,182,201,.20)",
              boxShadow: "0 1px 0 rgba(0,0,0,.30)",
            }}
            onFocus={(e) => (e.currentTarget.style.boxShadow = "0 0 0 3px rgba(23,182,201,.30)")}
            onBlur={(e) => (e.currentTarget.style.boxShadow = "0 1px 0 rgba(0,0,0,.30)")}
          >
            {options.map((label, idx) => (
              <option key={idx} value={idx} className="bg-[#0D182A]" style={{ color: "rgba(225,226,225,.95)" }}>
                {label}
              </option>
            ))}
          </select>
        </div>
      </TiltCard>
    )
  }

  return (
    <div
      className="min-h-dvh selection:bg-[#22D3EE] selection:text-[#0D182A] relative"
      style={{
        background: "linear-gradient(135deg, #0D182A 0%, #1a2842 35%, #16365C 70%, #2A3A54 100%)",
        color: "rgba(225,226,225,.95)",
      }}
    >
      <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
        <div
          className="absolute left-[-15%] top-[-15%] h-[500px] w-[500px] rounded-full blur-[160px] opacity-18"
          style={{ background: "radial-gradient(circle, rgba(23,182,201,.85) 0%, transparent 70%)" }}
        />
        <div
          className="absolute right-[-15%] bottom-[-15%] h-[600px] w-[600px] rounded-full blur-[180px] opacity-14"
          style={{ background: "radial-gradient(circle, rgba(25,59,111,.85) 0%, transparent 70%)" }}
        />
      </div>

      {hydrated && showLangModal && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 grid place-items-center p-4 bg-black/50 backdrop-blur-md"
        >
          <div className="w-full max-w-md animate-in fade-in zoom-in-95 duration-200">
            <TiltCard className="rounded-2xl">
              <div className="p-8">
                <h2 className="mb-6 text-center text-2xl font-bold text-balance">{t.languagePromptTitle}</h2>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <RippleButton ariaLabel="Select English" onClick={() => selectLanguage("en")} variant="soft">
                    {translations.en.english}
                  </RippleButton>
                  <RippleButton ariaLabel="தமிழ் தேர்வு" onClick={() => selectLanguage("ta")} variant="ghost">
                    {translations.ta.tamil}
                  </RippleButton>
                </div>
              </div>
            </TiltCard>
          </div>
        </div>
      )}

      <header
        className="sticky top-0 z-30 bg-[rgba(13,24,42,.70)] backdrop-blur-lg supports-[backdrop-filter]:bg-[rgba(13,24,42,.55)]"
        style={{ borderBottom: "1px solid rgba(34,211,238,.12)" }}
      >
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div
              className="rounded-xl p-2.5 shadow-lg"
              style={{ background: "rgba(34,211,238,.15)", border: "1px solid rgba(34,211,238,.30)" }}
            >
              <Activity className="h-6 w-6" style={{ color: PALETTE.accent }} />
            </div>
            <h1 className="text-xl font-bold text-balance">{t.appTitle}</h1>
          </div>
          <div className="flex items-center gap-2">
            <RippleButton
              ariaLabel="Switch to English"
              variant={lang === "en" ? "soft" : "ghost"}
              onClick={() => setLang("en")}
              className="px-4 py-2"
            >
              EN
            </RippleButton>
            <RippleButton
              ariaLabel="தமிழ்"
              variant={lang === "ta" ? "soft" : "ghost"}
              onClick={() => setLang("ta")}
              className="px-4 py-2"
            >
              தமிழ்
            </RippleButton>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6">
        <section className="grid gap-12 py-16 md:grid-cols-2 md:py-24 items-center">
          <div className="space-y-6">
            <h2 className="text-4xl font-bold leading-tight text-balance md:text-5xl lg:text-6xl">
              {lang === "en"
                ? "Shaping the future of health prediction, for all to see"
                : "எல்லோருக்கும் சுகாதார கணிப்பு எதிர்காலம்"}
            </h2>
            <p className="text-lg leading-relaxed" style={{ color: "rgba(225,226,225,.80)" }}>
              {lang === "en"
                ? "For millions of people around the world, accurate health insights and limited access to excellent predictive care are just the way it is. We don't see it that way."
                : "உலகம் முழுவதும் மில்லியன் கணக்கான மக்களுக்கு துல்லியமான சுகாதார புள்ளிவிவரங்கள் மற்றும் கணிப்பு பராமரிப்பு அணுகல் குறைவாக உள்ளது. நாங்கள் அதை அப்படி பார்க்கவில்லை."}
            </p>
            <p className="text-base leading-relaxed" style={{ color: "rgba(225,226,225,.75)" }}>
              {lang === "en"
                ? "Introducing our AI Health Predictor — the first Random Forest-powered platform to advance personalized health assessment for all, worldwide."
                : "எங்கள் ஏஐ ஹெல்த் ப்ரிடிக்டரை அறிமுகப்படுத்துகிறோம் — உலகம் முழுவதும் அனைவருக்கும் தனிப்பயன் சுகாதார மதிப்பீட்டை மேம்படுத்த முதல் ரேண்டம் ஃபாரஸ்ட் இயங்கும் தளம்."}
            </p>
            <div className="flex flex-col gap-3 sm:flex-row">
              <RippleButton
                ariaLabel="Start Assessment"
                variant="cta"
                onClick={() => {
                  const el = document.getElementById("assessment")
                  if (el) el.scrollIntoView({ behavior: "smooth", block: "start" })
                }}
              >
                {lang === "en" ? "Start Assessment →" : "மதிப்பீடு தொடங்கு →"}
              </RippleButton>
              <RippleButton
                ariaLabel="Learn More"
                variant="ghost"
                onClick={() => {
                  const el = document.getElementById("features")
                  if (el) el.scrollIntoView({ behavior: "smooth", block: "start" })
                }}
              >
                {lang === "en" ? "Learn More" : "மேலும் அறிக"}
              </RippleButton>
            </div>
          </div>

          <div className="relative">
            <div
              className="absolute inset-0 rounded-full blur-[90px] opacity-16"
              style={{ background: "radial-gradient(circle, rgba(23,182,201,.75) 0%, transparent 70%)" }}
            />
            <img
              src={HERO_IMAGE || "/placeholder.svg?height=480&width=720&query=ai%20health%20prediction%20illustration"}
              alt={lang === "en" ? "AI health prediction illustration with DNA and ECG" : "ஏஐ சுகாதார கணிப்பு படிமம்"}
              className="relative w-full h-auto"
              style={{ filter: "drop-shadow(0 14px 40px rgba(0,0,0,.35))" }}
            />
          </div>
        </section>

        <section id="features" className="py-16">
          <div className="mb-12 text-center">
            <h3 className="text-3xl font-bold mb-4 text-balance">
              {lang === "en" ? "Why Choose AI Health Predictor?" : "ஏஐ சுகாதார கணிப்பை ஏன் தேர்வு செய்ய வேண்டும்?"}
            </h3>
            <p className="text-lg max-w-3xl mx-auto" style={{ color: "rgba(225,226,225,.80)" }}>
              {lang === "en"
                ? "Combining cutting-edge machine learning with bilingual support to bring accessible health insights to everyone."
                : "அனைவருக்கும் அணுகக்கூடிய சுகாதார நுண்ணறிவுகளை கொண்டு வர அதிநவீன இயந்திர கற்றல் மற்றும் இரு மொழி ஆதரவை இணைக்கிறது."}
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-3">
            <TiltCard>
              <div className="p-8">
                <div
                  className="mb-4 inline-flex rounded-xl p-3"
                  style={{ background: "rgba(34,211,238,.12)", border: "1px solid rgba(34,211,238,.25)" }}
                >
                  <Brain className="h-8 w-8" style={{ color: PALETTE.accent }} />
                </div>
                <h4 className="text-xl font-semibold mb-3">{lang === "en" ? "Random Forest AI" : "ரேண்டம் ஃபாரஸ்ட் ஏஐ"}</h4>
                <p className="leading-relaxed" style={{ color: "rgba(225,226,225,.80)" }}>
                  {lang === "en"
                    ? "Powered by a robust Random Forest model trained on real health data to deliver accurate predictions with confidence scores."
                    : "துல்லியமான கணிப்புகளை நம்பிக்கை மதிப்பெண்களுடன் வழங்க உண்மையான சுகாதார தரவுகளில் பயிற்சி பெற்ற வலுவான ரேண்டம் ஃபாரஸ்ட் மாதிரியால் இயக்கப்படுகிறது."}
                </p>
              </div>
            </TiltCard>

            <TiltCard>
              <div className="p-8">
                <div
                  className="mb-4 inline-flex rounded-xl p-3"
                  style={{ background: "rgba(34,211,238,.12)", border: "1px solid rgba(34,211,238,.25)" }}
                >
                  <Globe className="h-8 w-8" style={{ color: PALETTE.accent }} />
                </div>
                <h4 className="text-xl font-semibold mb-3">{lang === "en" ? "Bilingual Support" : "இரு மொழி ஆதரவு"}</h4>
                <p className="leading-relaxed" style={{ color: "rgba(225,226,225,.80)" }}>
                  {lang === "en"
                    ? "Full support for English and Tamil languages, making health predictions accessible to diverse communities worldwide."
                    : "ஆங்கிலம் மற்றும் தமிழ் மொழிகளுக்கான முழு ஆதரவு, உலகெங்கிலும் உள்ள பல்வேறு சமூகங்களுக்கு சுகாதார கணிப்புகளை அணுகக்கூடியதாக ஆக்குகிறது."}
                </p>
              </div>
            </TiltCard>

            <TiltCard>
              <div className="p-8">
                <div
                  className="mb-4 inline-flex rounded-xl p-3"
                  style={{ background: "rgba(34,211,238,.12)", border: "1px solid rgba(34,211,238,.25)" }}
                >
                  <Shield className="h-8 w-8" style={{ color: PALETTE.accent }} />
                </div>
                <h4 className="text-xl font-semibold mb-3">
                  {lang === "en" ? "Personalized Care" : "தனிப்பயன் பராமரிப்பு"}
                </h4>
                <p className="leading-relaxed" style={{ color: "rgba(225,226,225,.80)" }}>
                  {lang === "en"
                    ? "Get tailored treatment recommendations and prevention tips based on your symptoms, age, and health profile."
                    : "உங்கள் அறிகுறிகள், வயது மற்றும் சுகாதார சுயவிவரத்தின் அடிப்படையில் தனிப்பயனாக்கப்பட்ட சிகிச்சை பரிந்துரைகள் மற்றும் தடுப்பு குறிப்புகளைப் பெறுங்கள்."}
                </p>
              </div>
            </TiltCard>
          </div>
        </section>

        <section id="assessment" className="py-16">
          <div className="mb-8 text-center">
            <h3 className="text-3xl font-bold mb-4 text-balance">
              {lang === "en" ? "Start Your Health Assessment" : "உங்கள் சுகாதார மதிப்பீட்டைத் தொடங்குங்கள்"}
            </h3>
            <p className="text-base max-w-2xl mx-auto" style={{ color: "rgba(225,226,225,.80)" }}>
              {t.appSubtitle}
            </p>
          </div>

          <div className="space-y-6">
            <TiltCard className="rounded-2xl">
              <div className="p-8">
                <h4 className="mb-6 flex items-center gap-3 text-xl font-semibold">
                  <span
                    className="rounded-lg p-2"
                    style={{
                      background: "rgba(34,211,238,.12)",
                      color: PALETTE.accent,
                      border: `1px solid rgba(34,211,238,.25)`,
                    }}
                  >
                    <Thermometer className="h-5 w-5" />
                  </span>
                  {t.currentSymptoms}
                </h4>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-12">
                  <div className="md:col-span-6">
                    <SymptomToggle
                      icon={Thermometer}
                      title={t.fever}
                      name="fever"
                      value={formData.fever}
                      onChange={onInputChange}
                    />
                  </div>
                  <div className="md:col-span-6">
                    <SymptomToggle
                      icon={Wind}
                      title={t.cough}
                      name="cough"
                      value={formData.cough}
                      onChange={onInputChange}
                    />
                  </div>
                  <div className="md:col-span-4">
                    <SymptomToggle
                      icon={Activity}
                      title={t.fatigue}
                      name="fatigue"
                      value={formData.fatigue}
                      onChange={onInputChange}
                    />
                  </div>
                  <div className="md:col-span-8">
                    <SymptomToggle
                      icon={Wind}
                      title={t.difficultyBreathing}
                      name="difficulty_breathing"
                      value={formData.difficulty_breathing}
                      onChange={onInputChange}
                    />
                  </div>
                </div>
              </div>
            </TiltCard>

            <TiltCard className="rounded-2xl">
              <div className="p-8">
                <h4 className="mb-6 flex items-center gap-3 text-xl font-semibold">
                  <span
                    className="rounded-lg p-2"
                    style={{
                      background: "rgba(34,211,238,.12)",
                      border: "1px solid rgba(34,211,238,.25)",
                      color: PALETTE.accent,
                    }}
                  >
                    <User className="h-5 w-5" />
                  </span>
                  {t.patientProfile}
                </h4>

                <div className="grid grid-cols-1 gap-4 md:grid-cols-12">
                  <div className="md:col-span-12">
                    <TiltCard className="rounded-xl">
                      <div className="p-4">
                        <div className="mb-3 flex items-center gap-2">
                          <div
                            className="rounded-md p-1.5"
                            style={{
                              background: "rgba(23,182,201,.10)",
                              border: "1px solid rgba(23,182,201,.20)",
                              color: PALETTE.accent,
                            }}
                          >
                            <User className="h-4 w-4" />
                          </div>
                          <span className="text-sm font-medium">{t.age}</span>
                        </div>
                        <input
                          type="number"
                          inputMode="numeric"
                          min={0}
                          max={120}
                          value={formData.age}
                          onChange={(e) => onInputChange("age", e.target.value)}
                          placeholder="0–120"
                          className="w-full rounded-lg p-3 text-sm font-medium focus:outline-none transition-shadow"
                          style={{
                            background: "rgba(255,255,255,.05)",
                            color: "rgba(225,226,225,.95)",
                            border: "1px solid rgba(23,182,201,.20)",
                            boxShadow: "0 1px 0 rgba(0,0,0,.30)",
                          }}
                          onFocus={(e) => (e.currentTarget.style.boxShadow = "0 0 0 3px rgba(23,182,201,.30)")}
                          onBlur={(e) => (e.currentTarget.style.boxShadow = "0 1px 0 rgba(0,0,0,.30)")}
                          aria-label={t.age}
                        />
                      </div>
                    </TiltCard>
                  </div>

                  <div className="md:col-span-4">
                    <SelectField
                      icon={User}
                      title={t.gender}
                      name="gender"
                      value={formData.gender}
                      options={[t.female, t.male]}
                    />
                  </div>
                  <div className="md:col-span-4">
                    <SelectField
                      icon={Heart}
                      title={t.bloodPressure}
                      name="blood_pressure"
                      value={formData.blood_pressure}
                      options={[t.low, t.normal, t.high]}
                    />
                  </div>
                  <div className="md:col-span-4">
                    <SelectField
                      icon={Droplets}
                      title={t.cholesterolLevel}
                      name="cholesterol_level"
                      value={formData.cholesterol_level}
                      options={[t.low, t.normal, t.high]}
                    />
                  </div>
                </div>
              </div>
            </TiltCard>

            <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
              <RippleButton
                ariaLabel="Predict"
                variant="cta"
                onClick={handleSubmit}
                disabled={loading || !formData.age}
              >
                {loading ? (
                  <>
                    <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-[var(--noir)] border-t-transparent" />
                    <span className="ml-2">{t.analyzing}</span>
                  </>
                ) : (
                  <>
                    <Brain className="h-4 w-4 mr-2" />
                    {t.predictAnalyze}
                  </>
                )}
              </RippleButton>

              <RippleButton ariaLabel="Reset" variant="ghost" onClick={resetForm}>
                {t.resetForm}
              </RippleButton>
            </div>

            {error && (
              <TiltCard className="rounded-xl">
                <div
                  className="p-4"
                  style={{ background: "rgba(239,68,68,.10)", border: "1px solid rgba(239,68,68,.35)" }}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold">{t.errorPrefix}</span>
                    <span className="text-sm font-medium">{error}</span>
                  </div>
                </div>
              </TiltCard>
            )}

            {prediction && (
              <>
                <TiltCard className="rounded-2xl">
                  <div className="p-8">
                    <h4 className="mb-6 flex items-center gap-3 text-xl font-semibold">
                      <span
                        className="rounded-lg p-2"
                        style={{
                          background: "rgba(34,211,238,.12)",
                          color: PALETTE.accent,
                          border: `1px solid rgba(34,211,238,.25)`,
                        }}
                      >
                        <Brain className="h-5 w-5" />
                      </span>
                      {t.aiResults}
                    </h4>

                    <TiltCard className="rounded-xl">
                      <div className="p-6">
                        <div className="mb-1 text-sm font-medium" style={{ color: "rgba(225,226,225,.75)" }}>
                          {t.rfModel}
                        </div>
                        <div className="text-2xl font-bold">{diseaseDisplay(prediction.disease)}</div>
                        {typeof prediction.confidence === "number" && (
                          <div className="mt-4 flex items-center gap-3">
                            <div className="h-3 flex-1 rounded-full" style={{ background: "rgba(225,226,225,.10)" }}>
                              <div
                                className="h-3 rounded-full transition-all duration-700"
                                style={{
                                  width: `${Math.max(0, Math.min(100, prediction.confidence * 100))}%`,
                                  background: "linear-gradient(90deg, #22D3EE, #193B6F)",
                                }}
                              />
                            </div>
                            <span className="text-sm font-bold">{(prediction.confidence * 100).toFixed(1)}%</span>
                          </div>
                        )}
                      </div>
                    </TiltCard>
                  </div>
                </TiltCard>

                <TiltCard className="rounded-2xl">
                  <div className="p-8">
                    <h4 className="mb-6 flex items-center gap-3 text-xl font-semibold">
                      <span
                        className="rounded-lg p-2"
                        style={{
                          background: "rgba(34,211,238,.12)",
                          color: PALETTE.accent,
                          border: `1px solid rgba(34,211,238,.25)`,
                        }}
                      >
                        <Pill className="h-5 w-5" />
                      </span>
                      {t.carePrevention}
                    </h4>

                    {(() => {
                      const r = remediesFor(prediction.disease)
                      return (
                        <div className="grid gap-6 md:grid-cols-12">
                          <div className="md:col-span-7">
                            <TiltCard className="rounded-xl">
                              <div className="p-6">
                                <h5 className="mb-4 flex items-center gap-2 text-lg font-semibold">
                                  <Stethoscope className="h-5 w-5" style={{ color: PALETTE.accent }} />
                                  {t.treatmentCare}
                                </h5>
                                <ul className="space-y-2">
                                  {r.remedies.map((item, idx) => (
                                    <li
                                      key={idx}
                                      className="text-sm leading-relaxed"
                                      style={{ color: "rgba(225,226,225,.85)" }}
                                    >
                                      • {item}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </TiltCard>
                          </div>

                          <div className="md:col-span-5">
                            <TiltCard className="rounded-xl">
                              <div className="p-6">
                                <h5 className="mb-4 flex items-center gap-2 text-lg font-semibold">
                                  <Pill className="h-5 w-5" style={{ color: PALETTE.accent }} />
                                  {t.preventionTips}
                                </h5>
                                <ul className="space-y-2">
                                  {r.prevention.map((item, idx) => (
                                    <li
                                      key={idx}
                                      className="text-sm leading-relaxed"
                                      style={{ color: "rgba(225,226,225,.85)" }}
                                    >
                                      • {item}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </TiltCard>
                          </div>
                        </div>
                      )
                    })()}

                    <TiltCard className="mt-6 rounded-xl">
                      <div className="p-6">
                        <p className="text-sm leading-relaxed">
                          <strong>{t.disclaimerTitle}</strong> {t.disclaimerBody}
                        </p>
                      </div>
                    </TiltCard>
                  </div>
                </TiltCard>
              </>
            )}
          </div>
        </section>
      </main>

      <footer className="border-t mt-16 py-8" style={{ borderColor: "rgba(34,211,238,.12)" }}>
        <div className="mx-auto max-w-7xl px-6 text-center text-sm" style={{ color: "rgba(225,226,225,.70)" }}>
          <p>
            {lang === "en"
              ? "© 2025 AI Health Predictor. For educational purposes only. Always consult qualified healthcare providers."
              : "© 2025 ஏஐ ஹெல்த் ப்ரிடிக்டர். கல்வி நோக்கங்களுக்காக மட்டுமே. எப்போதும் தகுதியான மருத்துவர்களை அணுகவும்."}
          </p>
        </div>
      </footer>
    </div>
  )
}
