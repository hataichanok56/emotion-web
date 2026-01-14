"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

/** * CSS SECTION
 * หากคุณใช้ Next.js แนะนำให้แยกส่วน @keyframes นี้ไปไว้ใน globals.css
 * แต่ถ้าต้องการเทสทันที สามารถใส่ในแท็ก <style> ภายใน component ได้
 */

type CvType = any;

const EMOTION_COLORS: Record<string, string> = {
  angry: "#FF4B4B",
  disgust: "#FFD700",
  fear: "#9CA3AF",
  happy: "#FFC0CB",
  neutral: "#10B981",
  sad: "#8B5CF6",
  surprise: "#F59E0B",
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const requestRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [status, setStatus] = useState<string>("กำลังเตรียมระบบ...");
  const [emotion, setEmotion] = useState<string>("neutral");
  const [conf, setConf] = useState<number>(0);
  const [isCameraOn, setIsCameraOn] = useState<boolean>(false);
  const [isModelReady, setIsModelReady] = useState<boolean>(false);

  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  // 1) Load OpenCV.js
  async function loadOpenCV() {
    if (typeof window === "undefined") return;
    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }

    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;
      script.onload = () => {
        const cv = (window as any).cv;
        const waitReady = () => {
          if (cv?.Mat) {
            cvRef.current = cv;
            resolve();
          } else {
            setTimeout(waitReady, 50);
          }
        };
        if ("onRuntimeInitialized" in cv) {
          cv.onRuntimeInitialized = () => waitReady();
        } else {
          waitReady();
        }
      };
      script.onerror = () => reject(new Error("โหลด OpenCV ไม่สำเร็จ"));
      document.body.appendChild(script);
    });
  }

  // 2) Load Cascade
  async function loadCascade() {
    const cv = cvRef.current;
    const res = await fetch("/opencv/haarcascade_frontalface_default.xml");
    const data = new Uint8Array(await res.arrayBuffer());
    const cascadePath = "haarcascade_frontalface_default.xml";
    try {
      cv.FS_unlink(cascadePath);
    } catch {}
    cv.FS_createDataFile("/", cascadePath, data, true, false, false);
    const faceCascade = new cv.CascadeClassifier();
    faceCascade.load(cascadePath);
    faceCascadeRef.current = faceCascade;
  }

  // 3) Load Model
  async function loadModel() {
    const session = await ort.InferenceSession.create(
      "/models/emotion_yolo11n_cls.onnx",
      {
        executionProviders: ["wasm"],
      },
    );
    sessionRef.current = session;
    const clsRes = await fetch("/models/classes.json");
    classesRef.current = await clsRes.json();
  }

  // 4) Toggle Camera
  async function toggleCamera() {
    if (isCameraOn) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
      setIsCameraOn(false);
      setStatus("ปิดกล้องแล้ว");
    } else {
      try {
        setStatus("กำลังเปิดกล้อง...");
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setIsCameraOn(true);
          setStatus("กำลังประมวลผล...");
          requestRef.current = requestAnimationFrame(loop);
        }
      } catch (err) {
        setStatus("ไม่สามารถเข้าถึงกล้องได้");
      }
    }
  }

  function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size;
    tmp.height = size;
    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);
    const imgData = ctx.getImageData(0, 0, size, size).data;
    const float = new Float32Array(1 * 3 * size * size);
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) {
        float[c * size * size + i] = imgData[i * 4 + c] / 255.0;
      }
    }
    return new ort.Tensor("float32", float, [1, 3, size, size]);
  }

  function softmax(logits: Float32Array) {
    const maxLogit = Math.max(...Array.from(logits));
    const scores = logits.map((l) => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map((s) => s / sum);
  }

  async function loop() {
    if (!videoRef.current || videoRef.current.paused || videoRef.current.ended)
      return;

    try {
      const cv = cvRef.current;
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!cv || !canvas || !video) return;

      const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      const src = cv.imread(canvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      const faces = new cv.RectVector();
      faceCascadeRef.current.detectMultiScale(gray, faces, 1.1, 3, 0);

      let bestRect = null;
      let maxArea = 0;

      for (let i = 0; i < faces.size(); i++) {
        const r = faces.get(i);
        if (r.width * r.height > maxArea) {
          maxArea = r.width * r.height;
          bestRect = r;
        }
      }

      if (bestRect) {
        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = bestRect.width;
        faceCanvas.height = bestRect.height;
        faceCanvas
          .getContext("2d")!
          .drawImage(
            canvas,
            bestRect.x,
            bestRect.y,
            bestRect.width,
            bestRect.height,
            0,
            0,
            bestRect.width,
            bestRect.height,
          );

        const input = preprocessToTensor(faceCanvas);
        const feeds = { [sessionRef.current!.inputNames[0]]: input };
        const out = await sessionRef.current!.run(feeds);
        const probs = softmax(
          out[sessionRef.current!.outputNames[0]].data as Float32Array,
        );

        let maxIdx = 0;
        probs.forEach((p, i) => {
          if (p > probs[maxIdx]) maxIdx = i;
        });

        const currentEmotion = classesRef.current![maxIdx];
        const currentConf = probs[maxIdx];

        setEmotion(currentEmotion);
        setConf(currentConf);

        const color = EMOTION_COLORS[currentEmotion] || "#white";
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(bestRect.x, bestRect.y, bestRect.width, bestRect.height);

        ctx.fillStyle = color;
        ctx.fillRect(bestRect.x, bestRect.y - 35, 160, 35);
        ctx.fillStyle = "white";
        ctx.font = "bold 18px sans-serif";
        ctx.fillText(
          `${currentEmotion.toUpperCase()} ${(currentConf * 100).toFixed(1)}%`,
          bestRect.x + 10,
          bestRect.y - 10,
        );
      }

      src.delete();
      gray.delete();
      faces.delete();

      requestRef.current = requestAnimationFrame(loop);
    } catch (e) {
      console.error(e);
      requestRef.current = requestAnimationFrame(loop);
    }
  }

  useEffect(() => {
    (async () => {
      try {
        await loadOpenCV();
        await loadCascade();
        await loadModel();
        setIsModelReady(true);
        setStatus("ระบบพร้อมใช้งาน");
      } catch (e) {
        setStatus("โหลดไม่สำเร็จ: " + e);
      }
    })();
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, []);

  const themeColor = EMOTION_COLORS[emotion] || "#10B981";

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 p-4 md:p-8 font-sans transition-colors duration-1000"
          style={{ backgroundImage: `radial-gradient(circle at center, ${themeColor}10 0%, transparent 70%)` }}>

      {/* Dynamic CSS for Scanning Line */}
      <style jsx global>{`
        @keyframes scan {
          0% { transform: translateY(-10%); opacity: 0; }
          50% { opacity: 1; }
          100% { transform: translateY(1000%); opacity: 0; }
        }
        .animate-scan {
          animation: scan 3s linear infinite;
        }
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-spin-slow {
          animation: spin-slow 8s linear infinite;
        }
      `}</style>

      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header Section */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-zinc-900/50 p-6 rounded-2xl border border-zinc-800 backdrop-blur-md">
          <div>
            <h1 className="text-3xl font-black tracking-tighter bg-gradient-to-r from-white to-zinc-500 bg-clip-text text-transparent">
              Face Emotion (NoName!!)
            </h1>
            <p className="text-zinc-400 text-sm mt-1 flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isModelReady ? "bg-green-500 animate-pulse" : "bg-red-500"}`}></span>
              {status}
            </p>
          </div>

          <button
            onClick={toggleCamera}
            disabled={!isModelReady}
            className={`px-8 py-3 rounded-xl font-bold transition-all active:scale-95 flex items-center justify-center gap-2 ${
              isCameraOn
                ? "bg-red-500/10 text-red-500 border border-red-500/50 hover:bg-red-500 hover:text-white"
                : "bg-white text-black hover:bg-zinc-200 disabled:opacity-50"
            }`}
          >
            {isCameraOn ? "◼️ Stop Camera" : " ▶ Start Camera "}
          </button>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Viewport พร้อมลูกเล่นใหม่ */}
          <div
            className="lg:col-span-2 relative aspect-[16/10] bg-black rounded-3xl overflow-hidden border-4 transition-all duration-700 ease-in-out"
            style={{
              borderColor: isCameraOn ? themeColor : "#27272a",
              boxShadow: isCameraOn ? `0 0 40px ${themeColor}30` : 'none'
            }}
          >
            <video ref={videoRef} className="hidden" playsInline />
            <canvas ref={canvasRef} className="w-full h-full object-cover" />

            {/* แถบเส้นสแกน (Scanning Line) */}
            {isCameraOn && (
              <div className="absolute inset-0 pointer-events-none overflow-hidden z-10">
                <div
                  className="w-full h-[3px] shadow-[0_0_20px_rgba(255,255,255,0.8)] animate-scan"
                  style={{
                    background: `linear-gradient(to right, transparent, ${themeColor}, transparent)`,
                  }}
                ></div>
              </div>
            )}

            {!isCameraOn && (
              <div className="absolute inset-0 flex items-center justify-center bg-zinc-900">
                <div className="text-center">
                  <div className="w-20 h-20 border-2 border-dashed border-zinc-700 rounded-full animate-spin-slow mx-auto mb-6 flex items-center justify-center">
                    <div className="w-12 h-12 border-2 border-zinc-800 rounded-full"></div>
                  </div>
                  <p className="text-zinc-500 font-bold tracking-[0.3em] uppercase text-sm">System Ready</p>
                </div>
              </div>
            )}

            {/* Overlay Emotion Badge (แบบปรับปรุงใหม่) */}
            {isCameraOn && (
              <>
                <div className="absolute top-6 left-6 flex items-center gap-3 z-20">
                  <div className="flex h-3 w-3 relative">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{backgroundColor: themeColor}}></span>
                    <span className="relative inline-flex rounded-full h-3 w-3" style={{backgroundColor: themeColor}}></span>
                  </div>
                  <span className="text-xs font-bold tracking-[0.2em] text-white/80 uppercase drop-shadow-md">Neural Processing...</span>
                </div>

                <div className="absolute top-6 right-6 z-20">
                  <div
                    className="px-6 py-3 rounded-2xl border backdrop-blur-xl transition-all duration-500 shadow-2xl"
                    style={{
                      backgroundColor: `${themeColor}15`,
                      borderColor: `${themeColor}50`,
                      color: "white",
                    }}
                  >
                    <span className="text-[10px] uppercase font-black tracking-widest block opacity-60 mb-1">
                      Current State
                    </span>
                    <span className="font-mono text-2xl font-bold italic" style={{ color: themeColor }}>
                      {emotion.toUpperCase()}
                    </span>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Stats & Info Card */}
          <div className="space-y-6">
            <div className="bg-zinc-900/50 p-8 rounded-3xl border border-zinc-800 h-full flex flex-col justify-between backdrop-blur-sm">
              <div className="space-y-8">
                <div>
                  <h2 className="text-zinc-500 text-xs font-black tracking-[0.2em] uppercase mb-6">
                    Analysis Confidence
                  </h2>
                  <div className="relative pt-1">
                    <div className="flex mb-2 items-center justify-between">
                      <div>
                        <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full bg-zinc-800 text-zinc-300">
                          Match Accuracy
                        </span>
                      </div>
                      <div className="text-right">
                        <span className="text-2xl font-mono font-bold" style={{ color: themeColor }}>
                          {(conf * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    
                  </div>
                </div>

                <div className="pt-8 border-t border-zinc-800/50">
                  <h3 className="text-zinc-500 text-[10px] font-black tracking-widest uppercase mb-4">Class Distribution</h3>
                  <div className="grid grid-cols-2 gap-3">
                    {Object.keys(EMOTION_COLORS).map((name) => (
                      <div
                        key={name}
                        className={`flex items-center gap-3 p-3 rounded-xl text-xs font-bold transition-all duration-300 border ${
                          emotion === name
                            ? "bg-zinc-800 border-zinc-600 scale-105 shadow-lg"
                            : "bg-transparent border-transparent opacity-30 grayscale"
                        }`}
                      >
                        <span
                          className="w-3 h-3 rounded-full shadow-[0_0_10px_currentColor]"
                          style={{ backgroundColor: EMOTION_COLORS[name], color: EMOTION_COLORS[name] }}
                        ></span>
                        <span className="tracking-tight">{name.toUpperCase()}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
