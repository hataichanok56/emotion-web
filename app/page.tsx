"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type CvType = any;

// การกำหนดสีตามอารมณ์ที่คุณต้องการ
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
    try { cv.FS_unlink(cascadePath); } catch {}
    cv.FS_createDataFile("/", cascadePath, data, true, false, false);
    const faceCascade = new cv.CascadeClassifier();
    faceCascade.load(cascadePath);
    faceCascadeRef.current = faceCascade;
  }

  // 3) Load Model
  async function loadModel() {
    const session = await ort.InferenceSession.create("/models/emotion_yolo11n_cls.onnx", {
      executionProviders: ["wasm"],
    });
    sessionRef.current = session;
    const clsRes = await fetch("/models/classes.json");
    classesRef.current = await clsRes.json();
  }

  // 4) Toggle Camera
  async function toggleCamera() {
    if (isCameraOn) {
      // ปิดกล้อง
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
      setIsCameraOn(false);
      setStatus("ปิดกล้องแล้ว");
    } else {
      // เปิดกล้อง
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

  // 7) Main loop
  async function loop() {
    if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) return;

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
        faceCanvas.getContext("2d")!.drawImage(canvas, bestRect.x, bestRect.y, bestRect.width, bestRect.height, 0, 0, bestRect.width, bestRect.height);

        const input = preprocessToTensor(faceCanvas);
        const feeds = { [sessionRef.current!.inputNames[0]]: input };
        const out = await sessionRef.current!.run(feeds);
        const probs = softmax(out[sessionRef.current!.outputNames[0]].data as Float32Array);

        let maxIdx = 0;
        probs.forEach((p, i) => { if (p > probs[maxIdx]) maxIdx = i; });

        const currentEmotion = classesRef.current![maxIdx];
        const currentConf = probs[maxIdx];

        setEmotion(currentEmotion);
        setConf(currentConf);

        // Drawing
        const color = EMOTION_COLORS[currentEmotion] || "#white";
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(bestRect.x, bestRect.y, bestRect.width, bestRect.height);

        ctx.fillStyle = color;
        ctx.fillRect(bestRect.x, bestRect.y - 35, 160, 35);
        ctx.fillStyle = "white";
        ctx.font = "bold 18px sans-serif";
        ctx.fillText(`${currentEmotion.toUpperCase()} ${(currentConf * 100).toFixed(1)}%`, bestRect.x + 10, bestRect.y - 10);
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
    <main className="min-h-screen bg-zinc-950 text-zinc-100 p-4 md:p-8 font-sans">
      <div className="max-w-5xl mx-auto space-y-6">

        {/* Header Section */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-zinc-900/50 p-6 rounded-2xl border border-zinc-800 backdrop-blur-md">
          <div>
            <h1 className="text-3xl font-black tracking-tighter bg-gradient-to-r from-white to-zinc-500 bg-clip-text text-transparent">
              EMOTION AI <span className="text-sm font-mono text-zinc-500 ml-2">v1.1</span>
            </h1>
            <p className="text-zinc-400 text-sm mt-1 flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isModelReady ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></span>
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
            {isCameraOn ? "Stop Camera" : "Start Real-time AI"}
          </button>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Main Viewport */}
          <div className="lg:col-span-2 relative aspect-video bg-black rounded-3xl overflow-hidden border-4 transition-colors duration-500"
               style={{ borderColor: isCameraOn ? themeColor : '#27272a' }}>
            <video ref={videoRef} className="hidden" playsInline />
            <canvas ref={canvasRef} className="w-full h-full object-cover" />

            {!isCameraOn && (
              <div className="absolute inset-0 flex items-center justify-center bg-zinc-900">
                <p className="text-zinc-500 font-medium">กดปุ่ม Start เพื่อเริ่มใช้งานกล้อง</p>
              </div>
            )}

            {/* Overlay Emotion Badge */}
            {isCameraOn && (
              <div className="absolute top-4 right-4">
                <div
                  className="px-4 py-2 rounded-full blur-none border backdrop-blur-xl transition-all duration-500"
                  style={{ backgroundColor: `${themeColor}20`, borderColor: themeColor, color: themeColor }}
                >
                  <span className="text-xs uppercase font-black tracking-widest mr-2 opacity-70">Detecting:</span>
                  <span className="font-bold text-lg">{emotion.toUpperCase()}</span>
                </div>
              </div>
            )}
          </div>

          {/* Stats & Info Card */}
          <div className="space-y-6">
            <div className="bg-zinc-900/50 p-6 rounded-3xl border border-zinc-800">
              <h2 className="text-zinc-400 text-xs font-black tracking-widest uppercase mb-4">Live Statistics</h2>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-zinc-500">Confidence Score</span>
                    <span className="font-mono" style={{ color: themeColor }}>{(conf * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-zinc-800 h-3 rounded-full overflow-hidden">
                    <div
                      className="h-full transition-all duration-500"
                      style={{ width: `${conf * 100}%`, backgroundColor: themeColor }}
                    />
                  </div>
                </div>

                <div className="pt-4 border-t border-zinc-800">
                  <div className="grid grid-cols-2 gap-2">
                    {Object.keys(EMOTION_COLORS).map((name) => (
                      <div
                        key={name}
                        className={`flex items-center gap-2 p-2 rounded-lg text-xs font-medium transition-all ${emotion === name ? 'bg-zinc-800 ring-1 ring-inset ring-zinc-700' : 'opacity-40'}`}
                      >
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: EMOTION_COLORS[name] }}></span>
                        {name.charAt(0).toUpperCase() + name.slice(1)}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-zinc-900/50 p-6 rounded-3xl border border-zinc-800">
              <h3 className="text-sm font-bold mb-2">Technical Guide</h3>
              <ul className="text-xs text-zinc-500 space-y-2 list-disc pl-4">
                <li>โมเดลถูกประมวลผลบน WebAssembly (WASM) โดยตรงบนเครื่องของคุณ</li>
                <li>ใช้ Haar Cascade ในการตีกรอบหน้าก่อนส่งเข้า YOLO11-CLS</li>
                <li>สีของกรอบจะเปลี่ยนไปตามอารมณ์ที่ AI คาดการณ์ได้แม่นยำที่สุด</li>
              </ul>
            </div>
          </div>

        </div>
      </div>
    </main>
  );
}
