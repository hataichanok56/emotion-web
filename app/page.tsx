"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

// 1. ตั้งค่า WASM ให้เสถียรและปิด Proxy เพื่อเลี่ยงปัญหา SharedArrayBuffer
if (typeof window !== 'undefined') {
    // ใช้เวอร์ชัน 1.19.0 ให้ตรงกับที่ install
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/";
    // ปิด Proxy เพื่อไม่ให้เกิด Error: SharedArrayBuffer usage is restricted
    ort.env.wasm.numThreads = 1; 
    ort.env.wasm.proxy = false;
}

type CapturedImage = {
  id: number;
  src: string;
  emotion: string;
  conf: number;
  color: string;
};

const EMOTION_COLORS: Record<string, string> = {
  angry: "#FF0000",    // แดง
  disgust: "#FFFF00",  // เหลือง
  fear: "#000000",     // ดำ
  happy: "#FFC0CB",    // ชมพู
  neutral: "#00FF00",  // เขียว
  sad: "#800080",      // ม่วง
  surprise: "#FFA500", // ส้ม
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const loopRef = useRef<number | null>(null);

  const [status, setStatus] = useState<string>("กำลังโหลดระบบ...");
  const [emotion, setEmotion] = useState<string>("-");
  const [conf, setConf] = useState<number>(0);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [capturedImages, setCapturedImages] = useState<CapturedImage[]>([]);

  const cvRef = useRef<any>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  // 1. โหลด OpenCV.js แบบปลอดภัย
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
          } else { setTimeout(waitReady, 50); }
        };
        waitReady();
      };
      script.onerror = () => reject(new Error("โหลด OpenCV ไม่สำเร็จ"));
      document.body.appendChild(script);
    });
  }

  // 2. เตรียม Assets
  async function initAssets() {
    try {
      await loadOpenCV();
      const cv = cvRef.current;
      
      const res = await fetch("/opencv/haarcascade_frontalface_default.xml");
      const data = new Uint8Array(await res.arrayBuffer());
      cv.FS_createDataFile("/", "face.xml", data, true, false, false);
      const faceCascade = new cv.CascadeClassifier();
      faceCascade.load("face.xml");
      faceCascadeRef.current = faceCascade;

      sessionRef.current = await ort.InferenceSession.create("/models/emotion_yolo11n_cls.onnx", { 
        executionProviders: ["wasm"],
        graphOptimizationLevel: 'all'
      });

      const clsRes = await fetch("/models/classes.json");
      classesRef.current = await clsRes.json();
      
      setStatus("พร้อมใช้งาน");
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    }
  }

  useEffect(() => { initAssets(); }, []);

  // 3. ระบบเปิด-ปิดกล้อง (Safe Guard)
  async function toggleCamera() {
    if (isCameraOpen) {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
        videoRef.current.srcObject = null;
      }
      if (loopRef.current) cancelAnimationFrame(loopRef.current);
      setIsCameraOpen(false);
      setStatus("ปิดกล้องแล้ว");
    } else {
      try {
        setStatus("กำลังเปิดกล้อง...");
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: { ideal: 1280 }, height: { ideal: 720 } } 
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            setIsCameraOpen(true);
            setStatus("กำลังสแกน...");
            loopRef.current = requestAnimationFrame(loop);
          };
        }
      } catch (err) {
        setStatus("ไม่สามารถเข้าถึงกล้องได้");
      }
    }
  }

  const capturePhoto = () => {
    if (capturedImages.length >= 5 || !canvasRef.current || !isCameraOpen) return;
    const imageData = canvasRef.current.toDataURL("image/png");
    const currentEmotion = emotion;
    setCapturedImages([{
      id: Date.now(),
      src: imageData,
      emotion: currentEmotion,
      conf: conf,
      color: EMOTION_COLORS[currentEmotion.toLowerCase()] || "#FFFFFF"
    }, ...capturedImages]);
  };

  const resetGallery = () => setCapturedImages([]);

  function preprocess(faceCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size; tmp.height = size;
    tmp.getContext("2d")!.drawImage(faceCanvas, 0, 0, size, size);
    const imgData = tmp.getContext("2d")!.getImageData(0, 0, size, size).data;
    const float = new Float32Array(3 * size * size);
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) {
        float[c * size * size + i] = imgData[i * 4 + c] / 255;
      }
    }
    return new ort.Tensor("float32", float, [1, 3, size, size]);
  }

  // 4. Main AI Loop (Safe Check Included)
  async function loop() {
    // 1. เช็คความพร้อมพื้นฐาน
    if (!videoRef.current || !canvasRef.current || !cvRef.current || !isCameraOpen) return;

    // 2. เช็คว่าวิดีโอโหลดข้อมูลภาพเสร็จหรือยัง (ต้อง > 0 และ readyState >= 2)
    if (videoRef.current.videoWidth === 0 || videoRef.current.readyState < 2) {
        loopRef.current = requestAnimationFrame(loop);
        return;
    }

    const cv = cvRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    
    // ตั้งค่าขนาด canvas ให้เท่ากับวิดีโอที่เข้ามาจริง
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.drawImage(videoRef.current, 0, 0);

    try {
      const src = cv.imread(canvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      const faces = new cv.RectVector();
      faceCascadeRef.current.detectMultiScale(gray, faces, 1.1, 3, 0);

      if (faces.size() > 0) {
        const r = faces.get(0);
        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = r.width; faceCanvas.height = r.height;
        faceCanvas.getContext("2d")!.drawImage(canvas, r.x, r.y, r.width, r.height, 0, 0, r.width, r.height);

        const input = preprocess(faceCanvas);
        const feeds: any = {}; feeds[sessionRef.current!.inputNames[0]] = input;
        const out = await sessionRef.current!.run(feeds);
        const logits = out[sessionRef.current!.outputNames[0]].data as Float32Array;
        
        const exps = logits.map(v => Math.exp(v - Math.max(...logits)));
        const probs = exps.map(v => v / exps.reduce((a, b) => a + b, 0));
        const maxIdx = probs.indexOf(Math.max(...probs));
        const detectedEmotion = classesRef.current![maxIdx] || "Unknown";

        setEmotion(detectedEmotion);
        setConf(probs[maxIdx]);

        const color = EMOTION_COLORS[detectedEmotion.toLowerCase()] || "#FFFFFF";
        ctx.strokeStyle = color;
        ctx.lineWidth = 5;
        ctx.strokeRect(r.x, r.y, r.width, r.height);
        
        ctx.fillStyle = color;
        ctx.fillRect(r.x, r.y - 35, 160, 35);
        ctx.fillStyle = (color === "#FFFF00" || color === "#FFC0CB") ? "black" : "white";
        ctx.font = "bold 20px sans-serif";
        ctx.fillText(`${detectedEmotion} ${(probs[maxIdx]*100).toFixed(0)}%`, r.x + 8, r.y - 10);
      }

      src.delete(); gray.delete(); faces.delete();
    } catch (e) {
      console.error(e);
    }

    loopRef.current = requestAnimationFrame(loop);
  }

  return (
    <main className="min-h-screen bg-zinc-950 text-white p-4 md:p-10 flex flex-col md:flex-row gap-10">
      {/* ส่วนซ้าย: Gallery */}
      <div className="w-full md:w-80 flex flex-col gap-4">
        <div className="flex justify-between items-center border-b border-zinc-800 pb-4">
          <h2 className="text-xl font-black italic">GALLERY ({capturedImages.length}/5)</h2>
          <button onClick={resetGallery} className="text-[10px] bg-red-600 px-3 py-1 rounded-full font-bold hover:bg-red-500">RESET</button>
        </div>
        <div className="flex flex-col gap-4 overflow-y-auto max-h-[70vh] custom-scrollbar pr-2">
          {capturedImages.map(img => (
            <div key={img.id} className="bg-zinc-900 rounded-2xl overflow-hidden border-l-[6px] shadow-2xl animate-in" style={{ borderColor: img.color }}>
              <img src={img.src} className="w-full h-32 object-cover" alt="Captured" />
              <div className="p-3 flex justify-between items-center bg-zinc-900/80">
                <span className="font-black text-xs uppercase" style={{ color: img.color }}>{img.emotion}</span>
                <span className="text-[10px] bg-zinc-800 px-2 py-1 rounded-md">{(img.conf * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))}
          {capturedImages.length === 0 && <div className="h-40 border-2 border-dashed border-zinc-800 rounded-3xl flex items-center justify-center text-zinc-600 font-bold italic uppercase">Empty</div>}
        </div>
      </div>

      {/* ส่วนขวา: กล้อง */}
      <div className="flex-1 flex flex-col items-center">
        <div className="relative w-full max-w-4xl aspect-video bg-zinc-900 rounded-[3rem] overflow-hidden border-8 border-zinc-900 shadow-2xl">
          <video ref={videoRef} className="hidden" playsInline muted />
          <canvas ref={canvasRef} className={`w-full h-full object-cover transition-opacity duration-700 ${!isCameraOpen ? 'opacity-0' : 'opacity-100'}`} />
          
          {!isCameraOpen && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950">
               <div className="w-24 h-24 bg-zinc-900 rounded-full flex items-center justify-center mb-6 animate-pulse">
                 <span className="text-5xl">📷</span>
               </div>
               <p className="text-zinc-600 font-black tracking-widest uppercase text-sm">System Ready</p>
            </div>
          )}

          {/* ปุ่มถ่ายรูปในกรอบหน้า */}
          {isCameraOpen && (
            <button 
              onClick={capturePhoto} 
              disabled={capturedImages.length >= 5}
              className="absolute bottom-10 right-10 w-20 h-20 bg-white/10 backdrop-blur-xl border-4 border-white/50 rounded-full flex items-center justify-center hover:scale-110 active:scale-90 transition-all shadow-2xl disabled:opacity-10 group"
            >
              <div className="w-14 h-14 bg-white rounded-full group-hover:bg-zinc-200" />
            </button>
          )}
        </div>

        {/* ปุ่มควบคุมล่างกลาง (เปิด-ปิด) */}
        <div className="mt-12 flex flex-col items-center gap-6">
          <button 
            onClick={toggleCamera} 
            className={`w-28 h-28 rounded-full flex items-center justify-center shadow-2xl transition-all hover:scale-105 active:scale-90 ${isCameraOpen ? 'bg-red-600' : 'bg-emerald-500'}`}
          >
            {isCameraOpen ? (
              <div className="w-10 h-10 bg-white rounded-xl shadow-lg" /> /* สี่เหลี่ยม */
            ) : (
              <div className="w-0 h-0 border-t-[22px] border-t-transparent border-l-[38px] border-l-white border-b-[22px] border-b-transparent ml-2 drop-shadow-lg" /> /* สามเหลี่ยม */
            )}
          </button>
          <p className="text-zinc-400 font-bold italic text-sm uppercase tracking-widest">{status}</p>
        </div>
      </div>
    </main>
  );
}