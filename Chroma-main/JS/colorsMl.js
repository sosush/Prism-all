'use strict';

// ==========================================
// 1. VISUALIZER & BACKGROUND LOGIC
// ==========================================
const work = document.querySelector(".workspace");
const colored = [
  "rgba(255, 0, 0, 0.8)",    // RED
  "rgba(0, 255, 255, 0.8)",   // CYAN (Blue-ish)
  "rgba(255, 255, 255, 0.8)", // WHITE
  "rgba(0, 255, 0, 0.8)"      // GREEN
];

let colorIndex = 0;
// We will track the current color to send to backend later
let currentScreenColor = "RED"; 

setInterval(() => {
  if (work) {
    work.style.setProperty("--workspace-bg", colored[colorIndex]);
    
    // Map index to color name for the backend
    if (colorIndex === 0) currentScreenColor = "RED";
    else if (colorIndex === 1) currentScreenColor = "BLUE";
    else if (colorIndex === 2) currentScreenColor = "WHITE";
    else if (colorIndex === 3) currentScreenColor = "GREEN";

    colorIndex = (colorIndex + 1) % colored.length;
  }
}, 2000); // Changes color every 2 seconds

// ==========================================
// 2. MAIN VERIFICATION LOGIC
// ==========================================

let verificationStatus = "IDLE"; // IDLE, RECORDING, PROCESSING, VERIFIED, FAILED
let mediaRecorder;
let recordedChunks = [];

document.addEventListener("DOMContentLoaded", () => {
  const overlay = document.getElementById("overlay");
  const mainContent = document.getElementById("mainContent");
  const audio = document.getElementById("audio");
  const visualizerContainer = document.querySelector(".visualizer-container");
  const calibrationText = document.querySelector(".calibration-text");
  const spinner = document.getElementById("loadingSpinner");
  const statusText = document.getElementById("uploadStatus"); // If you have a status box
  
  // --- A. START SEQUENCE (Clicked "Start") ---
  if(overlay) {
      overlay.addEventListener("click", async () => {
        // 1. UI Transition
        overlay.style.opacity = "0";
        overlay.style.pointerEvents = "none";
        if(mainContent) mainContent.classList.remove("waiting-to-start");
        if(visualizerContainer) visualizerContainer.style.opacity = "1";
        if (audio) audio.play().catch(e => console.log("Audio error:", e));

        // 2. Start Webcam & Record
        await startLiveCapture();
        
        // 3. Start UI Text Animation
        runCalibrationSequence();
      });
  }

  // --- B. WEBCAM & RECORDING LOGIC ---
  async function startLiveCapture() {
    try {
        console.log("📷 Requesting Camera...");
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        
        // Create an invisible video element to keep the stream active
        let videoElement = document.getElementById("webcam");
        if(!videoElement) {
            videoElement = document.createElement("video");
            videoElement.id = "webcam";
            videoElement.style.display = "none"; 
            document.body.appendChild(videoElement);
        }
        videoElement.srcObject = stream;
        videoElement.play();

        // Start Recording
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
        recordedChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) recordedChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            console.log("🎥 Recording Stopped. Processing...");
            const blob = new Blob(recordedChunks, { type: "video/webm" });
            
            // Convert Blob to File (Simulating the file upload)
            const videoFile = new File([blob], "live_capture.webm", { type: "video/webm" });
            
            // AUTO-SEND TO JAVA
            uploadToPrism(videoFile);
        };

        mediaRecorder.start();
        verificationStatus = "RECORDING";
        console.log("🔴 Recording Started...");

        // STOP AUTOMATICALLY AFTER 9 SECONDS
        setTimeout(() => {
            if(mediaRecorder.state === "recording") {
                mediaRecorder.stop();
                // Stop the camera stream
                stream.getTracks().forEach(track => track.stop());
            }
        }, 9000); // 9 Seconds match your color cycle

    } catch (err) {
        console.error("Camera Error:", err);
        alert("Camera access denied or missing!");
        verificationStatus = "FAILED";
    }
  }

  // --- C. BACKEND UPLOAD (The "Bridge") ---
  async function uploadToPrism(file) {
    verificationStatus = "PROCESSING";
    console.log("🚀 Sending Video to Java Backend...");
    
    const userWallet = localStorage.getItem("walletAddress") || "0xUNKNOWN_WALLET";
    const formData = new FormData();
    formData.append("video", file);
    formData.append("wallet", userWallet);
    // Send the color sequence logic (or just "RED" if simple)
    formData.append("screenColor", "RED"); 

    try {
        const response = await fetch("http://localhost:8080/api/video/upload", {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            const text = await response.text();
            console.log("✅ Backend Response:", text);
            
            // Check for success keywords in the response
            if (text.toUpperCase().includes("VERIFIED") || text.includes("0x")) {
                verificationStatus = "VERIFIED";
            } else {
                verificationStatus = "FAILED";
            }
        } else {
            console.error("Server Error:", response.status);
            verificationStatus = "FAILED";
        }
    } catch (error) {
        console.error("Network Error:", error);
        verificationStatus = "FAILED";
    }
  }

  // --- D. UI ANIMATION LOOP (Smart Polling) ---
  function runCalibrationSequence() {
    if(spinner) spinner.style.display = "block";
    if(calibrationText) calibrationText.innerText = "CALIBRATING OPTICS...";

    // This loop checks the status variable every 1 second
    const poller = setInterval(() => {
        if(verificationStatus === "RECORDING") {
            if(calibrationText) calibrationText.innerText = "ANALYZING BIOMETRICS...";
        }
        else if (verificationStatus === "PROCESSING") {
            if(calibrationText) calibrationText.innerText = "VERIFYING ON BLOCKCHAIN...";
        }
        else if (verificationStatus === "VERIFIED") {
            clearInterval(poller);
            showSuccess();
        } 
        else if (verificationStatus === "FAILED") {
            clearInterval(poller);
            showFailure();
        }
    }, 1000);
  }

  function showSuccess() {
    if(spinner) spinner.style.display = "none";
    if(calibrationText) {
        calibrationText.innerText = "VERIFIED HUMAN";
        calibrationText.style.color = "#00ffcc";
        calibrationText.style.textShadow = "0 0 15px #00ffcc";
    }
  }

  function showFailure() {
    if(spinner) spinner.style.display = "none";
    if(calibrationText) {
        calibrationText.innerText = "NOT VERIFIED";
        calibrationText.style.color = "#ff0055";
        calibrationText.style.textShadow = "0 0 15px #ff0055";
    }
  }
});