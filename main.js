
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");

let camStream = null;
let interval = null;


startBtn.onclick = async () => {
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      }
    });

    video.srcObject = camStream;

    video.onloadedmetadata = () => {
      
      video.width = 640;
      video.height = 480;

      canvas.width = 640;
      canvas.height = 480;
    };

    statusEl.innerText = "Webcam Running";

    interval = setInterval(sendFrame, 500);

  } catch (err) {
    console.error(err);
    statusEl.innerText = "Camera Error";
  }
};

stopBtn.onclick = () => {
  clearInterval(interval);

  if (camStream) {
    camStream.getTracks().forEach(t => t.stop());
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  statusEl.innerText = "Stopped";
};

function sendFrame() {
  if (!video.videoWidth) return;

  ctx.drawImage(video, 0, 0, 640, 480);

  const img = canvas.toDataURL("image/jpeg");

  fetch("http://127.0.0.1:5001/predict-face", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: img })
  })
  .then(r => r.json())
  .then(data => {
    ctx.clearRect(0, 0, 640, 480);
    ctx.drawImage(video, 0, 0, 640, 480);

    data.forEach(f => {
      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;

      ctx.strokeRect(f.x, f.y, 100, 100);

      ctx.fillStyle = "red";
      ctx.font = "18px Arial";
      ctx.fillText(f.emotion, f.x, f.y - 10);
    });
  })
  .catch(err => console.error("Face Error:", err));
}


let mediaRecorder;
let audioChunks = [];

const recordBtn = document.getElementById("recordBtn");
const voiceText = document.getElementById("voiceText");
const voiceResult = document.getElementById("voiceResult");

recordBtn.onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    startWaveform(stream);

    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.start();
    recordBtn.innerText = "Recording...";

    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    setTimeout(() => {
      mediaRecorder.stop();
      recordBtn.innerText = "🎤 Start Recording";
    }, 4000);

    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, { type: "audio/webm" });

      const formData = new FormData();
      formData.append("audio", blob);

      try {
        const res = await fetch("http://127.0.0.1:5001/predict-voice", {
          method: "POST",
          body: formData
        });

        const data = await res.json();

        if (data.error) {
          voiceText.innerText = "Error: " + data.error;
          voiceResult.innerText = "";
          return;
        }

        voiceText.innerText = "You said: " + data.text;
        voiceResult.innerText = `Emotion: ${data.emotion} (${data.confidence}%)`;

        updateUIEmotion(data.emotion);
        updateChart(data.emotion);

      } catch (err) {
        console.error("Voice Error:", err);
        voiceText.innerText = "Server error";
      }
    };

  } catch (err) {
    console.error("Mic Error:", err);
    voiceText.innerText = "Microphone access denied";
  }
};



function startWaveform(stream) {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const analyser = audioCtx.createAnalyser();
  const source = audioCtx.createMediaStreamSource(stream);

  source.connect(analyser);
  analyser.fftSize = 256;

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  const canvas = document.getElementById("waveform");
  const ctx = canvas.getContext("2d");

  function draw() {
    requestAnimationFrame(draw);

    analyser.getByteFrequencyData(dataArray);

    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      let barHeight = dataArray[i];
      ctx.fillStyle = "lime";
      ctx.fillRect(x, canvas.height - barHeight / 2, 3, barHeight / 2);
      x += 4;
    }
  }

  draw();
}



function updateUIEmotion(emotion) {
  const body = document.body;

  if (emotion === "Happy") body.style.background = "#d4f8d4";
  else if (emotion === "Sad") body.style.background = "#d6e4ff";
  else if (emotion === "Angry") body.style.background = "#ffd6d6";
  else body.style.background = "#f5f7fa";
}


const textInput = document.getElementById("textInput");
const textResult = document.getElementById("textResult");

let timer;

textInput.addEventListener("input", () => {
  clearTimeout(timer);

  timer = setTimeout(() => {
    const text = textInput.value.trim();

    if (!text) {
      textResult.innerText = "";
      return;
    }

    fetch("http://127.0.0.1:5001/predict-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    })
    .then(r => r.json())
    .then(d => {
      if (!d.error) {
        textResult.innerText = `Emotion: ${d.emotion} (${d.confidence}%)`;

        updateUIEmotion(d.emotion);
        updateChart(d.emotion);
      } else {
        textResult.innerText = "Error detecting text emotion";
      }
    })
    .catch(() => {
      textResult.innerText = "Server error";
    });

  }, 600);
});



let emotionData = {
  Happy: 0,
  Sad: 0,
  Angry: 0,
  Neutral: 0
};

const chart = new Chart(
  document.getElementById("emotionChart"),
  {
    type: "bar",
    data: {
      labels: ["Happy", "Sad", "Angry", "Neutral"],
      datasets: [{
        label: "Emotion Count",
        data: [0, 0, 0, 0]
      }]
    }
  }
);

function updateChart(emotion) {
  if (emotionData[emotion] !== undefined) {
    emotionData[emotion]++;
  }

  chart.data.datasets[0].data = [
    emotionData.Happy,
    emotionData.Sad,
    emotionData.Angry,
    emotionData.Neutral
  ];

  chart.update();
}

const finalBtn = document.getElementById("checkFinalBtn");
const finalEmotionEl = document.getElementById("finalEmotion");
const finalDetailsEl = document.getElementById("finalDetails");

finalBtn.onclick = async () => {
  try {
    const res = await fetch("http://127.0.0.1:5001/final-emotion");
    const data = await res.json();

    if (data.error) {
      finalEmotionEl.innerText = "No data yet!";
      finalDetailsEl.innerText = "";
      return;
    }

    finalEmotionEl.innerText =
      `Final Emotion: ${data.final_emotion} (${data.confidence}%)`;

    finalDetailsEl.innerText =
      `Face: ${data.details.face || "N/A"} | ` +
      `Text: ${data.details.text || "N/A"} | ` +
      `Voice: ${data.details.voice || "N/A"}`;

  } catch (err) {
    console.error(err);
    finalEmotionEl.innerText = "Error fetching result";
  }
};


const imageUpload = document.getElementById("imageUpload");
const audioUpload = document.getElementById("audioUpload");
const uploadText = document.getElementById("uploadText");
const uploadResult = document.getElementById("uploadResult");

let debounceTimer;


function autoAnalyze() {
  clearTimeout(debounceTimer);

  debounceTimer = setTimeout(async () => {
    const imageFile = imageUpload.files[0];
    const audioFile = audioUpload.files[0];
    const text = uploadText.value.trim();

    
    if (!imageFile && !audioFile && !text) {
      uploadResult.innerText = "";
      return;
    }

    const formData = new FormData();

    if (imageFile) formData.append("image", imageFile);
    if (audioFile) formData.append("audio", audioFile);
    if (text) formData.append("text", text);

    try {
      
      uploadResult.innerText = "🔍 AI is analyzing emotions...";

      const res = await fetch("http://127.0.0.1:5001/predict-all", {
        method: "POST",
        body: formData
      });

      const data = await res.json();

      if (data.error) {
        uploadResult.innerText = "❌ Error: " + data.error;
        return;
      }

      
      uploadResult.innerText =
        `🎯 Final Emotion: ${data.final_emotion} (${data.confidence}%)\n\n` +
        `📸 Face: ${data.face || "N/A"}\n` +
        `💬 Text: ${data.text || "N/A"}\n` +
        `🎤 Voice: ${data.voice || "N/A"}`;

    } catch (err) {
      console.error("Upload Error:", err);
      uploadResult.innerText = "❌ Server Error";
    }

  }, 800); 
}

const textFileUpload = document.getElementById("textFileUpload");


textFileUpload.addEventListener("change", () => {
  const file = textFileUpload.files[0];

  if (!file) return;

  const reader = new FileReader();

  reader.onload = function (e) {
    uploadText.value = e.target.result;

    
    autoAnalyze();
  };

  reader.readAsText(file);
});

imageUpload.addEventListener("change", () => {
  const file = imageUpload.files[0];

  if (file) {
    const url = URL.createObjectURL(file);
    document.getElementById("imagePreview").src = url;
    document.getElementById("imagePreview").style.display = "block";
  }

  autoAnalyze();
});


audioUpload.addEventListener("change", () => {
  const file = audioUpload.files[0];

  if (file) {
    document.getElementById("audioName").innerText = file.name;
  }

  autoAnalyze();
});



textFileUpload.addEventListener("change", () => {
  const file = textFileUpload.files[0];

  if (!file) return;

  const reader = new FileReader();

  reader.onload = function (e) {
    uploadText.value = e.target.result;
    autoAnalyze();
  };

  reader.readAsText(file);
});

imageUpload.addEventListener("change", autoAnalyze);
audioUpload.addEventListener("change", autoAnalyze);
uploadText.addEventListener("input", autoAnalyze);