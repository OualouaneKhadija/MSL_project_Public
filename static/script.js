const videoFeed = document.querySelector("#videoFeed");
const landmarksToggle = document.querySelector("#landmarksToggle");
const voiceToggle = document.querySelector("#voiceToggle");
const predictionText = document.querySelector("#predictionText");
const confidenceText = document.querySelector("#confidenceText");
const statusText = document.querySelector("#status");
const languageSelect = document.querySelector("#languageSelect");

let lastPlayedLabel = null;
let lastPlayedTime = 0;
let currentAudio = null;

function refreshVideoFeed() {
    const landmarks = landmarksToggle.checked ? "true" : "false";
    const voice = voiceToggle.checked ? "true" : "false";

    videoFeed.src = `/video_feed?landmarks=${landmarks}&voice=${voice}&time=${Date.now()}`;
}

async function playAudio(label, prediction) {
    if (!voiceToggle.checked || !label) return;

    const now = Date.now();
    if (label === lastPlayedLabel && now - lastPlayedTime < 2500) return;

    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }

    const lang = languageSelect.value;

    if (lang === "darija") {
        currentAudio = new Audio(`/static/audio/${label}.mp3`);
        currentAudio.preload = "auto";
        currentAudio.volume = 1.0;
        currentAudio.play().catch((e) => console.error("Audio playback error:", e));
    } else {
        try {
            const res = await fetch(`/translate_speak?label=${label}&language=${lang}`);
            if (res.ok) {
                const url = URL.createObjectURL(await res.blob());
                currentAudio = new Audio(url);
                currentAudio.volume = 1.0;
                currentAudio.play().catch((e) => console.error("Audio playback error:", e));
            }
        } catch (e) {
            console.error("Translation error:", e);
        }
    }

    lastPlayedLabel = label;
    lastPlayedTime = now;
}

async function refreshPrediction() {
    try {
        const res = await fetch("/prediction");
        const data = await res.json();

        if (data.prediction) {
            const lang = languageSelect.value;

            if (lang === "darija") {
                predictionText.textContent = data.prediction;
            } else {
                try {
                    const tr = await fetch(`/get_translation?label=${data.label}&language=${lang}`);
                    const trData = await tr.json();
                    predictionText.textContent = trData.text || data.prediction;
                } catch {
                    predictionText.textContent = data.prediction;
                }
            }

            confidenceText.textContent = `Confiance : ${Math.round(data.confidence * 100)}%`;
            statusText.textContent = "Reconnaissance active";

            playAudio(data.label, data.prediction);
        }
    } catch {
        statusText.textContent = "Connexion perdue";
    }
}

landmarksToggle.addEventListener("change", refreshVideoFeed);

voiceToggle.addEventListener("change", () => {
    refreshVideoFeed();
    if (!voiceToggle.checked && currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }
});

languageSelect.addEventListener("change", () => {
    lastPlayedLabel = null;
    lastPlayedTime = 0;
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }
});

setInterval(refreshPrediction, 800);

if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/static/service-worker.js");
}
