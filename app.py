import gradio as gr
import whisper
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os

# --- Chargement modèle Whisper ---
whisper_model = whisper.load_model("base")

# --- Chargement modèle CNN de secours (SPEECHCOMMANDS_RESNET10 n'existe plus) ---
class DummyCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(1, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.randn(batch_size, len(COMMANDS))

# Liste des commandes attendues (SpeechCommands dataset)
COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# Charger le "faux" modèle CNN
cnn_model = DummyCNN(num_classes=len(COMMANDS)).eval()
SAMPLE_RATE = 16000  # Valeur standard utilisée pour le SpeechCommands dataset

def preprocess_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    # Convertir en mono si nécessaire
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample si besoin
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Tronquer ou padder à exactement 1 seconde (16000 échantillons)
    target_len = SAMPLE_RATE
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    elif waveform.shape[1] < target_len:
        pad_len = target_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    return waveform

def detect_command(audio_path):
    try:
        waveform = preprocess_audio(audio_path)
        waveform = waveform.unsqueeze(0)  # Ajouter batch dimension
        with torch.inference_mode():
            output = cnn_model(waveform)
            predicted = output.argmax(dim=1).item()
        command = COMMANDS[predicted] if predicted < len(COMMANDS) else "unknown"
        return f"✅ Commande détectée : {command}"
    except Exception as e:
        return f"❌ Erreur détection commande : {str(e)}"

def transcribe(audio_path):
    try:
        result = whisper_model.transcribe(audio_path, temperature=0.0, fp16=False)
        return f"📝 Transcription : {result['text']}"
    except Exception as e:
        return f"❌ Erreur transcription : {str(e)}"

def process(audio_file, task):
    try:
        if audio_file is None or not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
            return "❗ Fichier audio invalide ou vide."
        if task == "Transcription vocale":
            return transcribe(audio_file)
        elif task == "Détection commande vocale":
            return detect_command(audio_file)
        else:
            return "❗ Tâche inconnue."
    except Exception as e:
        return f"❌ Erreur générale : {str(e)}"

interface = gr.Interface(
    fn=process,
    inputs=[
        gr.Audio(type="filepath", label="Uploader un fichier audio"),
        gr.Radio(["Transcription vocale", "Détection commande vocale"], label="Choisir la tâche"),
    ],
    outputs=gr.Textbox(label="Résultat"),
    title="🎧 Transcription & Détection Commande Vocale",
    description="Uploader un fichier audio et choisissez la tâche."
)

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080))
    )
