import os
import sys
import torch
import torchaudio
import librosa
import numpy as np
import gradio as gr
from transformers import AutoTokenizer

# ----- Définir le chemin absolu vers le projet final
PROJECT_PATH = r"C:\Users\Boumahdi\Documents\hayat\S2\ml\projet final"
sys.path.append(PROJECT_PATH)  # Ajouter ce chemin aux modules python pour importer les modèles locaux

# ----- Import des modèles locaux
from model_trans import SpeechToTextModel  # Import du modèle de transcription personnalisé
from model_cnn import SimpleCNN            # Import du modèle CNN pour la commande vocale

# ----- Constantes
SAMPLE_RATE = 16000  # Fréquence d'échantillonnage pour l'audio
COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']  # Liste des commandes détectées
TRANS_MODEL_PATH = os.path.join(PROJECT_PATH, "model.pth")  # Chemin vers les poids du modèle transcription
CNN_MODEL_PATH = os.path.join(PROJECT_PATH, "simple_cnn_speech_commands.pth")  # Chemin vers les poids du modèle CNN

# ----- Chargement des modèles
def load_models():
    # Charger le tokenizer pré-entraîné de Hugging Face (wav2vec2)
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

    # Initialiser le modèle de transcription avec la taille du vocabulaire du tokenizer
    stt_model = SpeechToTextModel(vocab_size=tokenizer.vocab_size)
    # Charger les poids du modèle transcription sauvegardé localement
    stt_model.load_state_dict(torch.load(TRANS_MODEL_PATH, map_location="cpu"))
    stt_model.eval()  

    # Initialiser le modèle CNN pour la commande vocale avec le nombre de classes
    cnn_model = SimpleCNN(num_classes=len(COMMANDS))
    # Charger les poids du modèle CNN sauvegardé localement
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location="cpu"))
    cnn_model.eval()  

    # Définir la transformation MelSpectrogram utilisée pour extraire les features audio
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=80)

    # Retourner les objets chargés pour utilisation dans la fonction principale
    return tokenizer, stt_model, cnn_model, mel_transform

# Charger les modèles et tokenizer une fois au lancement du script
tokenizer, stt_model, cnn_model, mel_transform = load_models()

# ----- Fonction principale appelée par l'interface Gradio
def process_audio(audio, task_choice):
    # Vérifier qu'un fichier audio a bien été fourni
    if audio is None:
        return "Aucun fichier audio fourni."

    # Charger le fichier audio avec librosa en forcant l'échantillonnage à SAMPLE_RATE
    y, sr = librosa.load(audio, sr=SAMPLE_RATE)

    # Limiter la durée à 10 secondes pour éviter des traitements trop longs
    max_duration = 10
    if len(y) > max_duration * sr:
        y = y[:max_duration * sr]

    # Convertir en tenseur PyTorch avec dimension batch (1, longueur)
    waveform = torch.tensor(y).unsqueeze(0)

    # Selon la tâche choisie par l'utilisateur
    if task_choice == "Transcription":
        # Calculer le spectrogramme Mel (forme : batch=1, 80 bandes Mel, temps)
        mel_spec = mel_transform(waveform)  # (1, 80, T)
      
        with torch.no_grad():  # Pas de calcul de gradients pour l'inférence
            logits = stt_model(mel_spec)  # Sortie brute du modèle
            pred = torch.argmax(logits, dim=-1)[0].tolist()  # Prendre l'index max à chaque pas temporel
            transcription = tokenizer.decode(pred, skip_special_tokens=True)  # Décoder en texte lisible
        return f"📝 Transcription : {transcription}"

    elif task_choice == "Commande vocale":
        # Extraire un spectrogramme Mel classique avec librosa pour la détection commande vocale
        S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=64)
        S_dB = librosa.power_to_db(S, ref=np.max)  # Convertir en dB pour normalisation

        # Ajuster la largeur à une taille fixe (32 frames) pour le CNN
        fixed_width = 32
        if S_dB.shape[1] < fixed_width:
            pad_width = fixed_width - S_dB.shape[1]
            # Compléter avec des zéros à droite si spectrogramme trop court
            S_dB = np.pad(S_dB, ((0, 0), (0, pad_width)), mode='constant')
        elif S_dB.shape[1] > fixed_width:
            # Tronquer si trop long
            S_dB = S_dB[:, :fixed_width]

        # Transformer en tenseur PyTorch avec batch et canal (1,1,64,32)
        S_dB = torch.tensor(S_dB).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            output = cnn_model(S_dB)  # Prédiction brute du CNN
            _, predicted = torch.max(output, 1)  # Classe prédite
            command = COMMANDS[predicted.item()]  # Nom de la commande correspondante
        return f"🎯 Commande détectée : {command}"

    else:
        # Si choix invalide dans l'interface
        return "Choix invalide."

# ----- Interface Gradio : configuration de l'application web
interface = gr.Interface(
    fn=process_audio,  # Fonction appelée à chaque upload/interaction
    inputs=[
        gr.Audio(type="filepath", label="🎧 Fichier audio (wav/mp3/flac/ogg)"),  # Upload audio
        gr.Radio(choices=["Transcription", "Commande vocale"], label="🛠️ Choix de la tâche"),  # Choix de tâche
    ],
    outputs="text",  # Sortie texte simple
    title="🎙️ Détection vocale et transcription",
    description="Choisissez une tâche, puis importez un fichier audio pour obtenir soit la transcription, soit la commande détectée.",
)

# Lancer l'interface 
if __name__ == "__main__":
    interface.launch(share=True)  
