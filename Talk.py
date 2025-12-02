import threading
import tkinter as tk

import sounddevice as sd
from kokoro import KPipeline

# 0. (Optional) Ensure you have GPU‐enabled JAX installed:
#    pip install --upgrade jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_releases.html
#
# Kokoro (Misaki) uses JAX under the hood, so if JAX sees a GPU it’ll automatically
# dispatch there—no extra “.to('cuda')” calls needed on the pipeline.

# 1. Initialize your Kokoro pipeline (American English voice set)
pipeline = KPipeline(lang_code='a')

# 2. Worker that TTSs and plays each chunk
def speak_text():
    raw = text_box.get("1.0", tk.END).strip()
    if not raw:
        return

    # UI feedback
    speak_btn.config(state=tk.DISABLED)
    status_label.config(text="Synthesizing…")
    root.update_idletasks()

    # split on blank lines, send each to the model
    # you can change voice='af_nicole' or any other supported voice
    generator = pipeline(
        raw,
        voice='bm_daniel',
        speed=1,
        split_pattern=r'\n+'
    )

    status_label.config(text="Playing…")
    for i, (_gs, _ps, audio_chunk) in enumerate(generator):
        sd.play(audio_chunk, samplerate=24000)
        sd.wait()

    status_label.config(text="Ready")
    speak_btn.config(state=tk.NORMAL)

# 3. Fire off in a thread so the UI never blocks
def on_speak():
    threading.Thread(target=speak_text, daemon=True).start()

# 4. Build the Tkinter UI
root = tk.Tk()
root.title("Kokoro TTS Demo")

tk.Label(root, text="Enter text to speak:", font=("Segoe UI", 12)).pack(padx=10, pady=(10,0))
text_box = tk.Text(root, height=8, width=60, font=("Segoe UI", 11))
text_box.pack(padx=10, pady=(0,10))

speak_btn = tk.Button(
    root,
    text="🔊 Speak",
    font=("Segoe UI", 12, "bold"),
    command=on_speak
)
speak_btn.pack(pady=5)

status_label = tk.Label(root, text="Ready", font=("Segoe UI", 10, "italic"))
status_label.pack(pady=(0,10))

root.mainloop()
