import asyncio
import threading
import re
import numpy as np
import sounddevice as sd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)
from kokoro import KPipeline

# ————————————
# 1) ChatPipeline with streaming
# ————————————
class ChatPipeline:
    def __init__(self, model_name: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype  = torch.float16 if device.type=="cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device.type=="cuda" else None
        )
        if device.type != "cuda":
            self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device    = device
        self.messages  = [{
            "role": "system",
            "content": "You are a helpful assistant. Please think step by step."
        }]

    def chat_stream(self, user_prompt: str, max_new_tokens: int = 250):
        self.messages.append({"role": "user", "content": user_prompt})
        prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # kick off generation in a thread
        threading.Thread(
            target=self.model.generate,
            kwargs={**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens},
            daemon=True
        ).start()

        full = ""
        for chunk in streamer:
            full += chunk
            yield chunk
        self.messages.append({"role": "assistant", "content": full})


# ————————————
# 2) Async TTS + playback
# ————————————
tts = KPipeline(lang_code="a")
VOICE = "bm_daniel"

async def tts_worker(text_q: asyncio.Queue, audio_q: asyncio.Queue):
    loop = asyncio.get_running_loop()
    while True:
        text = await text_q.get()
        if text is None:
            # signal playback end
            await audio_q.put(None)
            break

        # generate audio in threadpool
        def gen_audio():
            chunks = []
            for _g, _p, audio in tts(text, voice=VOICE, speed=1, split_pattern=r'\n+'):
                chunks.append(audio)
            return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

        audio = await loop.run_in_executor(None, gen_audio)
        await audio_q.put(audio)

async def playback_worker(audio_q: asyncio.Queue):
    loop = asyncio.get_running_loop()
    while True:
        audio = await audio_q.get()
        if audio is None:
            break
        # play without blocking loop
        await loop.run_in_executor(None, sd.play, audio, 24000)
        await loop.run_in_executor(None, sd.wait)


# ————————————
# 3) Orchestrator: stream → batch on punctuation → TTS → play
# ————————————
async def interactive_chat_async(prompt: str):
    text_q  = asyncio.Queue()
    audio_q = asyncio.Queue()

    # start TTS & playback tasks
    tts_task  = asyncio.create_task(tts_worker(text_q, audio_q))
    play_task = asyncio.create_task(playback_worker(audio_q))

    loop = asyncio.get_event_loop()

    # run LLM streaming in a thread, pushing batches into text_q
    def produce():
        buffer = ""
        for chunk in chat_pipe.chat_stream(prompt, max_new_tokens=20000):
            print(chunk, end="", flush=True)
            buffer += chunk

            # flush whenever punctuation appears
            while True:
                m = re.search(r"[,\.\!\?]", buffer)
                if not m:
                    break
                seg = buffer[: m.end() ].strip()
                if seg:
                    asyncio.run_coroutine_threadsafe(text_q.put(seg), loop)
                buffer = buffer[m.end():]

        # flush leftover
        if buffer.strip():
            asyncio.run_coroutine_threadsafe(text_q.put(buffer.strip()), loop)
        # signal end
        asyncio.run_coroutine_threadsafe(text_q.put(None), loop)

    threading.Thread(target=produce, daemon=True).start()

    # wait for both to finish
    await tts_task
    await play_task


# ————————————
# 4) REPL entrypoint
# ————————————
if __name__ == "__main__":
    model_name = "PowerInfer/SmallThinker-3B-Preview"
    chat_pipe  = ChatPipeline(model_name)

    print("Type your questions (or 'exit' to quit).")
    while True:
        user = input("\nYou: ").strip()
        if not user or user.lower() == "exit":
            print("Goodbye!")
            break

        print("Assistant: ", end="", flush=True)
        asyncio.run(interactive_chat_async(user))
