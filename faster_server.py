import uvicorn
from fastapi import FastAPI, UploadFile, Form
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()
loaded_models = {}

@app.post("/transcribe")
async def transcribe(file: UploadFile, model: str = Form("large-v2"), device: str = Form("cpu")):
    try:
        if (model, device) not in loaded_models:
            loaded_models[(model, device)] = WhisperModel(model, device=device, compute_type="int8")
        whisper_model = loaded_models[(model, device)]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        segments, info = whisper_model.transcribe(tmp_path)
        text = ""
        result_segments = []
        for seg in segments:
            seg_dict = {"start": seg.start, "end": seg.end, "text": seg.text}
            result_segments.append(seg_dict)
            text += seg.text.strip() + " "

        os.remove(tmp_path)
        return {"text": text.strip(), "segments": result_segments}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)