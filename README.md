# Geo Short Maker V2

An automated video pipeline for generating "Geo Shorts" (YouTube Shorts, TikToks, etc.). The system uses AI to generate scripts, source stock footage/YouTube clips, synthesize voiceovers using Qwen DashScope, and automatically assemble everything into a color-graded, captioned video with Urban Atlas branding.

## Architecture & Pipeline Steps

The pipeline is split into a robust 7-step process:
1. **Script Generation:** Gemini AI takes a topic prompt and outputs a JSON script with beats and visual directives.
2. **Footage Sourcing:** Sources parallel video candidates from YouTube, Pexels, and Pixabay.
3. **Cinematic Fallbacks:** If a beat fails to source, a fallback cinematic clip is generated.
4. **Voiceover Synthesis:** Qwen TTS clones a voice from `voices/` and synthesizes the script.
5. **Audio Mixing:** Mixes the Voiceover, SFX, and Background Music.
6. **Video Assembly:** Cuts clips to beat durations, applies color grading LUTs, and normalizes audio.
7. **Captions & Branding:** Burns synchronized captions using Whisper and applies the Urban Atlas branding/watermark.

## Setup & Installation

1. **Python Dependencies:**
   Ensure you have Python installed, then install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. **System Dependencies:**
   - **FFmpeg:** Required for audio/video processing and assembly. Ensure `ffmpeg` is available in your PATH or configured in `.env`.
   - **yt-dlp:** Required for downloading YouTube B-roll. (Automatically installed via requirements, but you can set a custom path in `.env`).

3. **Environment Variables:**
   The pipeline requires several API keys to function. Ensure your `.env` file is populated with:
   - `GEMINI_API_KEY`: For script and metadata generation.
   - `DASHSCOPE_API_KEY`: For Qwen Voice Cloning.
   - `PEXELS_API_KEY` & `PIXABAY_API_KEY`: For stock footage.
   - `MAPBOX_TOKEN` (optional but recommended): For map-based clips.
   
   See the provided `.env` for the complete list of overrides and paths.

4. **Assets Configuration:**
   - **Voices:** Place your reference `.mp3` files in the `voices/` directory (e.g., `guy_michaels.mp3`, `viral_generic.mp3`). Note that the default `david_attenborough.mp3` might be missing from your fork, so either provide it or override the voice with the `--voice` flag.
   - **Music:** Place background tracks in the `music/` directory.

## Usage

You can test your environment setup and cache using the `--check` flag:
```bash
python -m pipeline --check
```

**Generate a Video:**
To generate a video for a specific topic, simply provide a prompt. The pipeline will handle the rest:
```bash
python -m pipeline "The Hidden Tunnels of Tokyo"
```

**Advanced Arguments:**
- `--voice <name>`: Override the default voice (e.g., `--voice viral`).
- `--resume <run_dir>`: Resume a failed run from its directory (found in `runs_v2/`).
- `--beats <1,2,3>`: Only process specific beat IDs from the script.
