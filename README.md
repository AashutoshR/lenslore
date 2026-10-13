# 🔎 LensLore

Upload an image, ask anything about it. LensLore is a small multimodal app built
with Streamlit: it sends your image and your question to a vision model together,
and streams back an answer grounded in what the image shows.

It runs for free two ways: on your own computer through Ollama (private, no
limits), or on Google Gemini's free tier (nothing to install, daily limits).
OpenAI's models, the successors to GPT-4V, are there too if you have a paid key.

## What it does

- **Four ways to add an image**: upload a file, pick a bundled sample, give a public
  image URL, or paste/drop an image straight into the message box.
- **Conversation, not one-shot**: ask follow-up questions about the same image.
- **One-click prompts**: Describe, Read text (OCR), Explain data (charts/tables), Alt text.
- **Controls in the sidebar**: provider, model, answer style.
- **Streaming answers**, each labelled with the model that gave it and how long it took.
- **Plain-language errors** (bad key, quota, no connection) and light and dark themes.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

Then pick one of the free options (or set up both):

- **Ollama, on your computer.** Install it from <https://ollama.com/download>, then
  download a vision model: `ollama pull gemma4:e4b` (about 6.6 GB). No key needed.
- **Gemini, in the cloud.** Create a free key at <https://aistudio.google.com/apikey>
  (sign in with a Google account; no billing details needed) and put it in `.env`
  as `GEMINI_API_KEY`.

Start the app:

```bash
streamlit run app.py
```

On Windows you can instead double-click `run.bat`. Either way, keep that window
open: if the server stops, the page shows a "Connection error".

The app opens at <http://localhost:8501>. Without a `.env`, you can paste the key
into the sidebar instead; it is kept for that browser session only.

## Providers and cost

| Provider | Cost | Runs | Key | Model |
|---|---|---|---|---|
| Gemini | Free tier | Google's cloud | `GEMINI_API_KEY` | `gemini-3-flash-preview` |
| Ollama | Free | This computer | none | any installed vision model |
| OpenAI | Paid per request | OpenAI's cloud | `OPENAI_API_KEY` | `gpt-4o` |

LensLore selects Ollama automatically once it finds a vision model installed there;
otherwise it starts on Gemini.

**Ollama.** The model runs on your own machine, so there are no quotas, no "high
demand" errors, and your images never leave the computer. The cost is disk space
and hardware: `gemma4:e4b` suits a GPU with 8 GB of memory, `gemma4:e2b` is a
lighter choice for less, and it also works without a GPU, only slower. The first
question after starting takes longer while the model loads. Any vision model you
pull (for example `llama3.2-vision`) shows up in the Model box; models that cannot
read images are left out. Ollama cannot fetch images from the web, so the URL
option is hidden while it is selected.

**Gemini free tier.** You are never charged unless you add billing to the Google
project yourself. In exchange there are rate limits (a handful of requests per
minute and a daily cap, which vary by model and are shown in Google AI Studio), and
Google may use free-tier content to improve its products, so don't upload anything
private.

Free models are also sometimes overloaded ("experiencing high demand"). When the
model you picked is busy or out of quota, LensLore automatically moves on to the
next Gemini model in the list and tells you it did. You only see an error if every
one of them is unavailable.

**OpenAI.** GPT-4V was the name of OpenAI's first vision model
(`gpt-4-vision-preview`), since retired; vision is now built into `gpt-4o` and
later models. OpenAI's API has no free tier.

Model names change often. The Model box accepts any id, so if a listed model has
been retired, type a current one. All three providers are called through the same
OpenAI-format Chat Completions API (Gemini and Ollama both offer a compatible
endpoint), which is why adding another is a few lines in `lenslore/providers.py`.

## How it works

```
app.py                  Streamlit UI: image picker, sidebar, chat
lenslore/prompts.py     System prompt, answer styles, one-click prompts
lenslore/providers.py   Provider endpoints, key names and model suggestions
lenslore/ollama.py      Finds the vision models installed in Ollama
lenslore/vision.py      Image preparation, message building, the model call
lenslore/markdown.py    Keeps prices like "$5" from being rendered as maths
.streamlit/config.toml  Theme (colours, fonts) for light and dark mode
scripts/make_samples.py Regenerates the images in samples/
tests/                  Unit tests + UI tests with the model stubbed out
```

1. **Prepare the image** (`prepare_image`). The upload is rotated according to its
   EXIF data, flattened onto white if it has transparency, shrunk so its longest
   side is at most 2048 px, and encoded as a base64 JPEG data URL. The model
   downsizes larger images anyway, so this only saves upload time.
2. **Build the messages** (`build_messages`). A multimodal message is a list of
   parts: a `text` part and an `image_url` part. The image is attached to the
   first question only; follow-ups are plain text, and the model still sees the
   image because the whole conversation is re-sent each turn.
3. **Call the model** (`stream_answer`). The Chat Completions API is called with
   `stream=True` at the chosen provider's endpoint, and the answer is drawn on
   the page as it arrives.

### Prompt design

Everything the model is told lives in `lenslore/prompts.py`:

- The **system prompt** makes the model ground claims in what is visible, admit
  when something is unreadable instead of guessing, separate observation from
  inference, and transcribe text and numbers exactly.
- An **answer style** (Balanced, Concise, Detailed, Simple) is appended to it.
- The **one-click prompts** are ordinary user questions, worded precisely.

Try editing a rule and asking the same question again to see how the answer shifts.

### Image detail (OpenAI only)

`low` gives the model a small fixed-size view: fast and cheap, fine for "what is
this?". `high` adds tiled close-ups, which is what you want for small text, receipts
and dense charts. `auto` lets the model choose.

## Tests

```bash
pip install pytest
pytest
```

The tests never call a real model: the UI tests replace the model call with a stub.

## Deploying to Render

The repository includes a Render Blueprint (`render.yaml`), so there is nothing to
configure by hand.

1. Push this folder to a GitHub repository (`.env` is git-ignored; keep it that way).
2. At <https://dashboard.render.com>, choose **New → Blueprint** and pick the repo.
3. When asked for `GEMINI_API_KEY`, paste your Gemini key, then apply.

Render builds the app and gives it a public `onrender.com` address. On the free
plan the app goes to sleep after 15 minutes without visitors, and the next visit
takes about a minute to wake it. Every push to the repository redeploys it.

Streamlit Community Cloud (<https://share.streamlit.io>) also works: create an app
from the repo with `app.py` as the entry point and add `GEMINI_API_KEY` under
**Advanced settings → Secrets**.

Anyone who can open a deployed app uses your key's quota. On the Gemini free tier
that costs nothing but can exhaust your daily limit; alternatively leave the secret
out and let visitors paste their own key into the sidebar.

Ollama only works when the app runs on the same computer as Ollama, so a deployed
copy offers Gemini and OpenAI only (`render.yaml` sets `LENSLORE_HOSTED` to hide it).

## Privacy

With Ollama, images never leave your computer. With Gemini or OpenAI, they are sent
to that company for analysis; for URL images, the provider fetches the image
directly from the address you give. LensLore itself stores nothing.
