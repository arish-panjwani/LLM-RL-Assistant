This folder contains the **mobile-first web application** used for interacting with the LLM-RL-Assistant.

## Features
- Accepts **text, voice, and image inputs**
- Displays responses from the LLM (via Raspberry Pi backend)
- Allows users to provide feedback (thumbs up/down or rating)
- Shows metrics such as clarity score, hallucination alerts, etc.

## Files
- `index.html` – Main UI entry point
- `app.js` – Handles user input/output logic
- `styles.css` – UI styling (mobile responsive)
- `websocket.js` – Connects to Raspberry Pi for real-time feedback

## Image Input
- Users can upload or capture an image
- Image is sent via REST API to the backend
- Result is displayed as a generated caption or contextual reply

## To Run
Open `index.html` in a browser on a smartphone (served over local Wi-Fi).
