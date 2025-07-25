# 🤖 LLM-RL-Assistant

**LLM-RL-Assistant** is an intelligent conversational assistant that uses **Reinforcement Learning (RL)** to optimize user prompts in real-time before sending them to **Groq LLM APIs**. The system is designed to run the server-side components on a **Raspberry Pi**, while the user interacts via a **smartphone-based web UI**. The phone communicates with the Pi over a local **Wi-Fi network** using API and WebSocket calls.

---

## 🔁 Project Flow

1. **User speaks or types a prompt** into the web app (on smartphone)
2. The prompt is sent via **API call to Raspberry Pi**
3. A selected **RL model (PPO, DDPG, A2C, SAC)** rewrites the prompt for clarity
4. The optimized prompt is forwarded to **Groq LLM API**
5. The response is returned to the phone and shown/spoken to the user
6. The user provides **feedback (thumbs up/down or rating)**
7. Additionally, the system collects **AI-based feedback** such as:
   - Groq’s own self-evaluation of prompt clarity
   - Hallucination checks via fact-verification APIs or cosine similarity
   - Sentiment analysis of the response and user reaction
8. All feedback is converted into a **reward**, helping the RL model learn
9. All interactions and scores are stored and shown in a metrics dashboard

---

## ⚙️ Tech Stack

| Layer            | Technology Used                                      |
|------------------|------------------------------------------------------|
| 📱 UI (Phone)     | HTML/CSS/JS (Web App or PWA, mobile-first)           |
| 📡 Communication | REST API + WebSocket over local Wi-Fi                |
| 🍓 Backend Host   | **Raspberry Pi** (runs Flask/FastAPI + RL Models)   |
| 🧠 RL Models      | PPO, DDPG, A2C, SAC (Stable-Baselines3 / PyTorch)    |
| 🤖 LLM API        | Groq API (via HTTP request from Pi)                  |
| 📊 Metrics        | Cosine similarity, sentiment analysis, feedback logs |
| 🧾 Storage        | SQLite / Supabase / Firebase (for logs + analytics)  |

---

## 👥 Team

| Member       | Role                | Task Summary                               |
|--------------|---------------------|---------------------------------------------|
| Arish        | Frontend Dev        | Chat UI (Mobile Web App)                    |
| Tanzima      | Frontend Dev        | Metrics & Hallucination UI                  |
| Kanika       | API Dev             | Frontend API + WebSocket Integration        |
| Abdullah     | API Dev             | Groq API + Flask Server on Raspberry Pi     |
| Mueez        | Backend Dev         | Model APIs + DB Setup (on Pi)               |
| Riya         | BA & DB Specialist  | Requirements, DB Schema, Coordination       |
| Clifford     | RL Engineer         | PPO Model                                   |
| Indraja      | RL Engineer         | DDPG Model                                  |
| Kauthara     | RL Engineer         | A2C Model                                   |
| SriDatta     | RL Engineer         | SAC Model                                   |
| Ujju         | Prompt Engineer     | Prompt Rewriting Pipeline                   |
| Thejaswi     | Data Analyst        | Evaluation Metrics & Visualization          |

---

## 📱 Deployment Architecture

```plaintext
Smartphone (Browser UI)
     ↓   ↑     [ REST + WebSocket ]
Raspberry Pi (Server: API + RL + Groq)
     ↓   ↑
Groq API (LLM Responses + Self-Evaluation)
```

## 🎯 Project Goals
- Run lightweight server with RL models and Groq integration on Raspberry Pi
- Offer mobile-first web UI to users for real-time LLM interaction
- Optimize prompt quality using human and AI-based feedback
- Use RL to learn from both explicit ratings and implicit signals (sentiment, hallucination score, Groq evaluation)
- Visualize performance across multiple RL models over time

### Let’s build smarter prompts, one reward at a time.
