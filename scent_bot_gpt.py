import os, json, numpy as np, pandas as pd, sys, requests
from dotenv import load_dotenv
from typing import Dict

load_dotenv()
CSV_PATH   = "fra_simple_10family.csv"
GROQ_KEY   = os.getenv("GROQ_KEY") # find this inside the google doc 
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL      = "llama-3.3-70b-versatile"


EMOTIONS = [
    "happy","sad","angry","calm","anxious","excited","bored","nostalgic",
    "lonely","loved","confident","shy","playful","serious","mysterious",
    "adventurous","peaceful","melancholic","romantic","energetic","tired",
    "hopeful","hopeless","proud","embarrassed","curious","content",
    "rebellious","elegant","warm-hearted","cold-hearted"
]

ACCORDS = [
    "Citrus–Green","Amber–Resinous","Musky–Soapy","Marine–Ozonic",
    "Spicy–Aromatic","Woody–Earthy","Floral–Powdery","Fruity–Tropical",
    "Sweet–Gourmand","Smoky–Leathery"
]

# ---------- matrix builder ----------
def build_matrix() -> np.ndarray:
    df = pd.read_csv(CSV_PATH)
    fam_df = df[ACCORDS].astype(float)
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    pca = PCA(n_components=10, whiten=True)
    components = pca.fit_transform(fam_df)
    M = np.zeros((30,10))
    for i in range(30):
        reg = Ridge(alpha=1e-6, fit_intercept=False).fit(fam_df, components[:, i])
        M[i] = reg.coef_
    M = np.maximum(M, 0)
    M /= M.sum(axis=1, keepdims=True) + 1e-9
    json.dump(M.round(3).tolist(), open("emotion_matrix.json", "w"))
    return M

if not os.path.exists("emotion_matrix.json"):
    M = build_matrix()
else:
    M = np.array(json.load(open("emotion_matrix.json")))

# ---------- Groq caller ----------
SYSTEM_RQ = f"""You are a perfume-creator AI.
Classify the user sentence into one of three cases:
1 = scent-driven (odour scene)
2 = emotion-driven (literary / feeling)
3 = mixed (both scent and feeling)

Reply in the SAME language as the user.

Output format (strict, no extra text):
CASE|<1 or 2 or 3>
EMOTION|<30 comma-separated numbers 0-1>
ACCORD_RATIONALE|<one short sentence explaining how each part of the user sentence correlates to emotion and scent>

Map the 30 emotion numbers to this fixed order:
{','.join(EMOTIONS)}"""

def gpt_emotion_vector(prompt: str) -> tuple[np.ndarray, int, str]:
    try:
        hdr = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": MODEL,
            "temperature": 0.25,
            "max_tokens": 200,
            "messages": [
                {"role": "system", "content": SYSTEM_RQ},
                {"role": "user",   "content": prompt}
            ]
        }
        r = requests.post(GROQ_URL, headers=hdr, json=payload, timeout=30)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"].strip()
        print("RAW GROQ REPLY:", repr(txt))

        lines = [L for L in txt.splitlines() if L]
        case_line  = next((L for L in lines if L.startswith("CASE|")), "CASE|3")
        emo_line   = next((L for L in lines if L.startswith("EMOTION|")), "")
        rat_line   = next((L for L in lines if L.startswith("ACCORD_RATIONALE|")), "")

        case = int(case_line.split("|", 1)[1].strip())
        rationale = rat_line.split("|", 1)[1].strip()

        nums = emo_line.split("|", 1)[1].strip().split(",")
        if len(nums) != 30:
            print(f"WARNING: expected 30 numbers, got {len(nums)} → padding with 0")
            nums.extend(["0"] * (30 - len(nums)))
        vec = np.array([float(x) for x in nums], dtype=float)

    except Exception as e:
        print("Groq call failed:", e)
        vec, case, rationale = np.zeros(30), 2, ""
    return vec, case, rationale

# ---------- accord recipe + pretty print ----------
def vector_to_accord(vec: np.ndarray) -> dict[str, float]:
    raw = vec @ M
    raw = np.maximum(raw, 0)
    norm = raw / (raw.sum() or 1)
    return dict(zip(ACCORDS, map(float, norm.round(3))))

def prompt_to_scent(prompt: str):
    emo_vec, case, rationale = gpt_emotion_vector(prompt)
    accord = vector_to_accord(emo_vec)

    print("\n==========  EMOTION VECTOR  (30)  ==========")
    for e, v in zip(EMOTIONS, emo_vec.round(3)):
        print(f"{e:18} {v}")
    print("\n==========  ACCORD RATIO  (10)  ==========")
    for a, v in sorted(accord.items(), key=lambda x: -x[1]):
        print(f"{a:18} {v}")
    print("\n==========  CASE & RATIONALE  ==========")
    print(f"Detected case: {case}  ({['Scent','Emotion','Mixed'][case-1]}-driven)")
    print(f"Rationale: {rationale}")
    sys.stdout.flush()
    return emo_vec, accord, case, rationale

# ---------- CLI ----------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scent_bot_groq.py 'your prompt here'")
        sys.exit(1)
    prompt = " ".join(sys.argv[1:])
    prompt_to_scent(prompt)
