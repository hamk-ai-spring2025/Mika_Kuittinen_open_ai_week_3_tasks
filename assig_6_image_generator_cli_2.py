import os
import time
import argparse
import requests
import base64
from dotenv import load_dotenv
import openai

# Lataa API-avaimet
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
openai.api_key = OPENAI_API_KEY

# Tallennuskansio
SAVE_DIR = "C:/pics/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Tuetut kuvasuhteet DALL·E 3:lle
DALL_E_SIZES = {
    "1:1": "1024x1024",
    "16:9": "1792x1024",
    "9:16": "1024x1792"
}

def get_dalle_size(aspect_ratio):
    if aspect_ratio not in DALL_E_SIZES:
        print(f"[VAROITUS] Kuvasuhde '{aspect_ratio}' ei ole tuettu. Käytetään oletuksena 1:1.")
        return DALL_E_SIZES["1:1"]
    return DALL_E_SIZES[aspect_ratio]

def download_image(url, file_path):
    response = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(response.content)

def dalle_generate(prompt, n, aspect_ratio):
    size = get_dalle_size(aspect_ratio)
    print(f"[DALL·E 3] Luo {n} kuvaa: '{prompt}', koko: {size}")
    file_paths = []

    for i in range(n):
        try:
            response = openai.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size=size,
                quality="standard"
            )
            url = response.data[0].url
            file_name = f"image_{int(time.time())}_{i+1}.png"
            file_path = os.path.join(SAVE_DIR, file_name)
            download_image(url, file_path)
            print(f"{file_path}: {url}")
            file_paths.append(file_path)
        except openai.BadRequestError as e:
            print(f"[VIRHE] {e}")
    return file_paths

def stable_generate(prompt, n, negative_prompt=None, seed=None):
    print(f"[Stable Diffusion] Luo {n} kuvaa: '{prompt}'")
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json"
    }
    url = "https://api.stability.ai/v1/generation/stable-diffusion-v1-5/text-to-image"
    file_paths = []

    for i in range(n):
        payload = {
            "text_prompts": [{"text": prompt}],
            "cfg_scale": 7,
            "clip_guidance_preset": "FAST_BLUE",
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 30
        }
        if negative_prompt:
            payload["text_prompts"].append({"text": negative_prompt, "weight": -1.0})
        if seed:
            payload["seed"] = seed

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            print(f"[VIRHE] {response.status_code}: {response.text}")
            continue

        data = response.json()
        base64_image = data["artifacts"][0]["base64"]
        image_data = base64.b64decode(base64_image)

        file_name = f"image_{int(time.time())}_{i+1}.png"
        file_path = os.path.join(SAVE_DIR, file_name)
        with open(file_path, "wb") as f:
            f.write(image_data)
        print(f"{file_path}: (local file, seed={seed})")
        file_paths.append(file_path)

    return file_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monipuolinen kuvageneraattori (OpenAI / Stable Diffusion)")
    parser.add_argument("--engine", choices=["dalle", "stable"], required=True, help="Generaattori: dalle tai stable")
    parser.add_argument("--prompt", required=True, help="Kuvakehote")
    parser.add_argument("--n", type=int, default=1, help="Kuvien määrä")
    parser.add_argument("--aspect", default="1:1", help="DALL·E 3 kuvasuhde (1:1, 16:9, 9:16)")
    parser.add_argument("--negative_prompt", help="(Vain Stable Diffusion) Mitä ei haluta näkyvän")
    parser.add_argument("--seed", type=int, help="(Vain Stable Diffusion) Satunnaissiementä")

    args = parser.parse_args()

    if args.engine == "dalle":
        dalle_generate(args.prompt, args.n, args.aspect)
    elif args.engine == "stable":
        stable_generate(args.prompt, args.n, args.negative_prompt, args.seed)