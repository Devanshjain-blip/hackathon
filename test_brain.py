import os
from groq import Groq
from dotenv import load_dotenv

# Load the secret key
load_dotenv() 

# Wake up the Groq Brain
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Sending message to Llama 3.1 via Groq...")

try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # <--- THIS IS THE UPDATED MODEL
        messages=[{"role": "user", "content": "Hey Llama, reply with exactly 'Code Crew is ready to win!' if you hear me."}]
    )
    print("--- SUCCESS! ---")
    print("Brain says:", response.choices[0].message.content)
except Exception as e:
    print("--- ERROR ---")
    print(e)