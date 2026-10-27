import requests
question = input("Ask question: ")
try:   
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3:4b",
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "stream": False,
        },
        timeout=(5,300),
    )
    response.raise_for_status()
    data = response.json()
    print(data["message"]["content"])
except requests.exceptions.HTTPError as error:
    print(f"Ollama's error details: {error.response.text}")
except requests.exceptions.RequestException as error:
    print(f"Could not get a response from Ollama: {error}")
except KeyError as error:
    print(f"Ollama's response is missing an expected field: {error}")

