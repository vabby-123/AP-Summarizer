import google.generativeai as genai

genai.configure(api_key="AIzaSyArJjJFxy4LVC4vCgDrkppJdY61o5d9xuU")

available_models = genai.list_models()
for model in available_models:
    print(model.name)