from flask import Flask, request, jsonify
import requests
import os

IBM_API_KEY = os.getenv("IBM_API_KEY")
IBM_URL = "https://api.watsonx.ai/v1/generation/text"

def get_traffic_data(destination):
    return f"Traffic to {destination} is moderate, ~35 minutes travel time."

def get_weather_data(city):
    return f"Weather in {city}: Sunny, 30°C."

app = Flask(__name__)

def ask_watsonx(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {IBM_API_KEY}"
    }
    payload = {
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 200,
            "min_new_tokens": 30,
            "temperature": 0.7
        },
        "model_id": "ibm/granite-13b-chat-v2"
    }
    response = requests.post(IBM_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get("results", [{}])[0].get("generated_text", "No response")
    else:
        return f"Error: {response.text}"

def agentic_ai(user_query):
    planning_prompt = f"""
    You are an AI planner. Analyze the user request and identify what steps are needed.
    If the task involves weather or traffic, suggest calling those tools.
    User request: {user_query}
    """
    plan = ask_watsonx(planning_prompt)

    tool_responses = []
    if "traffic" in user_query.lower():
        tool_responses.append(get_traffic_data("Airport"))
    if "weather" in user_query.lower():
        tool_responses.append(get_weather_data("Chennai"))

    synthesis_prompt = f"""
    The user asked: {user_query}
    Plan: {plan}
    Tool results: {tool_responses}
    Please give a concise and helpful answer.
    """
    final_answer = ask_watsonx(synthesis_prompt)

    return {
        "plan": plan,
        "tools": tool_responses,
        "answer": final_answer
    }

@app.route("/")
def home():
    return "✅ Agentic AI Flask Backend is running!"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing 'query' in request"}), 400
    result = agentic_ai(query)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
