import requests
import torch
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify
import re
from difflib import get_close_matches
from groq import Groq
import mysql.connector
import logging
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

CORS(app, resources={r"/query": {"origins": "http://13.60.80.102/"}})

user_sessions = {}

client = Groq(api_key="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")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer('all-MiniLM-L6-v2')

VALID_STATIONS = [
    'Green Park', 'Vaishali', 'Rajiv Chowk', 'Hauz Khas', 'Central Secretariat',
    'Chandni Chowk', 'Dwarka', 'Noida City Centre', 'Huda City Centre', 'Saket',
    'Dilshad Garden', 'Rohini West', 'Kashmere Gate', 'Botanical Garden',
    'Dwarka Sector 21', 'Mandi House', 'Mayur Vihar', 'Nehru Place', 'Lajpat Nagar',
    'Jahangirpuri', 'Indraprastha', 'Shastri Park'
]

FAQ_KEYWORDS = [
    'what', 'who', 'where', 'when', 'how', 'why', 'am', 'are', 'was', 'were',
    'do', 'does', 'did', 'have', 'has', 'had', 'can', 'could', 'will', 'would',
    'shall', 'should', 'may', 'might', 'must', 'which', 'whose', 'whom'
]

# API endpoints
FAQ_API_URL = "http://13.60.80.102/:8080/faqs"
INTENT_API_URL = "http://13.60.80.102:8080/intents"
ROUTE_API_URL = "http://13.60.80.102:5000/get_route_and_fare"

def connect_db():
    """MySQL connection for saving answers"""
    connection = mysql.connector.connect(
        host="metrochatbot.cliqswukc30i.eu-north-1.rds.amazonaws.com",
        user="admin",
        password="Mayank2503",
        database="metro_chatbot"
    )
    return connection

def fetch_faqs():
    response = requests.get(FAQ_API_URL)
    return response.json()

def fetch_intents():
    response = requests.get(INTENT_API_URL)
    return response.json()

def get_sentence_embeddings(texts):
    batch_size = 32
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_embeddings = model.encode(texts[i:i + batch_size], convert_to_tensor=True)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

def classify_intent(user_query, intent_tensor_embeddings, intent_data, questions, threshold=0.7):
    user_embedding = get_sentence_embeddings([user_query])
    similarities = util.pytorch_cos_sim(user_embedding, intent_tensor_embeddings)
    most_similar_idx = torch.argmax(similarities).item()
    best_score = similarities[0, most_similar_idx].item()

    if best_score >= threshold:
        best_match_question = questions[most_similar_idx]
        for route, items in intent_data.items():
            for item in items:
                if item['question'].lower() == best_match_question:
                    return item['answer'], best_score * 100
    else:
        close_matches = get_close_matches(user_query, questions, n=1, cutoff=0.6)
        if close_matches:
            best_match_question = close_matches[0]
            for route, items in intent_data.items():
                for item in items:
                    if item['question'].lower() == best_match_question:
                        return item['answer'], best_score * 100

    return None, 0.0

def classify_faq(user_query, faq_tensor_embeddings, questions, faq_data, threshold=0.7):
    user_embedding = get_sentence_embeddings([user_query])
    similarities = util.pytorch_cos_sim(user_embedding, faq_tensor_embeddings)
    most_similar_idx = torch.argmax(similarities).item()
    best_score = similarities[0, most_similar_idx].item()

    if best_score >= threshold:
        best_match_question = questions[most_similar_idx]
        for section, faqs in faq_data.items():
            for item in faqs:
                if item['question'].lower() == best_match_question:
                    return item['answer'], best_score * 100
    else:
        close_matches = get_close_matches(user_query, questions, n=1, cutoff=0.6)
        if close_matches:
            best_match_question = close_matches[0]
            for section, faqs in faq_data.items():
                for item in faqs:
                    if item['question'].lower() == best_match_question:
                        return item['answer'], best_score * 100

    return None, 0.0

def parse_user_input(user_input):
    match = re.search(r'from (.+?) to (.+)', user_input, re.IGNORECASE)
    if match:
        from_station = match.group(1).strip()
        to_station = match.group(2).strip()
        return from_station, to_station
    else:
        return None, None

def correct_station_name(station_name):
    matches = get_close_matches(station_name, VALID_STATIONS, n=1, cutoff=0.7)
    if matches:
        return matches[0]
    return station_name

def fetch_route_and_fare(from_station, to_station):
    params = {
        'from_station': from_station,
        'to_station': to_station
    }
    try:
        response = requests.get(ROUTE_API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}

def save_to_database(table_name, question, answer):
    connection = connect_db()
    cursor = connection.cursor()
    query = f"INSERT INTO metro_chatbot.{table_name} (question, answer) VALUES (%s, %s)"
    cursor.execute(query, (question, answer))
    connection.commit()
    connection.close()

def call_groq_api(user_query):
    metro_hint = "This is related to metro services in Delhi. Please focus only on Indian Delhi-specific metro details and answers should be simple and short."

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": f"{user_query}. {metro_hint}"}],
            model="llama3-8b-8192",
            temperature=0.1 
        )
        answer = chat_completion.choices[0].message.content.strip()

        if "Please ask Delhi Metro-related questions" in answer:
            return "Please ask Delhi Metro-related questions."

        if any(keyword in user_query.lower() for keyword in FAQ_KEYWORDS):
            save_to_database("cleaned_faqs", user_query, answer)
        else:
            save_to_database("cleaned_intent", user_query, answer)

        return answer

    except Exception as e:
        return f"Error with Groq API: {str(e)}"

def generate_related_questions(user_query):
    try:
        metro_hint = "This is related to metro services. Please generate 3 simple questions based on the following query."
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": f"{user_query}. {metro_hint}"}],
            model="llama3-8b-8192",
            temperature=0.1  
        )
        generated_text = chat_completion.choices[0].message.content.strip()
        questions = re.findall(r'\d+\.\s*(.+)', generated_text)
        return questions[:3]
    except Exception as e:
        return [f"Error generating questions: {str(e)}"]

def prompt_user_for_question_selection(questions):
    print("Please select one of the following questions:")
    for idx, question in enumerate(questions, 1):
        print(f"{idx}. {question}")
    print("4. None of these")

    while True:
        user_selection = input("Pick One: ").strip()
        if user_selection in ['1', '2', '3', '4']:
            return int(user_selection)
        else:
            print("Invalid input. Please select 1, 2, 3, or 4.")

# Search for an answer in the database
def search_in_database(question):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    query = "SELECT answer FROM metro_chatbot.cleaned_faqs WHERE question = %s UNION SELECT answer FROM metro_chatbot.cleaned_intent WHERE question = %s"
    cursor.execute(query, (question, question))
    result = cursor.fetchone()
    connection.close()
    return result['answer'] if result else None

# Handle the user's query with suggestion functionality
# New set of route-related keywords
ROUTE_KEYWORDS = [
    "Travel to", "Go to", "Metro route to", "Find metro route to", "Route from", "Plan metro route to",
    "Commute to", "Take metro to", "Navigate to", "Metro route map to", "Best route to", "Fastest metro route to",
    "Direct route to", "Interchange metro route to", "Find your route to", "Plan your journey to", "Reach by metro",
    "Travel on metro route to", "Check metro route for", "Get to by metro", "Metro journey to", "Start metro route from",
    "Reach destination on metro", "Navigate metro route from", "Metro transfer route to", "Explore metro route to"
]


def handle_query_with_suggestion(user_id, query):
    """Main query handler with related question suggestion functionality."""
    
    # Initialize user session if not already present
    if user_id not in user_sessions:
        user_sessions[user_id] = {'last_query': '', 'related_questions_cache': {}}
    
    # Check if the query contains route information
    from_station, to_station = parse_user_input(query)
    if from_station and to_station:
        # Correct and validate station names
        from_station = correct_station_name(from_station)
        to_station = correct_station_name(to_station)
        
        # Fetch route and fare
        route_data = fetch_route_and_fare(from_station, to_station)

        if "error" in route_data:
            return {"error": route_data['error']}
        else:
            if 'full_route' in route_data:
                return {
                    "response": f"Route from {from_station} to {to_station}: {route_data['full_route']} Fare: ₹{route_data['total_fare']}"
                }
            else:
                return {"response": "No route found between the provided stations."}
    
    # Default fallback to process FAQ or intent-based query
    is_faq = query.lower().split()[0] in FAQ_KEYWORDS

    if is_faq:
        faq_data = fetch_faqs()
        faq_questions = [item['question'].lower() for section in faq_data for item in faq_data[section]]
        faq_tensor_embeddings = get_sentence_embeddings(faq_questions)
        answer, similarity = classify_faq(query, faq_tensor_embeddings, faq_questions, faq_data)

        if similarity < 70:
            answer = call_groq_api(query)
            save_to_database("cleaned_faqs", query, answer)
        return {"response": answer}
    else:
        intent_data = fetch_intents()
        intent_questions = [item['question'].lower() for route in intent_data for item in intent_data[route]]
        intent_tensor_embeddings = get_sentence_embeddings(intent_questions)
        answer, similarity = classify_intent(query, intent_tensor_embeddings, intent_data, intent_questions)

        if similarity < 70:
            answer = call_groq_api(query)
            save_to_database("cleaned_intent", query, answer)
        return {"response": answer}

    """Main query handler with related question suggestion functionality."""
    
    # Initialize user session if not already present
    if user_id not in user_sessions:
        user_sessions[user_id] = {'last_query': '', 'related_questions_cache': {}}
    
    # Initialization moved here
    count = 0
    step = 1
    print("Welcome to the Metro Chatbot! Type 'exit' to end the chat.")
    user_name = user_id  # Assume user_name is equivalent to user_id for personalization
    print(f"Nice to meet you, {user_name}! How can I assist you with the Delhi Metro today?")
    
    while True:
        query = input(f"{user_name}: ").strip()  # Capture user input with their name as prompt
        if count < step:
            count += 1

        if query.lower() == 'exit':
            print("Goodbye!")
            break

        # Parse input to check for source and destination stations
        from_station, to_station = parse_user_input(query)
        
        # Check if both source and destination were provided in the initial query
        if from_station and to_station:
            # Correct and validate station names
            from_station = correct_station_name(from_station)
            to_station = correct_station_name(to_station)
            
            # Fetch and display route and fare data directly
            route_data = fetch_route_and_fare(from_station, to_station)

            if "error" in route_data:
                print(f"Error fetching route and fare: {route_data['error']}")
            else:
                if 'full_route' in route_data:
                    print(f"Route from {from_station} to {to_station}: {route_data['full_route']}")
                    print(f"Fare: ₹{route_data['total_fare']}")
                else:
                    print("No route found between the provided stations.")
            return 

    # If only partial travel information (like "I want to travel") is given, prompt for details
        elif any(keyword in query.lower() for keyword in ["travel", "route", "go", "navigate"]):
          print("Bot: What is your source station?")
        from_station = input(f"{user_id}: ").strip()
        
        print("Bot: What is your destination station?")
        to_station = input(f"{user_id}: ").strip()

        # Fetch and display route and fare data after getting complete information
        route_data = fetch_route_and_fare(from_station, to_station)

        if "error" in route_data:
            print(f"Error fetching route and fare: {route_data['error']}")
        else:
            if 'full_route' in route_data:
                print(f"Route from {from_station} to {to_station}: {route_data['full_route']}")
                print(f"Fare: ₹{route_data['total_fare']}")
            else:
                print("No route found between the provided stations.")
        return  # End the function here to avoid additional question prompts

    # Check if the first word of the query is in FAQ_KEYWORDS
    first_word = query.lower().split()[0]
    is_faq = first_word in FAQ_KEYWORDS

    if is_faq:
        # Load FAQ data and embeddings
        faq_data = fetch_faqs()
        faq_questions = [item['question'].lower() for section in faq_data for item in faq_data[section]]
        faq_tensor_embeddings = get_sentence_embeddings(faq_questions)
        
        # Classify FAQ
        answer, similarity = classify_faq(query, faq_tensor_embeddings, faq_questions, faq_data)
        
        # If similarity score is below the threshold, call Groq API
        if similarity < 70:  # 70 is 0.7 threshold in percentage
            print(f"Bot: No exact match found in FAQ. Fetching answer from Groq API...")
            answer = call_groq_api(query)
            save_to_database("cleaned_faqs", query, answer)  # Save Groq-generated answer to FAQ DB
        else:
            print(f"Bot: {answer} (Similarity: {similarity:.2f}%)")
    else:
        # Handle as an intent-based query if not FAQ
        intent_data = fetch_intents()
        intent_questions = [item['question'].lower() for route in intent_data for item in intent_data[route]]
        intent_tensor_embeddings = get_sentence_embeddings(intent_questions)
        
        # Classify Intent
        answer, similarity = classify_intent(query, intent_tensor_embeddings, intent_data, intent_questions)
        
        if similarity < 70:
            print(f"Bot: No exact match found in Intent. Fetching answer from Groq API...")
            answer = call_groq_api(query)
            save_to_database("cleaned_intent", query, answer)  # Save Groq-generated answer to Intent DB
        else:
            print(f"Bot: {answer} (Similarity: {similarity:.2f}%)")

    # Skip generating related questions when route and fare information is provided
    if not is_faq:
        if query in user_sessions[user_id]['related_questions_cache']:
            related_questions = user_sessions[user_id]['related_questions_cache'][query]
        elif query in user_sessions[user_id]['related_questions_cache']:
            print(f"Bot: No exact match found in Intent. Fetching answer from Groq API...")
            #continue
        else:
            related_questions = generate_related_questions(query)
            user_sessions[user_id]['related_questions_cache'][query] = related_questions

        selected_option = prompt_user_for_question_selection(related_questions)

        if selected_option == 4:
            print("Bot: Session ended. Starting a new conversation.")
        else:
            selected_question = related_questions[selected_option - 1]
            cached_answer = search_in_database(selected_question)
            if cached_answer:
                print(f"Bot: {cached_answer}")
            else:
                groq_answer = call_groq_api(selected_question)
                print(f"Bot: {groq_answer}")


# Helper function to store query and response in MySQL
def store_conversation(user_query, chatbot_response):
    try:
        # Connect to MySQL database
        connection = connect_db()
        cursor = connection.cursor()

        # SQL query to insert the conversation into the table (without user_id)
        sql_query = "INSERT INTO chatbot_conversations (user_query, chatbot_response) VALUES (%s, %s)"
        
        # Execute the query with user_query and chatbot_response
        cursor.execute(sql_query, (user_query, chatbot_response))

        # Commit the transaction
        connection.commit()
        logging.info("Conversation stored in the database successfully.")
    except mysql.connector.Error as err:
        logging.error(f"Error storing conversation: {err}")
    finally:
        # Ensure connection is closed
        if connection.is_connected():
            cursor.close()
            connection.close()

@app.route('/query', methods=['POST'])
def handle_query_route():
    try:
        user_input = request.json.get('message', '').strip()
        user_id = request.json.get('user_id', '').strip()

        if not user_input:
            return jsonify({"error": "No query provided."}), 400

        if not user_id:
            return jsonify({"error": "No user_id provided."}), 400

        response = handle_query_with_suggestion(user_id, user_input)

        # Save conversation to DB
        store_conversation(user_input, response['response'])

        return jsonify(response)
    except Exception as e:
        logging.error(f"Error handling query route: {e}")
        return jsonify({"error": "An internal error occurred."}), 500
@app.route('/')
def hello():
    return "Hello, World!"



# Main entry point
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(port=5001, debug=True)
s
