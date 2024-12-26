#[15:56, 23/12/2024] Abhay Office Noida: # whats app file hai yee
import logging
import random
import os
import requests
import torch
from sentence_transformers import SentenceTransformer, util
import re
from difflib import get_close_matches
from groq import Groq
import mysql.connector
from flask_cors import CORS
from flask import Flask, request, jsonify, session, Response, stream_with_context
from difflib import SequenceMatcher
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Flask App Configuration
app = Flask(__name__)
CORS(app)
# app.secret_key = "your_secret_key"  # Replace with a secure random key for production

CORS(app, resources={r"/query": {"origins": "http://13.49.68.219/"}})
# Dictionary to store user sessions, keeping track of their last question and related questions cache
user_sessions = {}

# Initialize Groq client
client = Groq(api_key="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")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Initialize the valid metro stations
VALID_STATIONS = [
    'Green Park', 'Vaishali', 'Rajiv Chowk', 'Hauz Khas', 'Central Secretariat',
    'Chandni Chowk', 'Dwarka', 'Noida City Centre', 'Huda City Centre', 'Saket',
    'Dilshad Garden', 'Rohini West', 'Kashmere Gate', 'Botanical Garden',
    'Dwarka Sector 21', 'Mandi House', 'Mayur Vihar', 'Nehru Place', 'Lajpat Nagar',
    'Jahangirpuri', 'Indraprastha', 'Shastri Park'
]

# Define the keywords for FAQ detection
FAQ_KEYWORDS = [
    'what', 'who', 'where', 'when', 'how', 'why', 'am', 'are', 'was', 'were',
    'do', 'does', 'did', 'have', 'has', 'had', 'can', 'could', 'will', 'would',
    'shall', 'should', 'may', 'might', 'must', 'which', 'whose', 'whom'
]

# API endpoints
FAQ_API_URL = "http://13.49.68.219:8080/faqs"
INTENT_API_URL = "http://13.49.68.219:8080/intents"
ROUTE_API_URL = "http://13.49.68.219:5000/get_route_and_fare"

def connect_db():
    """MySQL connection for saving answers"""
    connection = mysql.connector.connect(
        host="metrochatbot.cliqswukc30i.eu-north-1.rds.amazonaws.com",
        user="admin",
        password="Mayank2503",
        database="chat_bot"
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
    query = f"INSERT INTO chat_bot.{table_name} (question, answer) VALUES (%s, %s)"
    cursor.execute(query, (question, answer))
    connection.commit()
    connection.close()


def call_groq_api(user_query):
    metro_hint = "This is related to metro services in Delhi. Please focus only on Indian Delhi-specific metro details and answers should be simple and short and give in single or double line. If this query is not related to the Delhi Metro."

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": f"{user_query}. {metro_hint}"}],
            model="llama3-8b-8192",
            temperature=0.1  # Lower temperature for deterministic output
        )
        answer = chat_completion.choices[0].message.content.strip()

        # Check if the answer says it's unrelated to Delhi Metro and respond accordingly
        if "Please ask Delhi Metro-related questions" in answer:
            return "Please ask Delhi Metro-related questions."
        
        # Save to database (FAQ or Intent depending on query)
        if any(keyword in user_query.lower() for keyword in FAQ_KEYWORDS):
            save_to_database("cleaned_faqs", user_query, answer)
        else:
            save_to_database("cleaned_intent", user_query, answer)
        return answer

    except Exception as e:
        return f"Error with Groq API: {str(e)}"

# Generate related questions using Groq API
def generate_related_questions(user_query):
    """Generate three related questions using the Groq API based on the user query."""
    try:
        metro_hint = "This is related to metro services. Please generate 3 simple questions based on the following query. Make sure the questions are specific to Delhi Metro. Questions should be simple and short"
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": f"{user_query}. {metro_hint}"}],
            model="llama3-8b-8192",
            temperature=0.1  # Lower temperature for deterministic output
        )
        generated_text = chat_completion.choices[0].message.content.strip()
        questions = re.findall(r'\d+\.\s*(.+)', generated_text)  # Extract numbered questions
        return questions[:3]  # Return the top 3 questions
    except Exception as e:
        return [f"Error generating questions: {str(e)}"]
    
#-------------------------------------------------------------------------------------
# its not usefull part 
#-------------------------------------------------------------------------------------
# # Prompt user to select one of the generated questions
# def prompt_user_for_question_selection(questions):
#     """Prompt the user to select one of the generated questions."""
#     print("Please select one of the following questions:")
#     for idx, question in enumerate(questions, 1):
#         yield f"{idx}. {question}"
#     print("4. None of these")
    
#     while True:
#         user_selection = input("Pick One: ").strip()
#         if user_selection in ['1', '2', '3', '4']:
#             return int(user_selection)
#         else:
#             print("Invalid input. Please select 1, 2, 3, or 4.")
#-------------------------------------------------------------------------------------------------------

def search_in_database(question):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    query = "SELECT answer FROM chat_bot.cleaned_faqs WHERE question = %s UNION SELECT answer FROM chat_bot.cleaned_intent WHERE question = %s"
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

def stream_response(query_generator):
    """Stream response chunks to the client."""
    for chunk in query_generator:
        yield chunk
        yield "\n"  # Separate chunks for better client-side processing.

def query_generator(query_result):
    """Generate chunks of the query result for streaming."""
    for line in query_result.split('. '):  # Break into chunks by sentences
        yield line + '.'



# Helper: Check similarity between two strings
def is_similar(a, b, threshold=0.8):
    return SequenceMatcher(None, a, b).ratio() >= threshold

# Helper: Fetch last response from the database
def get_last_response_from_db(user_id):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    # Fetch the latest response for the given user_id by ordering by the primary key id in descending order
    query = "SELECT chatbot_response FROM chat_bot.chatbot_conversations WHERE user_id = %s ORDER BY id DESC LIMIT 1"
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()
    connection.close()
    return result['chatbot_response'] if result else None

 # Helper: Fetch last response from the database
def route_from_db(user_id):
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    # Fetch the latest response for the given user_id by ordering by the primary key id in descending order
    query = "SELECT user_query FROM chat_bot.chatbot_conversations WHERE user_id = %s ORDER BY id DESC LIMIT 1"
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()
    connection.close()
    return result['user_query'] if result else None
 


def handle_query_with_stream(user_id, query):
    similarity_score = 0
    """Handles user queries, including route, FAQ, intent classification, and suggestions for related questions."""
    try:
        chatbot_response = get_last_response_from_db(user_id)
        
        # Initialize user session if not already present
        if user_id not in user_sessions: 
            user_sessions[user_id] = {"last_step": "", "last_station": "", "related_questions_cache": {}}

        session_state = user_sessions[user_id]
        session_state['last_query'] = query  # Store the current query as the last query

        # Process based on the current state
        # Parse input to check for source and destination stations
        from_station, to_station = parse_user_input(query)
    
    # Check if both source and destination were provided in the initial query
        if from_station and to_station:
            # Correct and validate station names
            from_station = correct_station_name(from_station)
            to_station = correct_station_name(to_station)
            
#----------------------------------------------------------------------------------------------------------------------------------------
# start here
# this change by abhay and sarthak date - 26/12//2024 (3:53 pm)
# ---------------------------------------------------------------------------------------------------------------------------------------            
            route_data = fetch_route_and_fare(from_station, to_station)

            if "error" not in route_data:
                # Format the response as plain text
                route_text = f"Route: {route_data['full_route']}\nZone: {route_data['distinct_zones']}\nTotal Fare: {route_data['total_fare']}"
                yield route_text  # Yield the formatted plain text
                return
#-------------------------------------------------------------------------------------------------------------------------------------
# end here
# ---------------------------------------------------------------------------------------------------------------------------------------
    
            if "error" in route_data:
                print(f"Error fetching route and fare: {route_data['error']}")
            else:
                if 'full_route' in route_data:
                    print(f"Route from {from_station} to {to_station}: {route_data['full_route']}")
                    print(f"Fare: ₹{route_data['total_fare']}")
                else:
                    print("No route found between the provided stations.")
            return  # End the function here to avoid additional question prompts

        # If only partial travel information (like "I want to travel") is given, prompt for details
        elif any(keyword in query.lower() for keyword in ["travel", "route", "go", "navigate"]):
            yield "What is your source station?"
            return
        
        if chatbot_response and isinstance(chatbot_response, str):
            similarity_score = is_similar(chatbot_response.lower(), "What is your source station?")
        if similarity_score >= 0.8:
                yield "What is your destination station?"
                return
            

        if chatbot_response and isinstance(chatbot_response, str):
            similarity_score = is_similar(chatbot_response.lower(), "What is your destination station?")
        if similarity_score >= 0.8:
                to_station = query
                from_station = route_from_db(user_id)
    
                # Fetch and display route and fare data after getting complete information
                route_data = fetch_route_and_fare(from_station, to_station)
    
                if "error" in route_data:
                    yield f"Error fetching route and fare: {route_data['error']}"
                else:
                    if 'full_route' in route_data:
                        yield f"Route from {from_station} to {to_station}: {route_data['full_route']}"
                        yield f"Fare: ₹{route_data['total_fare']}"
                    else:
                        yield "No route found between the provided stations."
                    return  # End the function here to avoid additional question prompts
        
 # Check if it's a lost item query
        if any(keyword in query.lower() for keyword in ['lost', 'theft', 'stolen']):
                user_sessions[user_id]['last_query'] = query
                yield f"Which was the last station you were at?"
                return  # Wait for the response for last_station

        if chatbot_response and isinstance(chatbot_response, str):
            similarity_score = is_similar(chatbot_response.lower(), "Which was the last station you were at?")
        if similarity_score >= 0.8:
                yield "Do you want to report your lost or stolen item? (yes/no)"
                return


        # If report_decision is provided, complete the lost item reporting process
        if chatbot_response and isinstance(chatbot_response, str):
            similarity_score = is_similar(chatbot_response.lower(), "Do you want to report your lost or stolen item? (yes/no)")
        if similarity_score >= 0.8:
            if query.lower() == "yes":
                station = route_from_db(user_id)
                groq_query = f"I lost my item at {station}. Can you guide me on how to report it to the Delhi Metro authorities?"
                report_info = call_groq_api(groq_query)
                yield f"Bot: {report_info}"
            else:
                yield "Okay, let me know if you need any other help."
            return
# ------------------------------------------------------------------------------------------------------------------
# This part is added by Abhay on 22-dec-2024
# ------------------------------------------------------------------------------------------------------------------
        if any(keyword in query.lower() for keyword in ['park', 'parking']):
                user_sessions[user_id]['last_query'] = query
                yield f"Which station are you looking to park at?"
                return  # Wait for the response for last_station

        if chatbot_response and isinstance(chatbot_response, str):
            similarity_score = is_similar(chatbot_response.lower(), "Which station are you looking to park at?")
        if similarity_score >= 0.9:
                yield "Do you want to know is parking is available or not? (yes/no)"
                return


        # If report_decision is provided, complete the lost item reporting process
        if chatbot_response and isinstance(chatbot_response, str):
            similarity_score = is_similar(chatbot_response.lower(), "Do you want to know is parking is available or not? (yes/no)")
        if similarity_score >= 0.9:
            if query.lower() == "yes":
                station = route_from_db(user_id)
                groq_query = f"I looking for parking at {station}. Can you guide me what is the process to park my vechicle and is parking available at that station and what is the fare or fine to park my vechical?"
                report_info = call_groq_api(groq_query)
                yield f"Bot: {report_info}"
            else:
                yield "Okay, let me know if you need any other help."
            return
            
# ------------------------------------------------------------------------------------------------------------------


        if any(keyword in query.lower() for keyword in ['fire', 'emergency', 'accident']):
                user_sessions[user_id]['last_query'] = query
                yield f"Which station are you looking for emergency exit information?"
                return  # Wait for the response for last_station

        if chatbot_response and isinstance(chatbot_response, str):
            similarity_score = is_similar(chatbot_response.lower(), "Which station are you looking for emergency exit information?")
        if similarity_score >= 0.9:
                yield "Do you want to know exit information for emergency? (yes/no)"
                return


        # If report_decision is provided, complete the lost item reporting process
        if chatbot_response and isinstance(chatbot_response, str):
            similarity_score = is_similar(chatbot_response.lower(), "Do you want to know exit information for emergency? (yes/no)")
        if similarity_score >= 0.9:
            if query.lower() == "yes":
                station = route_from_db(user_id)
                groq_query = f"I looking for emergency at {station} metro station. Can you guide me what to do in case of emergency and where is the exit gate for emergency?"
                report_info = call_groq_api(groq_query)
                yield f"Bot: {report_info}"
            else:
                yield "Okay, let me know if you need any other help."
            return

# ------------------------------------------------------------------------------------------------------------------
# Here it END
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# This part is added by Abhay on 25-dec-2024
# ------------------------------------------------------------------------------------------------------------------

        if any(keyword in query.lower() for keyword in ['1', '2', '3']):
            generated_question = get_last_response_from_db(user_id)
            questions = generated_question.split('\n')
            if query == '1' and len(questions) > 2:
                query1 = questions[2]
            elif query == '2' and len(questions) > 3:
                query1 = questions[3]
            elif query == '3' and len(questions) > 4:
                 query1 = questions[4]
            answer = call_groq_api(query1)
            yield answer
            
            # 3. Generate related questions for further engagement
            if query not in user_sessions[user_id]['related_questions_cache']:
                related_questions = generate_related_questions(query)
                user_sessions[user_id]['related_questions_cache'][query] = related_questions
            else:
                related_questions = user_sessions[user_id]['related_questions_cache'][query]
    
            yield "Here are some related questions you might find helpful:"
            for idx, question in enumerate(related_questions, start=1):
                yield f"{idx}. {question}"
            return


# ------------------------------------------------------------------------------------------------------------------
# Here it END
# ------------------------------------------------------------------------------------------------------------------
        # 2. Determine if query is an FAQ or intent-based query
        is_faq = query.lower().split()[0] in FAQ_KEYWORDS

        if is_faq:

            # Fetch FAQ data and process it
            faq_data = fetch_faqs()
            faq_questions = [item['question'].lower() for section in faq_data for item in faq_data[section]]
            faq_tensor_embeddings = get_sentence_embeddings(faq_questions)

            answer, similarity = classify_faq(query, faq_tensor_embeddings, faq_questions, faq_data)

            # If similarity is low, use Groq API
            if similarity < 70:
                answer = call_groq_api(query)
                save_to_database("cleaned_faqs", query, answer)

            yield answer
            return

        else:

            # Fetch intent data and process it
            intent_data = fetch_intents()
            intent_questions = [item['question'].lower() for route in intent_data for item in intent_data[route]]
            intent_tensor_embeddings = get_sentence_embeddings(intent_questions)

            answer, similarity = classify_intent(query, intent_tensor_embeddings, intent_data, intent_questions)

            # If similarity is low, use Groq API
            if similarity < 70:
                answer = call_groq_api(query)
                save_to_database("cleaned_intent", query, answer)

            yield answer
            
            
            # 3. Generate related questions for further engagement
            if query not in user_sessions[user_id]['related_questions_cache']:
                related_questions = generate_related_questions(query)
                user_sessions[user_id]['related_questions_cache'][query] = related_questions
            else:
                related_questions = user_sessions[user_id]['related_questions_cache'][query]
    
            yield "Here are some related questions you might find helpful:"
            for idx, question in enumerate(related_questions, start=1):
                yield f"{idx}. {question}"
            return

#---------------------------------------------------------------------------------------------------------------
# this part also not usefull           
#---------------------------------------------------------------------------------------------------------------   
        # 4. Handle related questions suggestion

            # selected_option = prompt_user_for_question_selection(related_questions)
            # if selected_option == 4:  # Assume 4 is the "End Session" option
            #     yield "Session ended. Thank you for your query."
            #     return
            # else:
            #     selected_question = related_questions[selected_option - 1]
            #     yield f"You selected: {selected_question}"

            #     # Search database for the selected question's answer
            #     cached_answer = search_in_database(selected_question)
            #     if cached_answer:
            #         yield cached_answer
            #         return
            #     else:
            #         groq_answer = call_groq_api(selected_question)
            #         yield groq_answer
            #         return
#-----------------------------------------------------------------------------------------------------------
    except Exception as e:
        print(e)


# Helper function to store query and response in MySQL
def store_conversation(user_query, chatbot_response, user_id):
    try:
        # Connect to MySQL database
        connection = connect_db()
        cursor = connection.cursor()

        # SQL query to insert the conversation into the table (without user_id)
        sql_query = "INSERT INTO chat_bot.chatbot_conversations (user_query, chatbot_response, user_id) VALUES (%s, %s, %s)"
        
        # Execute the query with user_query and chatbot_response
        cursor.execute(sql_query, (user_query, chatbot_response, user_id))

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
def handle_streaming_query(user_id=None, user_input=None):
    try:
        if(user_id and user_input):
            user_input = user_input.strip()
            user_id = user_id.strip()
        else:
            user_input = request.json.get('message', '').strip()
            user_id = request.json.get('user_id', '').strip()

        if not user_input:
            return jsonify({"error": "No query provided."}), 400
        if not user_id:
            return jsonify({"error": "No user_id provided."}), 400

        # Retrieve current state (if any)
        last_station = user_sessions.get(user_id, {}).get('last_station')
        report_decision = user_sessions.get(user_id, {}).get('report_decision')

# ------------------------------------------------------------------------------------------------------------------
# This part is changed by Abhay on 25-dec-2024
# ------------------------------------------------------------------------------------------------------------------

        # Generate the chatbot response (assuming handle_query_with_stream yields responses)
        response_chunks = []
        for chunk in handle_query_with_stream(user_id, user_input):
            response_chunks.append(chunk)

        # Ensure line breaks are added for the complete response
        chatbot_response = '\n'.join(response_chunks)

        # Save conversation to DB
        store_conversation(user_input, chatbot_response, user_id)

        # Stream the response with line breaks
        return Response(
            stream_with_context((chunk + '\n' for chunk in response_chunks)),
            content_type='text/plain'
        )

# ------------------------------------------------------------------------------------------------------------------
# It end here.
# -------------------------------------------------------------------------------------------------------------------

    except Exception as e:
        logging.error(f"Error in streaming query handler: {e}")
        return jsonify({"error": "An internal error occurred."}), 500
    
@app.route('/webhook', methods=['POST'])
def call_webhook_endpoint():
    try:
        data = request.get_json()
    
        # Log the received data
        print("Received POST data:")
        print(data)
        if(data['object']):
            messages = data['entry'][0]['changes'][0]['value']['messages']
            phone_number_id = data['entry'][0]['changes'][0]['value']['metadata']['phone_number_id']
            print('phone_number_id',phone_number_id)
            if (messages):
                messageFrom = messages[0]['from'] #// Sender's WhatsApp ID
                messageBody = messages[0]['text']['body'] #// Message content
#-------------------------------------------------------------------------------------
                # this line is change by mayank 12/24/2024
#                generate_4_digit_string = lambda: str(random.randint(1000, 9999))
#                user_id = generate_4_digit_string()
                user_id = str(messageFrom[-4:])
#-----------------------------------------------------------------------------------------------                                                    
                print('Received message from format:',messageFrom, messageBody)
                
                #payload = {"message": messageBody, "user_id": '123'}

                botResponse = handle_streaming_query(user_id, messageBody)        
                #print("response from chatbot",botResponse)

                payload = {
                    'messageFrom': messageFrom,
                    'botResponse': botResponse.get_data(as_text=True),
                    'phone_number_id': phone_number_id
                }
                print("payload",payload)

                whatsapp_service(payload)

                #response = await botApiService(payload)
                #Respond back with the same message
            #await sendMessage(from, You said: "${response}");
        
            # Respond with a 200 OK status
            return jsonify({"status": "success"}), 200
        
    except Exception as e:
        logging.error(f"call_webhook_endpoint: {e}")
        return jsonify({"error": "An internal error occurred."}), 500
    
@app.route('/webhook', methods=['GET'])
def webhook_get():
    # Retrieve all GET parameters
    get_params = request.args.to_dict()

    # Log the GET parameters
    print("Received GET parameters:")
    for key, value in get_params.items():
        print(f"{key}: {value}")

    # Respond to Facebook's verification challenge
    if "hub.challenge" in get_params:
        return get_params["hub.challenge"], 200

    return "Webhook received", 200

def whatsapp_service(payload):
    # Constants

    ACCESS_TOKEN='EAAyBDhP27soBO17ZCvnB61V7ao1EVUZBrysUtZADFRiuXvJO4Tfgol7ZBTouiB93AWWKRG6L6eZAT6yZAbZA03oRTQyas1NToEropj7USs1vCbrdZBWDRoKDVhDWzGQjJCZAK65IR2iTlunEZCJ14Mdf5ykXJhnw3MDYSfk3qZBZCExkZAc9DvrBvUXyHA1h6USZABIZATCE3Q1ZC6EITZB8mAjISGBHLL4WLk1WPdClOInyLgCtnTW8ZD'
    VERSION='v21.0'
    PHONE_NUMBER_ID = payload['phone_number_id']
    # URL and headers
    url = f"https://graph.facebook.com/{VERSION}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    
    # Payload
    payload = {
        "messaging_product": "whatsapp",
        "to": payload['messageFrom'],
        "type": "text",
        "text": {"body": payload['botResponse']},
    }

    #print("Payload:", payload)
    
    # Try to send the request
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        print("Message sent successfully:", response.json())
        return response.json()  # Return response data for further handling if needed
    except requests.exceptions.RequestException as e:
        # Log the error
        error_message = e.response.json() if e.response else str(e)
        print("Error sending message:", error_message)
        raise Exception("Failed to send message") from e
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(port=5001, debug=True)