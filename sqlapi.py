from flask import Flask, jsonify
import mysql.connector

app = Flask(__name__)

# MySQL connection setup
def connect_db():
    connection = mysql.connector.connect(
        host="metrochatbot.cliqswukc30i.eu-north-1.rds.amazonaws.com",
        user="admin",
        password="Mayank2503",
        database="metro_chatbot"
    )
    return connection

# Fetch FAQs from MySQL database
@app.route('/faqs', methods=['GET'])
def get_faqs():
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)     
    cursor.execute("SELECT question, answer FROM metro_chatbot.cleaned_faqs")
    faqs = cursor.fetchall()
    connection.close()
    
    response = {"faqs": faqs}  # Structuring response as a dictionary
    return jsonify(response)

# Fetch Intents from MySQL database
@app.route('/intents', methods=['GET'])
def get_intents():
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT question, answer FROM metro_chatbot.cleaned_intent")
    intents = cursor.fetchall()
    connection.close()

    response = {"intents": intents}  # Structuring response as a dictionary
    return jsonify(response)

if __name__ == '__main__':
     app.run(port=8080,debug=True)



