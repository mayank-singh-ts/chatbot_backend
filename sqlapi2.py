from flask import Flask, jsonify, request
import mysql.connector

app = Flask(__name__)

# MySQL Database connection
def get_db_connection():
    connection = mysql.connector.connect(
        host="metrochatbot.cliqswukc30i.eu-north-1.rds.amazonaws.com",  # Use 'localhost' if MySQL is on your local machine
        user="admin",
        password="Mayank2503",
        database='chatbot_1'  # Ensure this database exists
    )
    return connection

@app.route('/get_route_and_fare', methods=['GET'])
def get_route_and_fare():
    # Get station names from user input (query parameters)
    from_station = request.args.get('from_station', default='Green Park')
    to_station = request.args.get('to_station', default='Vaishali')

    try:
        # Connect to the database
        connection = get_db_connection()
        cursor = connection.cursor()

        # First SQL query: fetch stations, fare, and zones from top to bottom
        top_to_bottom_query = """
        SELECT 
            GROUP_CONCAT(DISTINCT s.station_name ORDER BY s.station_id SEPARATOR ', ') AS all_names, 
            (SELECT SUM(f.fare)
             FROM fares f
             WHERE f.source_station_id >= (SELECT s.station_id FROM stations s WHERE s.station_name = %s)
             AND f.destination_station_id <= (SELECT s.station_id FROM stations s WHERE s.station_name = %s)
            ) AS TotalFare,
            GROUP_CONCAT(DISTINCT s.zone ORDER BY s.zone SEPARATOR ', ') AS distinct_zones
        FROM (
            SELECT DISTINCT r.destination_station_id AS ID
            FROM routes r
            WHERE r.source_station_id >= (SELECT s.station_id FROM stations s WHERE s.station_name = %s)
            AND r.destination_station_id <= (SELECT s.station_id FROM stations s WHERE s.station_name = %s)
            
            UNION ALL

            SELECT DISTINCT r.source_station_id AS ID
            FROM routes r
            WHERE r.source_station_id >= (SELECT s.station_id FROM stations s WHERE s.station_name = %s)
            AND r.destination_station_id <= (SELECT s.station_id FROM stations s WHERE s.station_name = %s)
        ) AS a
        INNER JOIN stations s ON s.station_id = a.ID;
        """

        # Execute the first query (top to bottom)
        cursor.execute(top_to_bottom_query, (from_station, to_station, from_station, to_station, from_station, to_station))
        top_to_bottom_result = cursor.fetchone()

        if top_to_bottom_result and top_to_bottom_result[0] is not None:
            # If the first query returns a result, send it back
            full_route = top_to_bottom_result[0]
            total_fare = top_to_bottom_result[1]
            distinct_zones = top_to_bottom_result[2]
            route_data = {
                'full_route': full_route,
                'total_fare': total_fare,
                'distinct_zones': distinct_zones
            }
        else:
            # If no result is found, run the second query (bottom to top)
            bottom_to_top_query = """
            SELECT 
                GROUP_CONCAT(DISTINCT s.station_name ORDER BY a.ID DESC SEPARATOR ', ') AS all_names,  
                SUM(f.fare) AS TotalFare,                                                          
                GROUP_CONCAT(DISTINCT s.zone ORDER BY s.zone SEPARATOR ', ') AS distinct_zones
            FROM (
                SELECT r1.source_station_id AS ID, r1.destination_station_id
                FROM routes r1
                WHERE (r1.source_station_id >= (SELECT station_id FROM stations WHERE station_name = %s)
                  AND r1.destination_station_id <= (SELECT station_id FROM stations WHERE station_name = %s))
                  OR r1.destination_station_id = (SELECT station_id FROM stations WHERE station_name = %s)

                UNION  

                SELECT r2.source_station_id AS ID, r2.destination_station_id
                FROM routes r2
                WHERE (r2.source_station_id >= (SELECT station_id FROM stations WHERE station_name = %s)
                  AND r2.destination_station_id <= (SELECT station_id FROM stations WHERE station_name = %s))
                 OR r2.source_station_id = (SELECT station_id FROM stations WHERE station_name = %s)

                ORDER BY ID DESC
            ) AS a
            INNER JOIN stations s ON s.station_id = a.ID         
            INNER JOIN fares f ON f.source_station_id = a.ID AND f.destination_station_id = a.destination_station_id;
            """

            # Execute the second query (bottom to top)
            cursor.execute(bottom_to_top_query, (to_station, from_station, from_station, to_station, from_station, to_station))
            bottom_to_top_result = cursor.fetchone()

            if bottom_to_top_result and bottom_to_top_result[0] is not None:
                # Return result from bottom-to-top if found
                full_route = bottom_to_top_result[0]
                total_fare = bottom_to_top_result[1]
                distinct_zones = bottom_to_top_result[2]
                route_data = {
                    'full_route': full_route,
                    'total_fare': total_fare,
                    'distinct_zones': distinct_zones
                }
            else:
                # If no route is found at all, return an error
                route_data = {'error': 'No route found between the provided stations'}

        # Close the connection
        cursor.close()
        connection.close()

        # Print route_data for debugging
        print(route_data)

        # Return the result as JSON
        return jsonify(route_data)

    except Exception as e:
        # Log and return any exceptions
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)})

# Correct block to run the app
if __name__ == '__main__':
    app.run(debug=True)
