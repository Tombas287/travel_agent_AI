import json
from typing import Any
import requests
import os
import pandas as pd
import re
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
import PyPDF2
from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain_community.utilities import SearxSearchWrapper
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# API Keys from .env
serp_api_key = os.getenv("serp_api_key")
RAPIDAPI_KEY = os.getenv("FLIGHT_API")
api_key = os.getenv('API_KEY')
from cities import default_cities
# Mapping cities to their airport codes

CITY_TO_AIRPORT_CODE = {
    "Mumbai": "BOM",
    "Bangalore": "BLR",
    "Delhi": "DEL",
    "Chennai": "MAA",
    "Kolkata": "CCU",
    "Hyderabad": "HYD",
    "Goa": "GOI",
    "Pune": "PNQ",
    "Ahmedabad": "AMD",
    "Kochi": "COK",
    "Jaipur": "JAI",
    "Chandigarh": "IXC",
    "Guwahati": "GAU",
    "New York": "JFK",
    "Los Angeles": "LAX",
    "San Francisco": "SFO",
    "London": "LHR",
    "Paris": "CDG",
    "Berlin": "BER",
    "Tokyo": "HND",
    "Beijing": "PEK",
    "Shanghai": "PVG",
    "Dubai": "DXB",
    "Sydney": "SYD",
    "Singapore": "SIN",
    "Toronto": "YYZ",
    "Mexico City": "MEX",
    "Cairo": "CAI",
    "Moscow": "SVO",
    "Istanbul": "IST",
    "Seoul": "ICN",
    "Bangkok": "BKK",
    "Cape Town": "CPT",
    "Rio de Janeiro": "GIG",
    "Buenos Aires": "EZE"
}

# Search details using SearxSearchWrapper or SerpAPIWrapper
search = SerpAPIWrapper(serpapi_api_key=serp_api_key)
wikipedia_tool = WikipediaAPIWrapper()
# Function to match city to airport code using regex and a mapping

def get_airport_code(city_name: str) -> str:
    city_name = city_name.strip().lower()  # Convert to lowercase for case-insensitive matching
    if city_name in CITY_TO_AIRPORT_CODE:
        return CITY_TO_AIRPORT_CODE[city_name]
    for city, code in CITY_TO_AIRPORT_CODE.items():
        if re.search(city.lower(), city_name):  # Match city variations
            return code
    return None  # If no match found


# FlightSearchTool (LangChain Tool)
class FlightSearchTool(BaseTool):
    name: str = "FlightSearch"
    description: str = "Searches for flights from one destination to another using the RapidAPI flights API."

    def _run(self, origin: str, destination: str, date: str) -> str:
        url = "https://flights-sky.p.rapidapi.com/flights/search-one-way"
        querystring = {"fromEntityId": origin, "toEntityId": destination, "departDate": date, "currency": "INR"}
        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,
            "x-rapidapi-host": "flights-sky.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            try:
                itineraries = response.json().get('data', {}).get('itineraries', [])
                total_results_found = response.json().get('data', {}).get('context', {}).get('totalResults', 'Empty')
                if total_results_found != 'Empty':
                    flight_details = []
                    for iter in itineraries:
                        price = iter['price']['formatted']
                        origin = iter['legs'][0]['origin']['city']
                        dest = iter['legs'][0]['destination']['city']
                        duration = iter['legs'][0]['durationInMinutes']
                        hours = duration // 60
                        minutes = duration % 60
                        departure = iter['legs'][0]['departure']
                        departure_time = datetime.strptime(departure, "%Y-%m-%dT%H:%M:%S").strftime("%H:%M")
                        arrival = iter['legs'][0]['arrival']
                        arrival_time = datetime.strptime(arrival, "%Y-%m-%dT%H:%M:%S").strftime("%H:%M")
                        flight_details.append({
                            'price': price,
                            'origin': origin,
                            'duration': f'{hours}H:{minutes}M',
                            'destination': dest,
                            'departure': departure_time,
                            'arrival': arrival_time
                        })
                    return pd.DataFrame(flight_details)  # Return as a DataFrame
                else:
                    return "No results found."
            except Exception as e:
                return f"Error occurred: {e}"
        else:
            return f"Error: Received response with status code {response.status_code}"


# Weather forecast of the destination place
class WeatherReport(BaseTool):
    name: str = "Weather report of the destination"
    description: str = "This provides insights into the weather report of the destination."

    def _run(self, city_name: str) -> str:
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city_name,
            'units': 'metric',
            'appid': api_key,
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            weather_info = {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'weather': data['weather'][0]['description'],
            }
            return (
                f"Weather report for {weather_info['city']}:\n"
                f"Temperature: {weather_info['temperature']}°C\n"
                f"Humidity: {weather_info['humidity']}%\n"
                f"Condition: {weather_info['weather']}"
            )
        return f"Failed with {response.status_code} and {response.reason}"


# Tourism Places using DuckDuckGoSearch or SerpAPI
tourism_prompt_template = PromptTemplate(
    input_variables=["destination"],
    template=(
        "1. Give me detailed information about 5 must-visit tourist places in {destination}. Include the following for each place: 1) Name 2) History 3) Cultural Significance 4) Main Activities 5) Nearby Attractions 6) Best Time to Visit \n"
        "2. If you can't provide complete historical details for each place, include a reference link to an external source like Wikipedia for further reading."

    )
)
search_tool = DuckDuckGoSearchRun()  # You can also replace this with SerpAPIWrapper

# Set up LangChain tools
tools = [
    Tool(
        name="FlightSearch",
        func=FlightSearchTool()._run,
        description="Fetch flight details based on origin, destination, and date."
    ),
    # Tool(
    #     name="DestinationSummary",
    #     func=search_tool.run,
    #     description="Fetches a brief summary of the tourism destination with tourist places."
    # ),
    Tool(
        name="WeatherReport",
        func=WeatherReport()._run,
        description="Fetches the weather report of the destination."
    ),
    Tool(
        name="SearxSearch",
        func=search.run,  # Using SerpAPIWrapper's run method for search
        description="Search the tourism destination with tourist places."
    ),
    Tool(
        name="WikipediaSummary",
        func=wikipedia_tool.run,
        description="Fetches the Wikipedia summary for a given destination."
    )
]

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GENAI_API_KEY"),
    temperature=0.5
)

# Create LangChain Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Streamlit App Layout
st.set_page_config(layout="wide", page_title="Flight and Tourism Assistant")
st.title("Flight and Tourism Assistant")

# Sidebar Inputs
# origin_input = st.sidebar.text_input("Enter the origin city name (e.g., Mumbai):").lower()
origin_input = st.sidebar.selectbox(placeholder="Choice", options=default_cities(), label='origin').lower()
destination_input = st.sidebar.selectbox(placeholder="Choice", options=default_cities(), label='dest').lower()
date_input = st.sidebar.date_input("Choose a departure date:", min_value=datetime.today())

# Button for generating response
submit_button = st.sidebar.button("Search Flights and Tourist Info")


# Function to initialize or check the attempts file

def attempts() -> int:
    if not os.path.exists('attempts.json'):
        # Create the file with default attempts
        with open("attempts.json", 'w') as file:
            json.dump({'attempts': 5}, file, indent=2)

    # If not empty, load the JSON data
    with open('attempts.json', 'r') as file:
        data = json.load(file)
        return data.get("attempts", 5)  # Default to 5 if the key is missing

# Function to update the attempts value
def update_attempts(no_items: int):
    with open('attempts.json', 'w') as file:
        json.dump({'attempts': no_items}, file)


no_attempts = attempts()
st.sidebar.write(f"Number of attempts remaining: {no_attempts}")
# When submit button is pressed, execute the logic
if submit_button and origin_input and destination_input and date_input:
    if no_attempts > 0:

        with st.spinner("Processing your request..."):

            try:
                # Get the airport codes for the cities
                origin_code = get_airport_code(origin_input)
                destination_code = get_airport_code(destination_input)

                # Check if both airport codes are valid
                if not origin_code or not destination_code:
                    st.error("Invalid city names entered. Please check and try again.")
                else:
                    # Fetch flight information
                    flight_data = FlightSearchTool()._run(origin_code, destination_code, date_input)
                    # tourist_data = search_tool.run(destination_input)
                    weather_report = WeatherReport()._run(destination_input)

                    # Display flight data
                    if isinstance(flight_data, pd.DataFrame):
                        st.subheader("Flight Search Results:")
                        st.dataframe(flight_data)  # Display as DataFrame
                    else:
                        st.error(flight_data)  # Error message from the flight tool

                    # Generate detailed tourism places information
                    tourism_info = tourism_prompt_template.format(destination=destination_input)
                    destination = agent.run(input=tourism_info)

                    # Display tourism places information
                    st.subheader(f"Tourism Places in {destination_input}:")
                    st.write(destination)

                    # Display weather report
                    st.subheader(f"Weather Report for {destination_input}:")
                    st.write(weather_report)
                    no_attempts -= 1
                    update_attempts(no_attempts)

            except Exception as e:
                st.error(f"Error processing request: {e}")

