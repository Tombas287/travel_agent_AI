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
