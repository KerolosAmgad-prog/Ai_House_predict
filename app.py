import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import numpy as np

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# Environmental Configuration
load_dotenv()
OPENROUTER_API_KEY = os.getenv("Deepseek_Api_Key")
CHROMA_PATH = "Chroma_db"

# Initialize FastAPI
app = FastAPI(title="Real Estate AI Project", description="House Price Prediction & Interpretation with RAG Chatbot")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
try:
    regression_model = joblib.load("models/best_house_price_predictor.pkl")
    classifier_model = joblib.load("models/category_classifier_pipeline.pkl")
    # Some scikit-learn versions require specific feature names
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    regression_model = None
    classifier_model = None

# Initialize LLM (DeepSeek via OpenRouter)
llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="deepseek/deepseek-chat", # Use standard DeepSeek model
)

# Initialize RAG
embeddings_model = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Load rank mappings from data.csv on startup
CITY_RANK_MAP = {}
ZIP_RANK_MAP = {}

try:
    df_raw = pd.read_csv("data.csv")
    CITY_RANK_MAP = df_raw.groupby('city')['price'].mean().to_dict()
    ZIP_RANK_MAP = df_raw.groupby('statezip')['price'].mean().to_dict()
    print("Rank mappings initialized from data.csv")
except Exception as e:
    print(f"Error initializing ranks: {e}")

# Data Models
class HouseFeatures(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int
    city: str = "Seattle"  # Default
    statezip: str = "WA 98101" # Default

class PredictionResponse(BaseModel):
    predicted_price: float
    category: str
    interpretation: str

class ChatRequest(BaseModel):
    message: str

# Define a system prompt for the chatbot
SYSTEM_PROMPT = """
You are "EstateAI Assistant", a professional and friendly real estate consultant.
Your goal is to provide accurate and concise information based on the provided context.

GUIDELINES:
1. If the user says "hi" or greets you, respond with a friendly greeting and ask how you can help them with real estate today. Do NOT dump all the knowledge data immediately.
2. Only use the provided context to answer specific technical or data-related questions.
3. If the user asks something outside the scope of real estate or the provided data, politely inform them.
4. Keep your answers professional and helpful.
"""

qa_prompt = PromptTemplate(
    template=SYSTEM_PROMPT + "\nContext: {context}\n\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"]
)

# Re-initialize QA Chain with the prompt
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)

# Endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict_and_interpret(features: HouseFeatures):
    if not regression_model or not classifier_model:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        # FEATURE ENGINEERING
        # Calculate Ranks
        city_rank = CITY_RANK_MAP.get(features.city, sum(CITY_RANK_MAP.values())/len(CITY_RANK_MAP))
        zip_rank = ZIP_RANK_MAP.get(features.statezip, sum(ZIP_RANK_MAP.values())/len(ZIP_RANK_MAP))
        
        # Other Engineered features
        home_age = 2014 - features.yr_built
        renovated = 1 if features.yr_renovated > 0 else 0
        has_basement = 1 if features.sqft_basement > 0 else 0
        sqft_living_total = features.sqft_above + features.sqft_basement

        # Create DataFrames with exact feature names and order
        # 1. Regression features (expects engineered features)
        reg_data = pd.DataFrame([{
            'sqft_living': features.sqft_living,
            'sqft_lot': features.sqft_lot,
            'bathrooms': features.bathrooms,
            'bedrooms': features.bedrooms,
            'view': features.view,
            'zip_rank': zip_rank,
            'home_age': home_age,
            'waterfront': features.waterfront,
            'condition': features.condition,
            'sqft_above': features.sqft_above,
            'renovated': renovated,
            'floors': features.floors,
            'city_rank': city_rank,
            'has_basement': has_basement
        }])

        # 2. Classification features (based on the error, it expects RAW features)
        # Order: bedrooms, bathrooms, sqft_living, sqft_lot, floors, condition, waterfront, view, sqft_above, sqft_basement
        clf_data = pd.DataFrame([{
            'bedrooms': features.bedrooms,
            'bathrooms': features.bathrooms,
            'sqft_living': features.sqft_living,
            'sqft_lot': features.sqft_lot,
            'floors': features.floors,
            'condition': features.condition,
            'waterfront': features.waterfront,
            'view': features.view,
            'sqft_above': features.sqft_above,
            'sqft_basement': features.sqft_basement
        }])

        # Prediction
        price_pred_log = regression_model.predict(reg_data)[0]
        category_pred = classifier_model.predict(clf_data)[0]
        
        # Inverse log transformation (model was trained on log_price)
        price_pred = float(np.expm1(price_pred_log))
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    # Interpretation via LLM
    prompt = f"""
    As a real estate expert, interpret the following house evaluation:
    - Features: {features.model_dump()}
    - Predicted Price: ${price_pred:,.2f}
    - Predicted Category: {category_pred}
    
    Please explain:
    1. Why was this price predicted based on factors like size, age ({home_age} years), and location ({features.city})?
    2. Is this price realistic compared to a typical market for these features?
    3. What are the strongest factors influencing this valuation?
    Keep it professional and concise.
    """
    
    interpretation = llm.invoke(prompt).content

    return {
        "predicted_price": float(price_pred),
        "category": str(category_pred),
        "interpretation": interpretation
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = qa_chain.invoke({"question": request.message})
        return {"response": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "running"}

@app.get("/metadata")
async def get_metadata():
    # Return cities and zips for the frontend dropdowns
    return {
        "cities": sorted(list(CITY_RANK_MAP.keys())),
        "zips": sorted(list(ZIP_RANK_MAP.keys()))
    }

# Serve static files (frontend) - this should be at the end
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
