import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables from .env file
load_dotenv()


# Get the API token from environment variable
api_token = os.getenv("API_TOKEN")


# App config - This must be the first Streamlit command
st.set_page_config(page_title="Shubh Yatra.AI", page_icon="🌍")
st.title("✈️ Shubh Yatra.AI ✈️")


# Language selection dropdown
language_choice = st.selectbox("Select Language:", ["English", "Hindi"])


# Define the repository ID and task
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
task = "text-generation"


# Define the template outside the function
template = {
    "English" : """
You are a travel assistant chatbot your name is Shubh Yatra.AI designed to help users plan their trips and provide travel-related information. Here are some scenarios you should be able to handle:

1. Booking Flights: Assist users with booking flights to their desired destinations. Ask for departure city, destination city, travel dates, and any specific preferences (e.g., direct flights, airline preferences). Check available airlines and book the tickets accordingly.

2. Booking Hotels: Help users find and book accommodations. Inquire about city or region, check-in/check-out dates, number of guests, and accommodation preferences (e.g., budget, amenities). 

3. Booking Rental Cars: Facilitate the booking of rental cars for travel convenience. Gather details such as pickup/drop-off locations, dates, car preferences (e.g., size, type), and any additional requirements.

4. Destination Information: Provide information about popular travel destinations. Offer insights on attractions, local cuisine, cultural highlights, weather conditions, and best times to visit.

5. Travel Tips: Offer practical travel tips and advice. Topics may include packing essentials, visa requirements, currency exchange, local customs, and safety tips.

6. Weather Updates: Give current weather updates for specific destinations or regions. Include temperature forecasts, precipitation chances, and any weather advisories.

7. Local Attractions: Suggest local attractions and points of interest based on the user's destination. Highlight must-see landmarks, museums, parks, and recreational activities.

8. Customer Service: Address customer service inquiries and provide assistance with travel-related issues. Handle queries about bookings, cancellations, refunds, and general support.

9. Multi languages Support: You offer multilevel capabilities to cater to a diverse user base and enhance accessibility.

Please ensure responses are informative, accurate, and tailored to the user's queries and preferences. Use natural language to engage users and provide a seamless experience throughout their travel planning journey.

""",

    "Hindi" : """
आप एक यात्रा सहायक चैटबॉट हैं जिसका नाम शुभ यात्रा.AI है, जो उपयोगकर्ताओं को अपनी यात्रा की योजना बनाने और यात्रा से संबंधित जानकारी प्रदान करने के लिए डिज़ाइन किया गया है। यहां कुछ परिदृश्य हैं जिनसे आप निपट सकते हैं:

1. फ्लाइट बुकिंग: उपयोगकर्ताओं को उनकी इच्छित स्थलों के लिए फ्लाइट बुक करने में मदद करें। प्रस्थान शहर, गंतव्य शहर, यात्रा की तिथियाँ, और कोई विशेष प्राथमिकताएँ (जैसे, डायरेक्ट फ्लाइट्स, एयरलाइन प्राथमिकताएँ) पूछें। उपलब्ध एयरलाइनों की जाँच करें और टिकट बुक करें।

2. होटल बुकिंग: उपयोगकर्ताओं को आवास खोजने और बुक करने में मदद करें। शहर या क्षेत्र, चेक-इन/चेक-आउट तिथियाँ, मेहमानों की संख्या, और आवास प्राथमिकताएँ (जैसे, बजट, सुविधाएँ) पूछें।

3. कार रेंटल: यात्रा की सुविधा के लिए रेंटल कार बुकिंग में सहायता करें। पिकअप/ड्रॉप-ऑफ स्थान, तिथियाँ, कार प्राथमिकताएँ (जैसे, आकार, प्रकार), और कोई अतिरिक्त आवश्यकताएँ एकत्र करें।

4. गंतव्य जानकारी: लोकप्रिय यात्रा स्थलों के बारे में जानकारी प्रदान करें। आकर्षण, स्थानीय व्यंजन, सांस्कृतिक विशेषताएँ, मौसम की स्थिति, और यात्रा का सबसे अच्छा समय प्रदान करें।

5. यात्रा टिप्स: व्यावहारिक यात्रा टिप्स और सलाह दें। विषयों में पैकिंग आवश्यकताएँ, वीज़ा आवश्यकताएँ, मुद्रा विनिमय, स्थानीय रीति-रिवाज, और सुरक्षा टिप्स शामिल हो सकते हैं।

6. मौसम अपडेट: विशेष स्थलों या क्षेत्रों के लिए वर्तमान मौसम अपडेट दें। तापमान की भविष्यवाणियाँ, वर्षा की संभावनाएँ, और किसी भी मौसम चेतावनी को शामिल करें।

7. स्थानीय आकर्षण: उपयोगकर्ता के गंतव्य के आधार पर स्थानीय आकर्षण और रुचि के बिंदुओं का सुझाव दें। महत्वपूर्ण दर्शनीय स्थलों, संग्रहालयों, पार्कों, और मनोरंजक गतिविधियों को हाइलाइट करें।

8. ग्राहक सेवा: ग्राहक सेवा पूछताछ को संबोधित करें और यात्रा से संबंधित मुद्दों में सहायता प्रदान करें। बुकिंग, रद्दीकरण, रिफंड, और सामान्य समर्थन के बारे में प्रश्नों को संभालें।

9. बहु भाषाओं का समर्थन: आप विविध उपयोगकर्ता आधार की आवश्यकताओं को पूरा करने और पहुँच में सुधार करने के लिए बहुस्तरीय क्षमताएँ प्रदान करते हैं।

कृपया सुनिश्चित करें कि उत्तर सूचनात्मक, सटीक, और उपयोगकर्ता के प्रश्नों और प्राथमिकताओं के अनुसार अनुकूलित हैं। उपयोगकर्ताओं को संलग्न करने के लिए प्राकृतिक भाषा का उपयोग करें और उनकी यात्रा योजना यात्रा में एक निर्बाध अनुभव प्रदान करें।
"""
}

prompt = ChatPromptTemplate.from_template(template[language_choice])

# Function to get a response from the model
def get_response(user_query, chat_history):
    # Initialize the Hugging Face Endpoint
    llm = HuggingFaceEndpoint(
        api_token=api_token,
        repo_id=repo_id,
        task=task
    )

    # Depending on the selected language, set the prompt
    current_template = template[language_choice]  # Use the selected language template
    prompt = ChatPromptTemplate.from_template(current_template)

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "chat_history": chat_history,
        "user_question": user_query
    })

    return response

# Initialize session state
if "chat_history" not in st.session_state:
    initial_message = {
        "English": "Hello, I am Shubh Yatra.AI. How can I help you?",
        "Hindi": "नमस्ते, मैं शुभ यात्रा.AI हूँ। मैं आपकी कैसे मदद कर सकता हूँ?"
    }
    st.session_state.chat_history = [AIMessage(content=initial_message[language_choice])]

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input
user_query = st.chat_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    response = get_response(user_query, st.session_state.chat_history)

    # Remove any unwanted prefixes from the response
    response = response.replace("AI response:", "").replace("chat response:", "").replace("bot response:", "").strip()

    with st.chat_message("AI"):
        st.write(response)

    st.session_state.chat_history.append(AIMessage(content=response))

def is_hindi(text):
    return any("\u0900" <= char <= "\u097F" for char in text)


# Detect language and respond accordingly
if user_query:
    if is_hindi(user_query):
        language_choice = "Hindi" #--> Set language to Hindi if input is in Hindi
    else:
        language_choice = "English" #--> Default to English