import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import pickle #for pickle files



st.set_page_config(
        page_title="FlightPricePredictiom",
)

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#Code for loading the model from the pickle file
def load_model():
    with open('/Users/jannathshaik/Desktop/python/projects/flight_price_prediction/model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data 


#code for scaling the data passed into the function
@st.cache_data(persist=True)
def scale_input_data(input_data, scaler):
    scaled_data = scaler.transform(input_data)
    return scaled_data[0]


# Load the data from the CSV file into a DataFrame
file_path = '/Users/jannathshaik/Desktop/python/projects/flight_price_prediction/Clean_Dataset.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

#code for loading the model
regressor = load_model()

def predict_ticket_price(airline,departure_time, stops, arrival_time,class_type, duration, days_left):
    # Encode the categorical features
    airline_mapping = {'SpiceJet':4, 'AirAsia':0, 'Vistara':5, 'GO_FIRST':2, 'Indigo':3, 'Air_India':1}
    #source_city_mapping = {'Delhi': 2, 'Mumbai': 5, 'Bangalore': 0, 'Kolkata': 4, 'Hyderabad': 3, 'Chennai': 1}
    departure_time_mapping = {'Morning': 4, 'Early_Morning': 1, 'Evening': 2, 'Night': 5, 'Afternoon': 0, 'Late_Night': 3}
    arrival_time_mapping = {'Night': 5, 'Evening': 2, 'Morning': 4, 'Afternoon': 0, 'Early_Morning': 1, 'Late_Night': 3}
    #destination_city_mapping = {'Mumbai': 5, 'Delhi': 2, 'Bangalore': 0, 'Kolkata': 4, 'Hyderabad': 3, 'Chennai': 1}
    class_mapping = {'Economy': 1, 'Business': 0}
    stops_mapping = {'zero': 2, 'one': 0, 'two_or_more': 1}
    
    encoded_airline = airline_mapping.get(airline)
    #encoded_source_city = source_city_mapping.get(source_city)
    encoded_departure_time = departure_time_mapping.get(departure_time)
    encoded_arrival_time = arrival_time_mapping.get(arrival_time)
    #encoded_destination_city = destination_city_mapping.get(destination_city)
    encoded_class = class_mapping.get(class_type)
    encoded_stops = stops_mapping.get(stops.lower())  # Convert stops value to lowercase before encoding

    new_min = 0
    new_max = 1  # Set your desired maximum value

    duration_min = df['duration'].min()
    duration_max = df['duration'].max()
    days_left_min = df['days_left'].min()
    days_left_max = df['days_left'].max()
    price_min = df['price'].min()
    price_max = df['price'].max()

# Scale 'duration' and 'days_left' as per model data
    scaled_duration = (duration - duration_min) / (duration_max - duration_min) * (new_max - new_min) + new_min
    scaled_days_left = (days_left - days_left_min) / (days_left_max - days_left_min) * (new_max - new_min) + new_min
    
    # Create a list with the encoded features
    input_data = [encoded_airline, encoded_departure_time, encoded_stops, encoded_arrival_time,encoded_class, scaled_duration, scaled_days_left]
    
    # Make the prediction using the best model that is XGBRegressor .
    predicted_price = (regressor.predict([input_data])[0]).round(2) 
    
    reversed_predicted_value = round(((predicted_price - new_min) * (price_max - price_min) / (new_max - new_min) + price_min),2)
    
    return reversed_predicted_value


header = st.container()
selection = st.container()

# Create a Streamlit sidebar with options
st.sidebar.header("Flight Price Options")
with header:
    centered_title = "<h1 style='text-align: center;'>Flight Price Prediction</h1>"
    st.markdown(centered_title, unsafe_allow_html=True)
    
with st.sidebar:
    option = option_menu("Menu", ['📈Prediction','🔎About'], default_index=0)
    
if option == '📈Prediction':

    # Create two columns layout
    col1, col2 = st.columns(2)

    # Dropdowns for user input in the first column
    with col1:
        selected_airline_list = ['None'] + df['airline'].unique().tolist()
        selected_airline = st.selectbox("Select Airline", selected_airline_list)
        selected_departure_city_list = ['None'] + df['source_city'].unique().tolist()
        selected_departure_city = st.selectbox("Select Departure City", selected_departure_city_list)
        selected_departure_time_list = ['None'] +  df['departure_time'].unique().tolist()
        selected_departure_time = st.selectbox("Select Departure Time", selected_departure_time_list)
        selected_stops_list = ['None'] + df['stops'].unique().tolist()
        selected_stops = st.selectbox("Select Number of Stops", selected_stops_list)
       

    # Dropdowns for user input in the second column
    with col2:
        selected_arrival_city_list = ['None'] + df['destination_city'].unique().tolist()
        selected_arrival_city = st.selectbox("Select Arrival City", selected_arrival_city_list)
        selected_arrival_time_list = ['None'] + df['arrival_time'].unique().tolist()
        selected_arrival_time = st.selectbox("Select Arrival Time",selected_arrival_time_list)
        selected_class_list = ['None'] + df['class'].unique().tolist()
        selected_class = st.selectbox("Select Class", selected_class_list)
    
    selected_duration= st.number_input('Select Duration of Journey',min_value=0.80, max_value=50.0, step=0.05)
    selected_days_left = st.number_input('Select Days Left',min_value=1, max_value=50, step=1)

    price = predict_ticket_price(selected_airline, selected_departure_time, 
                                selected_stops,selected_arrival_time, selected_class,selected_duration,selected_days_left)
    
    output = st.container()
    
    with output:
        if(selected_airline != 'None' and selected_departure_city != 'None' and selected_departure_time != 'None' 
           and selected_stops != 'None' and selected_arrival_city != 'None' and selected_arrival_time != 'None' 
           and selected_class != 'None' and selected_duration!=0.00 and selected_days_left!= 0):
            #write code to show the predicted price here after running the model
            st.subheader(f'Predicted Price: ₹{price}')
        else:    
            st.warning("Please fill all the necessary options", icon="⚠️")
    
    
    

if option == '🔎About':
    st.write("Improving flight prediction models' accuracy is crucial as the world grows more interconnected and air \
        travel remains a crucial part of international trade and transportation. More accurate and reliable flight prediction \
        tools could have a significant positive impact on the aviation industry, helping with tasks like arrival and departure \
        time prediction, delay estimation, and resource allocation optimization. In order to achieve that goal, \
            this model attempts to offer important insightsin the context of flight price prediction")
    st.write("To view the code: [Click Here](https://colab.research.google.com/drive/16W2EDpbeiLb_mxAVxT6y8yWxX7ZZu9AN#scrollTo=317d8f31-a24a-416c-b366-8d70b2848e42)")
    st.write("To access the Dataset: [Click Here](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction?select=Clean_Dataset.csv)")
