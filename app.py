import os
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from dotenv import load_dotenv

from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.5-flash"

st.set_page_config(layout="wide")

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})


SIGNAL_MAP = {
    "Green": 1.0,
    "Yellow": 2.0,
    "Red": 3.0,
    "Long Red": 4.0
}

INCIDENT_MAP = {
    "None": 0.0,
    "Minor": 0.5,
    "Moderate": 1.0,
    "Major": 2.0
}


def get_streamlit_secret(name: str) -> Optional[str]:
    try:
        return st.secrets.get(name)
    except Exception:
        return None

def load_api_key() -> Optional[str]:
    load_dotenv()
    return (
        os.getenv("GEMINI_API_KEY")
        or get_streamlit_secret("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

def create_client(api_key: Optional[str]):
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)

api_key = load_api_key()
client = create_client(api_key)


@st.cache_data
def load_data():
    try:
        df = pd.read_csv('vanet_traffic_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_regressor_model():
    try:
        return joblib.load('WaitTimeRegressor.model')
    except FileNotFoundError:
        return None

@st.cache_resource
def load_classifier_model():
    try:
        return joblib.load('TrafficTypeClassifierLite.model')
    except FileNotFoundError:
        return None

df = load_data()
reg_model = load_regressor_model()
classifier_model = load_classifier_model()
traffic_order = ['Free-flow', 'Moderate', 'Heavy', 'Gridlock']

def build_traffic_context(traffic_df: Optional[pd.DataFrame]) -> str:
    if traffic_df is None:
        return "df not found"
    
    total_records = len(traffic_df)
    avg_speed = traffic_df['avg_speed_kmph'].mean()
    avg_delay = traffic_df['avg_comm_delay_ms'].mean()
    
    summary_lines = [
        "Traffic Data summary",
        f"- Total System Log Records: {total_records} rows",
        f"- Mean Network Velocity: {avg_speed:.2f} km/h",
        f"- Mean  Transmission Delay: {avg_delay:.2f} ms"
    ]
    return "\n".join(summary_lines)

def build_system_instruction() -> str:
    return "\n".join([
        "Your name is VANETBot, you are an AI assistant desgined to help manage VANET Network systems",
        "Your objective is to answer user questions within scope in english or vietnamese in concise and short responses",
        "Question scope includes, What VANET is and how it works, what certain metrics or traffic states mean and how they impact traffic and how they are dealt with"
    ])

def build_history_text(messages: list[dict], max_message: int = 8) -> str:
    recent_messages = messages[-max_message:]
    lines = []
    for message in recent_messages:
        role = "User" if message["role"] == "user" else "VANETBot"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)

def ask_bot(prompt: str, messages: list[dict], client, traffic_df: Optional[pd.DataFrame]) -> str:
    if client is None:
        return "API failed"
        
    system_instruction = build_system_instruction()
    traffic_context = build_traffic_context(traffic_df)
    history_text = build_history_text(messages)
    
    user_content = (
        "Current traffic data:\n" + traffic_context +
        "\n\nChat history:\n" + history_text +
        "\n\nUser Prompt:\n" + prompt
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            )
        )
        return response.text or "An error occurred."
    except Exception as exc:
        return f"Error: {exc}"


with st.sidebar:
    st.title("VANET AI Assistant")
    st.markdown("---")
    
    if not api_key:
        st.error("API KEY NOT FOUND")


    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask VANETBot a question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner():
                response_text = ask_bot(prompt=prompt, messages=st.session_state.messages, client=client, traffic_df=df)
                st.markdown(response_text)
                
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response_text})


st.title("VANET Traffic Dashboard")

tab1, tab2, tab3 = st.tabs([
    "Data Visualization", 
    "Wait Time Estimation", 
    "Traffic State Classification"
])

with tab1:
    st.header("Data Visualization")
    if df is None:
        st.warning("DF NOT FOUNd")
    else:
        df_sample = df.sample(min(1000, len(df)), random_state=42)
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)
        r3c1, r3c2 = st.columns(2)
        r4c1, r4c2 = st.columns(2)

        with r1c1:
            st.subheader("Distribution of Traffic Situations")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', cmap='tab20c', ax=ax)
            ax.set_ylabel('')
            st.pyplot(fig)

        with r1c2:
            st.subheader("Average Speed by Traffic Situation")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.barplot(x='label', y='avg_speed_kmph', data=df, order=traffic_order, palette='muted', ax=ax)
            st.pyplot(fig)

        with r2c1:
            st.subheader("Average Queue Length by Traffic Situation")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.barplot(x='label', y='queue_length_veh', data=df, order=traffic_order, palette='magma', ax=ax)
            st.pyplot(fig)

        with r2c2:
            st.subheader("Hourly Vehicle Density & Congestion Pressure")
            hourly_stats = df.groupby('hour')[['density_veh_per_km', 'congestion_pressure']].mean()
            fig, ax1 = plt.subplots(figsize=(6, 4.5))
            ax2 = ax1.twinx()
            sns.lineplot(x='hour', y='density_veh_per_km', data=hourly_stats, color='b', marker='o', ax=ax1)
            sns.lineplot(x='hour', y='congestion_pressure', data=hourly_stats, color='r', marker='s', ax=ax2)
            ax1.set_ylabel('Vehicle Density (veh/km)', color='b')
            ax2.set_ylabel('Congestion Pressure', color='r')
            ax2.grid(False)
            st.pyplot(fig)

        with r3c1:
            st.subheader("Hourly Average Communication Delay")
            hourly_comm = pd.DataFrame(df.groupby('hour')['avg_comm_delay_ms'].mean())
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.lineplot(x='hour', y='avg_comm_delay_ms', data=hourly_comm, color='purple', marker='^', ax=ax)
            ax.set_ylabel('Delay (ms)')
            st.pyplot(fig)

        with r3c2:
            st.subheader("Vehicle Density vs. Average Speed")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.scatterplot(x='density_veh_per_km', y='avg_speed_kmph', hue='label', data=df_sample, palette='viridis', ax=ax)
            st.pyplot(fig)

        with r4c1:
            st.subheader("Packet Loss vs. Communication Delay")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sns.scatterplot(x='packet_loss_pct', y='avg_comm_delay_ms', hue='label', data=df_sample, palette='coolwarm', ax=ax)
            st.pyplot(fig)

        with r4c2:
            st.subheader("Correlation between Physical Traffic Metrics & VANET Network Metrics")
            fig, ax = plt.subplots(figsize=(7, 5.5))
            core_metrics = ['avg_speed_kmph', 'density_veh_per_km', 'avg_wait_time_s', 'flow_veh_per_hr', 
                            'queue_length_veh', 'congestion_pressure', 'packet_loss_pct', 'avg_comm_delay_ms']
            sns.heatmap(df[core_metrics].corr(), annot=True, cmap='coolwarm', fmt='.2f', square=True, annot_kws={"size": 8}, ax=ax)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

with tab2:
    st.header("Wait Time Estimation")
    if reg_model is None:
        st.error("Model not found")
    else:
        with st.form("reg_predict_form"):
            col1, col2 = st.columns(2)
            with col1:
                reg_flow = st.number_input("Traffic Flow (flow_veh_per_hr):", min_value=0.0, value=1200.0, key="reg_flow")
                reg_density = st.number_input("Traffic Density (density_veh_per_km):", min_value=0.0, value=35.0, key="reg_density")
            with col2:
                reg_queue = st.number_input("Queue Length (queue_length_veh):", min_value=0.0, value=15.0, key="reg_queue")
                
                reg_signal_label = st.selectbox(
                    "Current Signal State (signal_state):", 
                    options=list(SIGNAL_MAP.keys()), 
                    key="reg_signal_lbl"
                )
                reg_signal_num = SIGNAL_MAP[reg_signal_label]
            
            reg_submit = st.form_submit_button("Estimate")
            
        if reg_submit:
            reg_input = pd.DataFrame(
                [[reg_flow, reg_density, reg_queue, reg_signal_num]], 
                columns=['flow_veh_per_hr', 'density_veh_per_km', 'queue_length_veh', 'signal_state_num']
            )
            reg_prediction = reg_model.predict(reg_input)[0]
            st.markdown("---")
            st.metric(label="Predicted Target Wait Time", value=f"{reg_prediction:.2f} Seconds")

with tab3:
    st.header("Traffic State Classifier")
    if classifier_model is None:
        st.error("model not found")
    else:
        with st.form("classifier_form"):
            col1, col2 = st.columns(2)
            with col1:
                clf_flow = st.number_input("Traffic Flow (flow_veh_per_hr):", min_value=0.0, value=1200.0, key="clf_flow")
                clf_ratio = st.number_input("Speed Density Ratio (speed_density_ratio):", min_value=0.0, value=1.5, key="clf_ratio")
                clf_delay = st.number_input("Network Delay (avg_comm_delay_ms):", min_value=0.0, value=45.0, key="clf_delay")
                clf_cbr = st.number_input("Channel Busy Ratio (channel_busy_ratio_pct):", min_value=0.0, max_value=100.0, value=30.0, key="clf_cbr")
                
  
                clf_incident_label = st.selectbox(
                    "Active Incidents (incident_severity):", 
                    options=list(INCIDENT_MAP.keys()), 
                    key="clf_incident_lbl"
                )
                clf_incident_num = INCIDENT_MAP[clf_incident_label]
                
            with col2:
                clf_wireless = st.number_input("Wireless Congestion Intensity:", min_value=0.0, value=0.2, key="clf_wireless")
                clf_density = st.number_input("Vehicle Density (density_veh_per_km):", min_value=0.0, value=35.0, key="clf_density")
                clf_speed = st.number_input("Average Speed (avg_speed_kmph):", min_value=0.0, value=45.0, key="clf_speed")
                clf_loss = st.number_input("Packet Loss Ratio (packet_loss_pct):", min_value=0.0, max_value=100.0, value=2.0, key="clf_loss")
                clf_visibility = st.number_input("Visibility Distance (visibility_km):", min_value=0.0, value=10.0, key="clf_visibility")
            clf_submit = st.form_submit_button("Classify")

        if clf_submit:
            feature_names = ['flow_veh_per_hr', 'speed_density_ratio', 'avg_comm_delay_ms', 'channel_busy_ratio_pct', 'incident_num', 'wireless_congestion_intensity', 'density_veh_per_km', 'avg_speed_kmph', 'packet_loss_pct', 'visibility_km']
            clf_input = pd.DataFrame([[clf_flow, clf_ratio, clf_delay, clf_cbr, clf_incident_num, clf_wireless, clf_density, clf_speed, clf_loss, clf_visibility]], columns=feature_names)
            predicted_class = classifier_model.predict(clf_input)[0]
            prediction_probs = classifier_model.predict_proba(clf_input)[0]
            st.markdown("---")
            prediction = f"Predicted Traffic State: **{predicted_class}**"
            match predicted_class:
                case 'Free-flow': 
                    st.success(prediction)
                case 'Moderate':
                    st.info(prediction)
                case 'Heavy':
                    st.warning(prediction)
                case _:
                    st.error(prediction)
            st.dataframe(pd.DataFrame({"Traffic Class": classifier_model.classes_, "Probability": [f"{p*100:.2f}%" for p in prediction_probs]}), use_container_width=True)
