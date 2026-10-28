
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Player Injury Risk Assessment",
    page_icon="🏃",
    layout="wide"
)

st.title("🏃 Player Injury Risk Assessment")
st.caption(
    "AI-assisted screening dashboard | "
    "Final participation decisions require medical review."
)

# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------
MODEL_PATH = Path("injury_model.pkl")
PROTOCOL_PATH = Path("medical_protocol.txt")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Please place injury_model.pkl "
        "in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()

# --------------------------------------------------
# 3. LOAD MEDICAL PROTOCOL
# --------------------------------------------------
def load_protocol():
    if PROTOCOL_PATH.exists():
        return PROTOCOL_PATH.read_text(encoding="utf-8")

    return (
        "Medical protocol file not found. "
        "Consult the team doctor for assessment and guidance."
    )

protocol_text = load_protocol()

# --------------------------------------------------
# 4. PLAYER INPUTS
# --------------------------------------------------
st.subheader("Player Information")

col1, col2 = st.columns(2)

with col1:
    sleep_hours = st.slider(
        "Sleep Hours",
        min_value=4.0,
        max_value=10.0,
        value=8.0,
        step=0.5
    )

    fatigue_score = st.slider(
        "Fatigue Score",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

with col2:
    workload_ratio = st.slider(
        "Acute Workload Ratio",
        min_value=0.8,
        max_value=2.0,
        value=1.0,
        step=0.05
    )

    past_injury = st.slider(
        "Past Injury",
        min_value=0,
        max_value=1,
        value=0,
        step=1,
        help="0 = No previous injury, 1 = Previous injury"
    )

st.caption(
    "Acute Workload Ratio is entered as a numerical ratio. "
    "Use the same definition as in the training dataset."
)

# --------------------------------------------------
# 5. PREDICT INJURY RISK
# --------------------------------------------------
if st.button("Assess Injury Risk", type="primary"):

    # Keep feature names and order consistent with training
    input_data = pd.DataFrame(
        [[
            sleep_hours,
            fatigue_score,
            workload_ratio,
            past_injury
        ]],
        columns=[
            "Sleep_Hours",
            "Fatigue_Score",
            "Acute_Workload_Ratio",
            "Past_Injury"
        ]
    )

    try:
        prediction = int(model.predict(input_data)[0])

        st.session_state["prediction"] = prediction
        st.session_state["input_data"] = input_data
        st.session_state["assessed"] = True
        st.session_state["doctor_approved"] = False

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --------------------------------------------------
# 6. DISPLAY PREDICTION AND RAG BRIEFING
# --------------------------------------------------
if st.session_state.get("assessed", False):

    prediction = st.session_state["prediction"]
    input_data = st.session_state["input_data"]

    st.divider()
    st.subheader("Assessment Results")

    if prediction == 1:
        st.error("HIGH RISK — Medical review recommended")
        st.write(
            "The model has flagged this player's inputs as "
            "high risk. Arrange assessment by the team doctor "
            "before making participation decisions."
        )
    else:
        st.success("LOW RISK — No high-risk flag from the model")
        st.write(
            "The model has not flagged this input as high risk. "
            "This does not rule out an injury or establish fitness to play."
        )

    # Display the input summary
    st.subheader("Player Input Summary")
    st.dataframe(input_data, use_container_width=True, hide_index=True)

    # --------------------------------------------------
    # 7. RAG-STYLE MEDICAL PROTOCOL BRIEFING
    # --------------------------------------------------
    st.subheader("Medical Protocol Briefing")

    # Simple local retrieval:
    # Split the protocol into paragraphs and retrieve
    # sections containing relevant keywords.
    paragraphs = [
        p.strip()
        for p in protocol_text.split("\n\n")
        if p.strip()
    ]

    keywords = [
        "injury",
        "fatigue",
        "workload",
        "sleep",
        "doctor",
        "medical",
        "clearance",
        "bench",
        "pain",
        "high risk",
        "low risk"
    ]

    relevant_sections = [
        paragraph
        for paragraph in paragraphs
        if any(
            keyword in paragraph.lower()
            for keyword in keywords
        )
    ]

    if relevant_sections:
        st.info(
            "The following protocol sections were retrieved "
            "from medical_protocol.txt. Review them with the "
            "qualified medical professional."
        )

        for index, section in enumerate(
            relevant_sections, start=1
        ):
            with st.expander(f"Protocol Reference {index}"):
                st.write(section)
    else:
        st.warning(
            "No matching protocol sections were found. "
            "Refer to the complete protocol and consult the doctor."
        )

    # --------------------------------------------------
    # 8. DOCTOR APPROVAL GATE
    # --------------------------------------------------
    st.divider()
    st.subheader("Doctor Review and Participation Decision")

    st.warning(
        "Do not clear or bench a player solely on the basis "
        "of an AI prediction. A qualified clinician must assess "
        "the player's symptoms, examination findings, and "
        "applicable return-to-play protocol."
    )

    doctor_approved = st.checkbox(
        "Doctor Approval: I confirm that a qualified doctor "
        "has reviewed this player and authorized the decision.",
        key="doctor_approval_checkbox"
    )

    decision_options = [
        "Pending medical decision",
        "Cleared by doctor",
        "Benched by doctor"
    ]

    decision = st.selectbox(
        "Doctor's participation decision",
        decision_options,
        key="doctor_decision"
    )

    if st.button("Confirm Medical Decision"):

        if not doctor_approved:
            st.error(
                "Confirmation blocked. Doctor approval is required."
            )

        elif decision == "Pending medical decision":
            st.warning(
                "Please select a final decision made by the doctor."
            )

        else:
            st.session_state["doctor_approved"] = True
            st.session_state["final_decision"] = decision

            if decision == "Cleared by doctor":
                st.success(
                    "Doctor-reviewed decision recorded: "
                    "Cleared by doctor."
                )
            else:
                st.error(
                    "Doctor-reviewed decision recorded: "
                    "Benched by doctor."
                )

            st.caption(
                "This demo displays the decision in the current "
                "session only; it does not create a permanent audit log."
            )

# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Disclaimer: This application is a demonstration tool. "
    "Model predictions and retrieved text do not replace "
    "clinical assessment, emergency care, or approved team protocols."
)
