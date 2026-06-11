import streamlit as st
import tempfile
import time
from pathlib import Path

from main import process_video as dumbbell_process
from plank_main import process_video as plank_process

st.set_page_config(
    page_title="AI Gym Form Checker",
    page_icon="🏋️",
    layout="wide"
)

st.markdown("""
<style>
.block-container{
    padding-top:1rem;
}

.stButton button{
    width:100%;
    height:60px;
    font-size:20px;
    font-weight:bold;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏋️ AI Gym Form Checker")
st.caption("YOLOv8 Pose Estimation • Rep Counter • Form Analysis")

left, right = st.columns([2, 1])

with left:
    uploaded_file = st.file_uploader(
        "Upload Workout Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

with right:

    mode = st.selectbox(
        "Select Analysis Type",
        [
            "DUMBBELL",
            "PLANK"
        ]
    )

    if mode == "DUMBBELL":

        exercise = st.selectbox(
            "Select Exercise",
            [
                "BICEP CURL",
                "HAMMER CURL",
                "SHOULDER PRESS"
            ]
        )

if uploaded_file:

    st.subheader("Video Preview")
    st.video(uploaded_file)

    if st.button("🚀 Analyze Workout"):

        progress_text = st.empty()
        progress_bar = st.progress(0)

        progress_text.info(
            "Preparing video..."
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as tmp:

            tmp.write(
                uploaded_file.read()
            )

            video_path = tmp.name

        progress_bar.progress(10)

        progress_text.info(
            "Loading AI model..."
        )

        time.sleep(1)

        progress_bar.progress(25)

        progress_text.info(
            "Analyzing video..."
        )

        start = time.time()

        try:

            if mode == "DUMBBELL":

                dumbbell_process(
                    video_path,
                    exercise
                )

                output_file = "output_dumbbell_ai.mp4"

                screenshots_dir = Path(
                    "screenshots_dumbbell"
                )

            else:

                plank_process(
                    video_path
                )

                output_file = "output_plank_ai.mp4"

                screenshots_dir = Path(
                    "screenshots_plank"
                )

            progress_bar.progress(90)

            progress_text.info(
                "Generating results..."
            )

            time.sleep(1)

            progress_bar.progress(100)

            elapsed = round(
                time.time() - start,
                2
            )

            st.success(
                f"Analysis completed in {elapsed} seconds"
            )

            # ==========================
            # PROCESSED VIDEO SECTION
            # ==========================

            if Path(output_file).exists():

                st.subheader(
                    "Processed Video"
                )

                with open(
                    output_file,
                    "rb"
                ) as f:

                    video_bytes = f.read()

                st.video(
                    video_bytes
                )

                st.download_button(
                    "⬇ Download Result",
                    data=video_bytes,
                    file_name=Path(output_file).name,
                    mime="video/mp4"
                )

            # ==========================
            # SCREENSHOTS SECTION
            # ==========================

            if screenshots_dir.exists():

                images = sorted(
                    screenshots_dir.glob("*.jpg"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )

                if images:

                    st.subheader(
                        "Captured Screenshots"
                    )

                    cols = st.columns(3)

                    for i, img in enumerate(images):

                        cols[i % 3].image(
                            str(img),
                            width="stretch"
                        )

        except Exception as e:

            st.error(
                f"Processing failed:\n{str(e)}"
            )