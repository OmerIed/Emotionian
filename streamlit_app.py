import json

import requests
import streamlit as st
import os


def _max_width_():
    max_width_str = f"max-width: 1200px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


_max_width_()

# logo and header -------------------------------------------------

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    st.image("logo.png", width=350)
    st.header("")

with c32:

    st.title("")
    st.title("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")

    st.write(
        "Made by Y. Sabag and O. Iedovnik - Team Emotionian"
    )

st.text("")
st.markdown(
    f"""
                    The speech to text recognition is done via the [Facebook's Wav2Vec2 model.](https://huggingface.co/facebook/wav2vec2-large-960h)
                    """
)
st.text("")


def record_page():
    """ This is the main page of the app
    Where the user can record a new entry."""
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:

        with st.form(key="my_form"):

            audio_file = st.file_uploader("", type=[".wav"])

            st.info(
                f"""
                            👆 Upload a .wav file. Or try a sample: [Wav sample 01](https://github.com/CharlyWargnier/CSVHub/blob/main/Wave_files_demos/Welcome.wav?raw=true) | [Wav sample 02](https://github.com/CharlyWargnier/CSVHub/blob/main/Wave_files_demos/The_National_Park.wav?raw=true)
                            """
            )

            submit_button = st.form_submit_button(label="Transcribe")
    if audio_file is not None:
        input_path = audio_file.name
        # Get file size from buffer
        # Source: https://stackoverflow.com/a/19079887
        old_file_position = audio_file.tell()
        audio_file.seek(0, os.SEEK_END)
        getsize = audio_file.tell()  # os.path.getsize(path_in)
        audio_file.seek(old_file_position, os.SEEK_SET)
        getsize = round((getsize / 1000000), 1)

        if getsize < 2:  # File less than 2MB
            # To read file as bytes:
            bytes_data = audio_file.getvalue()

            # Load your API key from an environment variable or secret management service
            # TODO: generate api_token and insert it to st secrets
            api_token = st.secrets["api_token"]

            # endregion API key
            headers = {"Authorization": f"Bearer {api_token}"}
            API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"
            def query(data):
                response = requests.request("POST", API_URL, headers=headers, data=data)
                return json.loads(response.content.decode("utf-8"))

            data = query(bytes_data)

            values_view = data.values()
            value_iterator = iter(values_view)
            text_value = next(value_iterator)
            text_value = text_value.lower()

            st.success(text_value)

            API_URL_EMOTION = "https://api-inference.huggingface.co/models/CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-adults"


            def query_emotion(data):
                response = requests.request("POST", API_URL_EMOTION, headers=headers, data=data)
                return json.loads(response.content.decode("utf-8"))

            data_emotion = query_emotion(bytes_data)
            print(data_emotion)
            values_view_emotion = data_emotion.values()
            value_iterator_emotion = iter(values_view_emotion)
            text_value_emotion = next(value_iterator_emotion)
            text_value_emotion = text_value_emotion.lower()

            st.success(text_value_emotion)

            c0, c1 = st.columns([2, 2])

            with c0:
                st.download_button(
                    "Download the transcription",
                    text_value,
                    file_name=None,
                    mime=None,
                    key=None,
                    help=None,
                    on_click=None,
                    args=None,
                    kwargs=None,
                )

        else:
            st.warning(
                "🚨 The file you uploaded is more than 2MB! Please switch to full mode ↖️ and add your HuggingFace API key."
            )
            st.stop()

    else:
        path_in = None
        st.stop()


def transcribe(audio_file):
    pass


def entry_history():
    pass


def emotion_recognition():
    pass


def analysis_of_emotion():
    pass





def main():
    pages = {
        "Record your entry": record_page,
        # "Look at your previous entries": entry_history,
        # "Analysis of your emotions": analysis_of_emotion
    }

    if "page" not in st.session_state:
        st.session_state.update(
            {
                # Default page
                "page": "Home",
            }
        )

    with st.sidebar:
        page = st.radio("Select your mode", tuple(pages.keys()))

    pages[page]()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # st.title('Emotionian')
    # st.write('Sabag has Brachydactyly type E (https://rarediseases.info.nih.gov/diseases/987/brachydactyly-type-e)')
    # audio_file = st.sidebar.file_uploader(label="",
    #                                       type=[".wav", ".wave", ".flac", ".mp3", ".ogg"])
    # st.audio(audio_file, format="audio/wav", start_time=0)
    main()
