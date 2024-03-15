import streamlit as st


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    st.write('Sabag has Brachydactyly type E (https://rarediseases.info.nih.gov/diseases/987/brachydactyly-type-e)')
    audio_file = st.sidebar.file_uploader(label="",
                                          type=[".wav", ".wave", ".flac", ".mp3", ".ogg"])

    st.audio(audio_file , format="audio/wav", start_time=0)
