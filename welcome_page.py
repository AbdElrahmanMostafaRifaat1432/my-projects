import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image


# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
img_contact_form = Image.open("images/yt_contact_form.png")
image_caption = Image.open("images/image_original (1).png")

# ---- HEADER SECTION ----
with st.container():
    st.subheader("Hi, I am Abd Elrahman Mostafa :wave:")
    st.title("A Machine learning engineer")
    # st.write(
    #     "I am passionate about machine learning."
    # )
    #st.write("[Learn More >](https://pythonandvba.com)")

# ---- WHAT I DO ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("profile")
        st.write("##")
        st.write(
            """
            Aspiring Machine Learning Engineer and a Data Scientist seeking a challenging position. Proficient in machine learning
algorithms, Recommendation systems, Natural Language Processing (NLP), Deep Learning and Computer Vision with
hands-on experience in Data analysis and Visualization. Possesses strong problem-solving abilities, attention to details,
Analytical skills and effective communication skills. Eager to leverage technical expertise and collaborative nature to
contribute effectively to projects and drive innovation in the field of Machine learning and Data analysis.
            """
        )
        #st.write("[YouTube Channel >](https://youtube.com/c/CodingIsFun)")
    with right_column:
        st_lottie(lottie_coding, height=300)

# ---- PROJECTS ----
with st.container():
    st.write("---")
    st.header("My Projects")
    st.write("##")
    image_column, text_column = st.columns((1, 1))
    with image_column:
        st.image(image_caption)
    with text_column:
        #st.markdown('<h3><a href="http://localhost:8501/Image_Captioning" target="_blank">Image Captioning</a></h3>', unsafe_allow_html=True)
        st.subheader("Image captioning")
        #st.markdown(f'[Image Captioning](./pages/Image_Captioning.py)')
        
        st.write(
            """
            using pretrained blip model for captioning from hugging face 
            """
        )
        #st.markdown("[Watch Video...](https://youtu.be/TXSOitGoINE)")
# with st.container():
#     image_column, text_column = st.columns((1, 2))
#     with image_column:
#         st.image(img_contact_form)
#     with text_column:
#         st.subheader("How To Add A Contact Form To Your Streamlit App")
#         st.write(
#             """
#             Want to add a contact form to your Streamlit website?
#             In this video, I'm going to show you how to implement a contact form in your Streamlit app using the free service ‘Form Submit’.
#             """
#         )
#         st.markdown("[Watch Video...](https://youtu.be/FOULV9Xij_8)")

# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("wanna Get In Touch With Me!")
    st.write("##")

    st.write(
            """
            Email : abdelrahmanriffat4568@gmail.com \n
            mobile: 01151025816
            """
        )

