import streamlit as st
from PIL import Image
import numpy as np
from clf import predict
import base64
import plotly.graph_objects as go
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """

# Loading Image using PIL
im = Image.open('logo.png')
# Adding Image to web app
st.set_page_config(page_title="Banclas", page_icon = im, layout = "wide")
st.markdown(hide_menu_style, unsafe_allow_html=True)

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
#add a background image    
add_bg_from_local('bg.png') 

st.set_option('deprecation.showfileUploaderEncoding', False)

head_col_1, head_col_2 = st.columns([0.15,2])
with head_col_1:
    st.image(im, width=80)
with head_col_2:
    st.title("BANCLAS: Cavendish Banana Maturity Classification App")

col_1, col_2 = st.columns([1,1])

def display_image(image):
    # Display the image with CSS styling for height and width
    st.image(image, caption='Uploaded Image',  
             use_column_width=True)
    
    return image

labels = []

def title(test = " "):
    st.markdown("**")

with col_1:
    uploaded_files = st.file_uploader("Upload an image", type=['jpeg', 'jpg', 'png'], 
                            accept_multiple_files=True)
    
    # Display the drop-down menu
    selected_image = st.selectbox("Select an image", uploaded_files, format_func=lambda file: file.name if file else "")
    # Display the selected image
    if selected_image:
        img = Image.open(selected_image)
        display_image(img)

with col_2:
    if uploaded_files:
        labels = predict(np.array(img))
        class_labels = []
        score = []
        colors = ["#718355", "#97A97C", "#CFE1B9", "#FFFBEB"]
        for i in labels:
            class_labels.append(i[0])
            score.append(i[1])
    
    #if uploaded_files:
        st.subheader("Result")
        # print out the prediction labels with scores
        st.write("Actual Prediction:",  labels[0][0])
        st.write("Score: ", str(labels[0][1]))
        st.write("Appearance:", labels[0][2])
        
        # Create the donut chart figure
        fig = go.Figure(data=[go.Pie(labels=class_labels, values=score, hole=0.6)])

        # Set layout options
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0),
                          width = 700,
                          paper_bgcolor='rgba(0,0,0,0)' , # Set paper_bgcolor to no color
                          colorway = colors)

        # Display the donut chart in Streamlit
        st.plotly_chart(fig)
