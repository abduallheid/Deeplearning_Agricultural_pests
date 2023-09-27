
from PIL import Image,ImageOps
import numpy as np
import streamlit as st 
import time
from tensorflow.keras.models import load_model

st.info("Develpoed by Abdallh Rawak")
# upload the model
MODEL= load_model("pest.h5")
# upload the image which classify
file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

st.title("Detection of agricultural insect pests using the cnn algorithm")
st.text("classify images using the cnn algorithm with 12 different types of insect pests agricultural")

def predict(image_D):
	
	size=(256,256)
	image=ImageOps.fit(image_D,size,Image.ANTIALIAS)
	img=np.asarray(image)
	img_reshape=img[np.newaxis,...]

	prediction = MODEL.predict(img_reshape)
	return prediction

if file is None:
	st.text("pleas upload img ")
else:
	image=Image.open(file)
	st.image(image,use_column_width=True)
	st.subheader("Image")
	result=predict(image)
	CLASS_NAMES=['Bemisia Argentifolii', 'Helicoverpa Armigera', 'Myzus Persicae', 'Spodoptera Exigua', 'Spodoptera litura', 'Thrips Palmi', 'Tetranychus Urticae', 'Zeugdacus Cucurbitae']
	image_class = CLASS_NAMES[np.argmax(result)]
	predictions = f"pest is probably {image_class}, accurcy {np.max(result):22f}"
	my_bar = st.progress(0)

	for percent_complete in range(100):
		time.sleep(0.1)
		my_bar.progress(percent_complete + 1)
	st.success(predictions)



