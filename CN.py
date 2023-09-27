
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
	image=ImageOps.fit(image_D,size,Image.LANCZOS)
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
	chemical_control={"Bemisia Argentifolii[Silverleaf whitefly]":"Natural oils are another important tool in the control of B. tabaci. Currently, the most effective oil in the market is the ultra-fine oil, which is a paraffinic oil product that reduces the settlement of the adult flies, decreases oviposition, and abates the transmission of the tomato yellow leaf curl virus."
				      ,"Helicoverpa Armigera":"In small plots, Plantwise suggests handpicking and destroying eggs and young caterpillars is possible.CABI and Plantwise partners recommend introducing light and pheromone traps to trap adult mothsPlantwise and partners have suggested the release of natural enemies, including the parasitoid Trichogramma brassilences or T. pretiosum as methods of control"
					  ,'Myzus Persicae':"It is commonly believed that cypermethrin, abamectin, chlorpyrifos, methylamine and imidacloprid could be the first chemical agents for aphid control in the field. Although imidacloprid is a good insecticide for the control of pests who have piercing-sucking mouthparts, frequent reuse may lead to the severe resistance of pests."
					  ,'Spodoptera Exigua':"Neem Oil, Cottonseed Oil, Horticultural Oil"
					  ,'Spodoptera litura':"Rice bran 5 Kg + Molasses or Brown sugar 500g + Carbaryl 50 WP 500g+ 3lit of water/ha Mix the ingredients well – Kept around the field in the evening hours Spray chlorpyriphos 20 EC 2 lit/ha or dichlorovos 76 WSC 1 lit/ha"
					  ,'Thrips Palmi':"Foliar insecticides are frequently applied for thrips suppression, but at times it has been difficult to attain effective suppression. Various foliar and drench treatments, alone or combined with oil, have achieved some success (Seal and Baranowski 1992, Seal et al. 1993, Seal 1994) though it is usually inadvisable to apply insecticides if predators are present, especially pyrethroids. The eggs, which occur in the foliar tissue, and the pupae, which reside in the soil, are relatively insensitive to insecticide application."
					  ,'Tetranychus Urticae':"Field experiments showed that spider mites can be controlled efficiently with Omite (propargite) 1 ml l-1 water, morocide (binapacryl) 2 g l-1 and plictran (cyhexatin) 3 g l-1 sprayed 5–9 times at weekly intervals. Yields were increased by 30–80%. Cymbush (cypermethrin) electrodyne gave a good control on young plants out did not increase yield, probably due to phytotoxicity. Karathane (dinocap), dimethoate, azodrin (monocrotophos) and phosdrin (mevinphos) were inferior."
					  ,'Zeugdacus Cucurbitae':"Melon fly | Business Queensland using pheromone-based insecticide baits (blocks) to attract and kill male fruit flies. using protein-based insecticide bait (spray) to kill adult female flies. movement restrictions to prevent pest introduction and spread by movement of infested fruit and vegetables"}

	my_bar = st.progress(0)

	for percent_complete in range(100):
		time.sleep(0.1)
		my_bar.progress(percent_complete + 1)
	st.success(predictions)
	st.info(f"Chemical Control:\n{chemical_control[image_class]}")



