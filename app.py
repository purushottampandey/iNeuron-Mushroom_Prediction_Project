##Imporrting all the library!
import streamlit as st
import numpy as np
import pandas as pd
import pickle


def main():
    st.markdown("<h1 style='text-align: center; color:red;'>Mushroom Prediction</h1>", unsafe_allow_html=True)
    Problem_Statement=st.container()
    Prediction=st.container()
    with Problem_Statement:
        st.title("Problem Statement:")
        st.write("""The Audubon Society Field Guide to North American Mushrooms contains descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the
        Agaricus and Lepiota Family Mushroom (1981). Each species is labelled as either definitely edible, definitely poisonous, or maybe edible but not recommended. This last
        category was merged with the toxic category. The Guide asserts unequivocally that there is no simple rule for judging a mushroom's edibility, such as "leaflets three, leave it
        be" for Poisonous Oak and Ivy.
        The main goal is to predict which mushroom is poisonous & which is edible.""")
    with Prediction:                             
        st.title("Let's predict your mushroom!")
        st.write("Enter the details to predict the class of mushroom.")

    # LABELLING THE DATA IN DICTIONARY FORMAT.

    cap_shape={'Bell':0,'Conical':1,'Convex':5,'Flat':2, 'Knobbed':3,'Sunken':4}
    cap_surface={'Fibrous':0,'Grooves':1,'Scaly':3,'Smooth':2}
    cap_color={'Brown':4,'Buff':0,'Cinnamon':1,'Gray':3,'Green':6,'Pink':5,'Purple':7,'Red':2,'White':8,'Yellow':9}
    bruises={'Bruises':1,'No':0}
    odor= {'Almond':0,'Anise':3,'Creosote':1,'Fishy':8,'Foul':2,'Musty':4,'None':5,'Pungent':6,'Spicy':7}
    gill_attachment={'Attached':0,'Free':1}
    gill_spacing={'Close':0,'Crowded':1}
    gill_size={'Broad':0,'Narrow':1}
    gill_color={'Black':4,'Brown':5,'Buff':0,'Chocolate':3,'Gray':2,'Green':8,'Orange':6,'Pink':7,'Purple':9,'Red':1,'White':10,'Yellow':11}
    stalk_shape={'Enlarging':0,'Tapering':1}
    stalk_root={'Bulbous':1,'Club':2,'Equal':3,'Rooted':4,'Missing':0}
    stalk_surface_above_ring={'Fibrous':0,'Scaly':3,'Silky':1,'Smooth':2}
    stalk_surface_below_ring={'Fibrous':0,'Scaly':3,'Silky':1,'Smooth':2}
    stalk_color_above_ring={'Brown':4,'Buff':0,'Cinnamon':1,'Gray':3,'Orange':5,'Pink':6,'Red':2,'White':7,'Yellow':8}
    stalk_color_below_ring={'Brown':4,'Buff':0,'Cinnamon':1,'Gray':3,'Orange':5,'Pink':6,'Red':2,'White':7,'Yellow':8}
    veil_type={'Partial':0,'Universal':1}
    veil_color={'Brown':0,'Orange':1,'White':2,'Yellow':3}
    ring_number={'None':0,'One':1,'Two':2}
    ring_type={'Evanescent':0,'Flaring':1,'Large':2,'None':3,'Pendant':4}
    spore_print_color={'Black':2,'Brown':3,'Buff':0,'Chocolate':1,'Green':5,'Orange':4,'Purple':6,'White':7,'Yellow':8}
    population={'Abundant':0,'Clustered':1,'Numerous':2,'Scattered':3,'Several':4,'Solitary':5}
    habitat={'Grasses':1,'Leaves':2,'Meadows':3,'Paths':4,'Urban':5,'Waste':6,'Woods':0}
    #  Display!
    c1,c2,c3,c4=st.columns(4)
    with c1:
        cap__shape = cap_shape[st.selectbox('Cap Shape :',options=sorted(cap_shape))]
        cap__surface = cap_surface[st.selectbox('Cap Surface:',options=sorted(cap_surface))]
        cap__color = cap_color[st.selectbox('Cap Color:',options=sorted(cap_color))]
        bruises_ = bruises[st.selectbox('Bruise:',options=sorted(bruises))]
        odor_ = odor[st.selectbox('Odor:',options=sorted(odor))]
        gill__attachment = gill_attachment[st.selectbox('Grill Attachment:',options=sorted(gill_attachment))]

    with c2:
        gill__spacing = gill_spacing[st.selectbox('Grill Spacing:',options=sorted(gill_spacing))]
        gill__size = gill_size[st.selectbox('Grill Size:',options=sorted(gill_size))]
        gill__color = gill_color[st.selectbox('Grill Color:',options=sorted(gill_color))]
        stalk_shape_ = stalk_shape[st.selectbox('Stalk Shape:',options=sorted(stalk_shape))]
        stalk__root = stalk_root[st.selectbox('Stalk Root:',options=sorted(stalk_root))]
        stalk_surface_above_ring_ = stalk_surface_above_ring[st.selectbox('Stalk Surface Above Ring:',options=sorted(stalk_surface_above_ring))]

    with c3:
        stalk_surface_below_ring_ = stalk_surface_below_ring[st.selectbox('Stalk Surface Below Ring:',options=sorted(stalk_surface_below_ring))]
        stalk_color_above_ring_ = stalk_color_above_ring[st.selectbox('Stalk Color Above Ring:',options=sorted(stalk_color_above_ring))]
        stalk_color_below_ring_ = stalk_color_below_ring[st.selectbox('Stalk Color Below Ring:',options=sorted(stalk_color_below_ring))]
        veil__type = veil_type[st.selectbox('Veil Type:',options=sorted(veil_type))]
        veil__color = veil_color[st.selectbox('Veil Color:',options=sorted(veil_color))]

    with c4:
        ring__number = ring_number[st.selectbox('Ring Number:',options=sorted(ring_number))]
        ring__type = ring_type[st.selectbox('Ring Color:',options=sorted(ring_type))]
        spore_print_color_ = spore_print_color[st.selectbox('Spore Print Color:',options=sorted(spore_print_color))]
        population_ = population[st.selectbox('Population:',options=sorted(population))]
        habitat_ = habitat[st.selectbox('Habitat:',options=sorted(habitat))]

    # Lets use the trained model to predict!

    trained_model=pickle.load(open('trained_model.sav','rb'))
    input_data=[cap__shape,cap__surface,cap__color,bruises_,odor_,gill__attachment,gill__spacing,gill__size,gill__color,stalk_shape_,stalk__root,stalk_surface_above_ring_,stalk_surface_below_ring_,stalk_color_above_ring_,stalk_color_below_ring_,veil__type,veil__color,ring__number,ring__type,spore_print_color_,population_,habitat_]

    input_as_npa=np.asarray(input_data)
    reshaped_data=input_as_npa.reshape(1,-1)

    if st.button('Predict Probability'):
        result=trained_model.predict(reshaped_data)
        if result[0] == 0:
            st.header("It's Eatable!")
        else:
            st.header("It's Poisonous!")





if __name__ == '__main__':
    main()
