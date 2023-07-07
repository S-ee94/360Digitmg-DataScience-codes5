import pandas as pd
import streamlit as st 
from statsmodels.regression.linear_model import OLSResults
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


model = OLSResults.load("model.pickle")


def main():
    
    st.title("Forecasting")
    st.sidebar.title("Forecasting")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Forecasting </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file", type = ['csv', 'xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("you need to upload a csv or excel file.")
    
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    
    if st.button("Predict"):
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
        
        
        ###############################################
        st.subheader(":red[Forecast for New data]", anchor = None)
         
        newdata_pred = pd.DataFrame(model.predict(start = data.index[0], end = data.index[-1]))
        results = pd.concat([data,newdata_pred], axis=1)
        results.to_sql('forecast_results_dd', con = engine, if_exists = 'replace', index = False, chunksize = 1000)        
        
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(results.style.background_gradient(cmap = cm).set_precision(2))
        
        
        st.text("")
        st.subheader(":red[plot forecasts against actual outcomes]", anchor = None)
        #plot forecasts against actual outcomes
        fig, ax = plt.subplots()
        ax.plot(data.Sales,'-b', label = 'Actual Value')
        ax.plot(newdata_pred, '-r', label = 'Predicted value')
        ax.legend();
        st.pyplot(fig)
        
        # data.to_sql('forecast_pred', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        # #st.dataframe(result) or
        # #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        # import seaborn as sns
        # cm = sns.light_palette("blue", as_cmap=True)
        # st.table(result.style.background_gradient(cmap=cm).set_precision(2))

                           
if __name__=='__main__':
    main()


