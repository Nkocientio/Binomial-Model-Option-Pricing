from binomial_class import *

def interval_dates(now,expire,freq,yrs):
    freq_dict ={
        'Monthly' : 1,
        'Quarterly' :3,
        'Half-Yearly': 6,
        'Anually': 12
    }
    
    start_date = now + timedelta(days=365*freq_dict[freq]/12)
    end_date = start_date + timedelta(days = yrs*365) 
    
    dates = np.arange(start_date, end_date, dtype=f'datetime64[{freq_dict[freq]}M]')
    dates = np.unique( np.append(dates, [expire]).astype('datetime64[D]') )
    return dates

def get_dates(freqency,yrs):
    st.write(f'{freqency} exercise Dates ')
    today = datetime.now()
    expire_date = today + timedelta(days=365*yrs)
    if 'b_dates' not in st.session_state:
        st.session_state.b_dates = np.array([expire_date], dtype='datetime64[D]')
    if freqency == 'Manual':
        with st.form(key='bermuda_dates'):
            dt = st.date_input('Enter the exercise date', value =today, min_value=today, max_value = expire_date)
            
            col1, col2 = st.columns(2) 
            with col1:
                submit_button = st.form_submit_button()
            with col2:
                clear_button = st.form_submit_button(label='Clear Date')
                
            if submit_button and dt not in st.session_state.b_dates:
                st.session_state.b_dates = np.hstack((st.session_state.b_dates, dt))  # Append new column
                
            if clear_button:
                st.session_state.b_dates = st.session_state.b_dates[:-1]
    else:
        st.session_state.b_dates = interval_dates(today,expire_date,freqency,yrs)
        
    # Display the dates Horizontally
    st.write(st.session_state.b_dates[:, np.newaxis].T)


def get_divs(expire):
    st.write('Cash Dividends')
    divs = np.zeros((2, 0))
    if 'divs' not in st.session_state:
        st.session_state.divs = divs
        
    with st.form(key='cash_divs'):
        year = st.number_input('Enter years from today of a div date', min_value=0.0, max_value =expire, format="%.2f")
        value = st.number_input('Div value', min_value=0.0, format="%.2f")
        
        col1, col2 = st.columns(2) 
        with col1:
            submit_button = st.form_submit_button()
        with col2:
            clear_button = st.form_submit_button(label='Clear Div')
            
        if submit_button:
            new_entry = np.array([[year], [value]])
            st.session_state.divs = np.hstack((st.session_state.divs, new_entry))  # Append new column
            
        if clear_button:
            st.session_state.divs = st.session_state.divs[:,:-1]

    st.write(st.session_state.divs)

def main():
    st.title("Binomial Option Pricing")
    st.markdown(f"##### (American, Asian, Bermudan, Compound and European) style options")
    st.sidebar.title("Options Pricer")
    st.sidebar.markdown("**Made by:**")
    st.sidebar.write(random.choice(['Acentio','Nkocie', 'Nkosenhle','Nkosembi','Nkosendala','Nkosentsha', 'Khathazile'])) 
    st.sidebar.header("Model Inputs")
    
    style = st.sidebar.selectbox("Style", ['American', 'Asian', 'Bermudan', 'Compound', 'European'])
    option_type = st.sidebar.selectbox("Option Type", ['Call', 'Put'])
    N = st.sidebar.number_input("Number of Time Steps (N)", min_value=1, max_value=6000, value=10) # max_value to be increases...
    
    avg_what, avg_method = 'Asset', 'Geometric'
    if style == 'Asian':
        st.write("Asian options look at the average of (asset or strike) overtime.")
        st.markdown(f"About {style} options [watch this](https://youtu.be/rsyBxMtnn9A?si=C5_xG9Z6R6hNFo2y)")
        avg_what = st.sidebar.selectbox("Average What", ['Asset', 'Strike'])
        avg_method = st.sidebar.selectbox("Average Method", ['Arithmetic', 'Geometric'])
        if avg_what == 'Strike':
            st.write('Strikes here are randomized.')
        elif avg_method == 'Arithmetic' and N > 1000:
            st.write('Max_Value of time steps is set to 1000')
            N = 1000
            
    elif style == 'Compound':
        st.write("Compound option types (Call on Call), (Call on Put), (Put on Call) and (Put on Put).")
        st.markdown(f"About {style} options [watch this](https://youtu.be/CC9JWooTGrQ?si=6mnoGL6am7MUvq9e)")
        compound_optTyp = st.sidebar.selectbox(f'{option_type} on', ['Call', 'Put'])
        compound_K1 = st.sidebar.number_input('Enter strike K1', min_value=0.0,value =0.0) # max should be K
        compound_n = st.sidebar.number_input('Enter exercise step of T1', min_value=1, max_value = N) 
    
    if style in ['Asian', 'Compound']:
        double_style = st.sidebar.selectbox("Exercise Style", ['American', 'Bermudan', 'European'])
        st.write(f"The exercise style of {style} options is like American, European or Bermudan.")
        file_name = f"{style}_{double_style}_{option_type} {N} steps.xlsx"
    else:
        double_style = style
        file_name = f"{style}_{option_type} {N} steps.xlsx"
    st.markdown(f"About {double_style} options [read here](https://corporatefinanceinstitute.com/resources/derivatives/american-vs-european-vs-bermudan-options/)")
    
    S = st.sidebar.number_input("Spot Price (S)", min_value=0.001, value=100.0)
    K = st.sidebar.number_input("Strike Price (K)", min_value=0.01*S, value=99.0)
    T = st.sidebar.number_input("Time to Maturity (T)", min_value=0.0001, value=1.0)
    sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.001,max_value=1.0, value=0.2)
    r = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.06)

    bermudan_dates = np.array([datetime.now()], dtype='datetime64[D]' )
    if style == 'Bermudan' or double_style == 'Bermudan':
        exercised_ = st.sidebar.selectbox('Enter exercise_dates', ['Half-Yearly','Monthly','Quarterly','Anually','Manual'])
        get_dates(exercised_,T)
        bermudan_dates = st.session_state.b_dates
    
    div_type = st.sidebar.selectbox("Dividends_type", ['Yield','Cash'])
    full_divs = np.zeros((2,1))
    if div_type == 'Cash':
        q=0.0
        get_divs(T)
        full_divs = full_divs if st.session_state.divs.size == 0 else st.session_state.divs
    else:
        q = st.sidebar.number_input("Dividend Yield (q)", min_value=0.0,max_value = r, value=0.02)

    st.sidebar.write("")
    
    if N < 1001:
        download = st.sidebar.selectbox("Download the tree ?", ['No','Yes'])
    else:
        download = 'No'
        
    if st.button(f"Evaluate {file_name[:-5]}"):
        option = Binomial_Model(S,K,T,r,q,sigma,style,double_style,option_type,N,full_divs,avg_what,avg_method)
        option.exercise_dates = bermudan_dates
        option.div_type = div_type
        
        with st.spinner('Calculating...'):
            if style == 'Compound':
                option.compound_option(compound_optTyp,compound_K1,compound_n)
            else:
                option.build_tree()

            st.markdown(f"#### Value of {style} {option_type}: {option.price:.6f}")
            if style == 'European':
                analytical_price = option.black_scholes()
                st.markdown(f"#### Black_Scholes gives: {analytical_price:.6f}")
            
        if download == 'Yes':
            with st.spinner('⌛ Saving...'):
                workbook_bytes = option.excel_values()
                st.download_button(
                    label="Save in Excel",
                    data=workbook_bytes,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
            
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.write("Oops sorry my friend")
