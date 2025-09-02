import streamlit as st
from langchain.chains import LLMChain
from langchain import PromptTemplate
import os
from langchain.callbacks import StreamlitCallbackHandler
# from .templates import Templates
# from llm_models import LLM_ChatOpenAI
# from .gmail import send_mail
from app_utilities import * 
from dotenv import load_dotenv
from PIL import Image
import random
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
from openai import OpenAI  # Updated import

try:
    api_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found in secrets or environment variables")

os.environ['OPENAI_API_KEY']=api_key
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def customer_profile(datapath="./customers.csv"):
    st.set_page_config(layout='wide')

    if 'full_response' not in st.session_state:
        st.session_state.full_response = ''

    if 'message_placeholder' not in st.session_state:
        st.session_state.message_placeholder = st.empty()

    # llm config ------------------------------------
    # llm = LLM_ChatOpenAI(streaming=True)
    # template = Templates.customer_profile

    # prompt = PromptTemplate(template=template, input_variables=["customer_name"])
    
    # chain = LLMChain(llm=llm, prompt=prompt)

    # customre data loading -------------------------
    @st.cache_data
    def get_data():
        df = pd.read_csv(datapath)
        return df
    
    st.cache_data.clear()
    df = get_data()

    customer_list = df.loc[:,'id'].unique()
    
    # Initialize selected customer in session state if not already present
    if 'selected_customer' not in st.session_state:
        st.session_state.selected_customer = customer_list[0] if len(customer_list) > 0 else None

    def customer_selection():
        customer = df.loc[df.id==selected,:]
        return customer

    def product_changed():
        # Reset the product stream count
        st.session_state.count=0
        # Also reset the messages
        st.session_state.messages=[]
        # Also reset the prompt history
        st.session_state.userPrompts=[]
    
    # Resets the selected product & email text when a new user is selected from the user dropdown
    def customer_changed():
        if 'product' not in st.session_state:
            st.session_state.product = None
        st.session_state.product = None
        if 'email' not in st.session_state:
            st.session_state.email = ''
        # Don't try to generate template here since customer_name isn't defined yet
        # Just reset the email to empty
        st.session_state.email = ''
        # Reset all the stored prompts / replies
        product_changed()
    
    # Helper functions

    def filter_prompt(prompt, sg_ic_format = r'S[0-9]{2}\w{6,}'
                    , my_ic_format = r'[0-9]{12}|\d{6}-\d{2}-\d{4}'
                    , card_format = r"\d{4}[\s-]{,1}\d{4}[\s-]{,1}\d{4}[\s-]{,1}\d{4}"):
        """
        Filters sensitive info out of a prompt and replaces it with a place holder
        """
        prompt = re.sub(sg_ic_format, "<IC Number>", prompt, flags=re.I)
        prompt = re.sub(my_ic_format, "<IC Number>", prompt, flags=re.I)
        prompt = re.sub(card_format, "<Card Number>", prompt, flags=re.I)
        return prompt

    def get_draft(prompt, product):
        """
            Generates the email draft template. If stream_count==0 (ie this is the start of a new product draft stream) then simply generate the template email. 

            Otherwise will send the request to the openai API and update the draft
        """
        if st.session_state.count == 0:
            email = generate_template(customer_name=customer_name, product_selected=st.session_state.product)
            return email
        else:
            # Send the API query based on what the recorded prompt was
            question, messages = construct_request(
                chat_history = st.session_state.messages
                , email_text = st.session_state.email
                , prompt = st.session_state.prompt
                , customer_name=customer_name
                , full_chain=True)
            # update session states
            st.session_state.userPrompts.append(prompt)
            st.session_state.messages.append(question)
            # st.session_state.messages.append(response)
            return (question, messages)

    def submit_clear():
            """
            This function handles the clearing of the text_input box once a user has entered their prompt. Also calls the helper method to query the API
            """
            st.session_state.prompt = filter_prompt(st.session_state.text)
            if st.session_state.prompt:
                # Increment this stream's count
                st.session_state.count += 1 
            st.session_state.text = None
    

    # sider bar ---------------------------------------
    with st.sidebar:
        st.markdown(' ')
        st.markdown(' ')
        with st.expander('Select Your Customer:', True):
            selected = st.selectbox('Select one here:',
                                   customer_list, 
                                   index=list(customer_list).index(st.session_state.selected_customer) if st.session_state.selected_customer in customer_list else 0,
                                   key='selected_customer',
                                   on_change=customer_changed)
    
    # Process customer data OUTSIDE of sidebar context so it's accessible everywhere
    selected = st.session_state.selected_customer
    customer = df.loc[df.id==selected,:]
    # This is the currently selected customer's name. Assumes id values are unique
    customer_name_list = customer[['first_name', 'last_name']].values.tolist()[0]
    initials = "".join(list(map(lambda x : x[0], customer_name_list)))
    customer_name = ' '.join(customer[['first_name', 'last_name']].values.tolist()[0])


    c1, c1_, c2, c2_, c3 = st.columns((5,0.2, 10, 0.2, 5))

    # Customer Info ------------------------------
    with c1:
        c10, c11, c12 = st.columns([0.2,3,4])
        with c11:
            with st.container():
                circle_size=95
                font_size=40
                initials = initials
                svg_code = f"""
                            <svg height="{circle_size}" width="{circle_size}">
                                <circle cx="{circle_size / 2}" cy="{circle_size / 2}" r="{circle_size / 2}" fill="#333333" />
                                <text x="50%" y="50%" alignment-baseline="middle" text-anchor="middle" font-size="{font_size}" fill="orange">{initials}</text>
                            </svg>
                        """
                st.markdown(svg_code, unsafe_allow_html=True)
            with c12:
                st.markdown(' ')
                markdown_text = f"<span style='font-size:{20}px;'>:orange[**{customer_name}**] </span><br><span style='font-size:{16}px;'>Kuala Lumpur, Malaysia</span>"
                st.write(markdown_text, unsafe_allow_html=True)                
        st.markdown("")
        # customer info ------------------------------------------
        with st.expander(":blue[**CUSTOMER INFO**]", True):
            st.write(f"🪪 Customer ID: **{customer.id.values[0]}**")
            st.write(f'📆 Age: **{customer.age.values[0]}**')
            st.write(f'🧰 Occupation: **{customer.occupation.values[0]}**')
            st.write(f'🧑‍🏫 Education: **{customer.education.values[0]}**')
            st.write(f'📧 Email Address: **{customer.email_address.values[0]}**')
            st.write(f'📱 Mobile Number: **{customer.mobile_number.values[0]}**')
            # st.write('📍 Customer Address: **S1234567Y**')        
            
        # credit profile ------------------------------------------
        with st.expander(":blue[**CREDIT PROFILE**]", True):
            st.write(f'💼 Customer Segment: **{customer.segment.values[0]}**')
            st.write(f'📝 Credit Bureau: **{customer.credit_bureau.values[0]}**')
            st.write(f'🔎 Credit Score: **{customer.credit_score.values[0]}**')

        # with st.expander(":blue[**PERSONA**]", True):
        #     st.write(f'Young postgraduate banker Ali navigates early career in finance. Tech-savvy and committed to continuous learning, they balance professional growth with an active social life.')        
        
        
        # customer value ------------------------------------------
        with st.expander(":blue[**CUSTOMER VALUE**]", True):
            st.markdown(f'💰 Lifetime Value: **RM{customer.lifetime_value.values[0]:,.0f}**')
            st.markdown(f'☺ Churn Likelihood: **Medium**')
            # st.slider('', None, None, 25)

        # NBO ------------------------------------------
        with st.expander(":blue[**NEXT BEST OFFER**]", True):
            insurance_value, car_loan_value, mortgage_value = customer[['Travel Insurance', 'Car Loan', 'Mortgage']].values.tolist()[0]
            st.markdown(f'🔎 Propensity: **{customer.propensity.values[0]}**', unsafe_allow_html=True)
            st.write(f'✈️ Travel Insurance: :green[**{insurance_value}**]')
            st.write(f'🚗 Auto Loan: **{car_loan_value}**')
            st.write(f'🏠 Mortgage: **{mortgage_value}**')

    # Customer Insights
    with c2:
        # tab1, tab2 = st.tabs(["Cat", "Dog", "Owl"])
        
        c2_heading = f"<center><h3 style='color: orange;'>{'CUSTOMER INSIGHTS'}</h3></center>"    
        st.markdown(c2_heading, unsafe_allow_html=True)
        st.markdown(' ')
        st.markdown(' ')

        casa_holdings = ["Saving Account", "Fixed Deposit", "Current Account"]
        banking_holdings= ["Credit Cards", "Personal Loan", "Mortgage"]
        investment_holdings = ["Unit Trust", "Funds", "FX"]
        insurance_holdings = ["Life", "Mortgage", "Term"]

        with st.expander(":blue[**Product Holdings**]", True):
            # holding1, holding2, holding3, holding4 = st.columns([4, 4, 4 , 4])
            holdings_dict = {
                "CASA": casa_holdings
                , 'Banking' : banking_holdings
                , 'Investment': investment_holdings
                , 'Insurance' : insurance_holdings
            }
            holding_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in holdings_dict.items() ]))
            holding_df = holding_df.fillna("")
            st.data_editor(holding_df, hide_index=True)
            
        # asset----------------------------------------------
        with st.expander(":blue[**ASSETS**]", True):
            asset, expense = st.columns([4,4])

            with asset:
                st.write('<span style="font-size: 14px;">Total Income</span>', unsafe_allow_html=True)
                months=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                fig = go.Figure([go.Bar(x=months, y=[20, 14, 23, 24, 25, 26], marker_color='#809EC2')])
                fig.update_layout(yaxis=dict(range=[0, 30], visible=False),
                                    showlegend=False,
                                    width=200, height=200, 
                                    margin=dict(l=0, r=0, t=0, b=0),
                                    colorway=px.colors.sequential.Magma,
                                    bargap=0.4,
                                    xaxis=dict(linecolor='#808080'),)
                st.plotly_chart(fig, use_container_width=False)
            with expense:
                st.write('<span style="font-size: 14px;">Total Expense</span>', unsafe_allow_html=True)
                months=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                fig = go.Figure([go.Bar(x=months, y=[19, 10, 20, 25, 20, 28], marker_color='#D092A7')])
                fig.update_layout(yaxis=dict(range=[0, 30], visible=False),
                                    showlegend=False,
                                    width=200, height=200, 
                                    margin=dict(l=0, r=0, t=0, b=0),
                                    colorway=px.colors.sequential.Magma,
                                    bargap=0.4,
                                    xaxis=dict(linecolor='#808080'),)
                st.plotly_chart(fig, use_container_width=False)
                
        
        # cards----------------------------------------------
        with st.expander(":blue[**CARDS**]", True):

            card1, card2= st.columns(2)
            with card1:
                labels = ['Travel','Dining','Shopping','Others']
                values = [4500, 2500, 1053, 500]
                st.write('<span style="font-size: 14px;">Spending in Last 6M</span>', unsafe_allow_html=True)
                # Use `hole` to create a donut-like pie chart
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
                fig.update_layout(showlegend=False)
                fig.update_layout(width=200, height=200, margin=dict(l=0, r=0, t=0, b=20))
                fig.update_layout(colorway=px.colors.sequential.Magma)
                st.plotly_chart(fig, use_container_width=False)
        
            with card2:
                items = ["✈️ Flight booking", "🍴 Dining", "🛍️ Online shopping", "💊 Medicine"]
                amount = ["3,000", "2,000", "1,000", "5,000"]
                tdf = pd.DataFrame({
                    "Recent Transactions (1M)" : items
                    , "MYR" : amount
                })
                st.markdown("")
                st.markdown("")
                st.markdown("")
                st.dataframe(tdf, hide_index=True)


        
        # liability----------------------------------------------
        with st.expander(":blue[**LOANS**]", True):
            # loan1, loan2, loan3 = st.columns([3,1,4])
            
            st.write('<span style="font-size: 14px;">📆 Tenure: 20 years; 15 remaining</span><br><span style="font-size: 14px;">🏠 Type: Semi-detached</span><br><span style="font-size: 14px;">📍 Address: 10 Lrg Jaya, Kuala Lumpur</span>', unsafe_allow_html=True)
            st.slider("Mortgage (RM'000)", 0, 1300, 800, disabled=False)
            st.markdown(' ')
            st.write('<span style="font-size: 14px;">📆 Tenure: 5 years; 1 remaining</span><br><span style="font-size: 14px;">🎯 Purpose: Study</span>', unsafe_allow_html=True)
            st.slider("Peronal Loan (RM'000)", 0, 100, 90, disabled=False)

        
        
        # insurance----------------------------------------------
        with st.expander(":blue[**INSURANCE**]", True):    
            # insurance1, insurance2, insurance3 = st.columns([3,1,4])
        
            st.write('<span style="font-size: 14px;">📆 Tenure: 20 years; 15 remaining</span>', unsafe_allow_html=True)
            st.slider("Life (RM'000)", 0, 1000, 500, disabled=False)
            st.markdown(' ')
            st.write('<span style="font-size: 14px;">📆 Tenure: 15 years; 5 remaining</span>', unsafe_allow_html=True)
            st.slider("Mortgage (RM'000)", 0, 100, 20, disabled=False)

    #########################
    # GENERATIVE AI SECTION #
    #########################

    # Record message states. Note that will need to re-initialize in the event of a re-selection of a NBO.
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    # Record the actual prompt. Note that this is different from above as the prompt will be the raw text not formatted for the API request 
    if 'userPrompts' not in st.session_state:
        st.session_state.userPrompts = []
    # record the currently selected product stream count
    if "count" not in st.session_state:
        st.session_state.count=0
    # This portion handles the submission of a user prompt. Want the text input to disappear once the use has submitted their prompt
    if 'prompt' not in st.session_state:
        st.session_state.prompt = ''
    # Handles the text displayed in the prompt box
    if 'text' not in st.session_state:
        st.session_state.text = None
    # Handles the email text
    if 'email' not in st.session_state:
        st.session_state.email = "Please select a product above to generate an email draft."

    
    with c3:
        # Title card & formatting
        c3_heading = f"<center><h3 style='color: orange;'>{'GEN AI ANALYTICS'}</h3></center>"    
        st.markdown(c3_heading, unsafe_allow_html=True)
        st.markdown(' ')
        st.markdown(' ')
        # Select which product to generate the draft for
        selected_product = st.selectbox(
            label = "Which product would you like to draft an email for?"
            , options = ["Travel Insurance", "Auto Loan", "Mortgage"]
            , index = None
            , key = "product"
            # Need this to indicate that a new product was selected for email drafting, hence reset all the session_state history
            , on_change = product_changed,
            
        )
        # full chat button
        # full_history = st.toggle(label="Include full chat history", value=False, help='Whether or not to include the full chat history when querying the GPT API for output.\n\nActivating this may raise the cost of per query.')

        # Get the draft text. If no prompt given (as in the case of a fresh start) then pull the draft template.
        # Otherwise send a request to the API with the relevantly formatted prompt.
        if st.session_state.product is None:
            # Show a placeholder message when no product is selected
            draft = "Please select a product above to generate an email draft."
        else:
            with st.spinner(text="Generating new draft..."):
                if st.session_state.count == 0:
                    text = generate_template(customer_name=customer_name, product_selected=st.session_state.product)
                    draft = text
                    # # UNCOMMENT THE CODE BELOW to implement the chatbot "typing" effect
                    # words = re.findall(r"\S+|\n*", text)
                    # words = list(filter(lambda x : x!="", words ))
                    # with st.chat_message("assistant"):
                    #     message_placeholder = st.empty()
                    #     draft = words[0]
                    #     for word in words[1:]:
                    #         if "\n"in draft[-4:]:
                    #             draft += word + " "
                    #         else:
                    #             draft += " "+word
                    #         time.sleep(0.05)
                    #         message_placeholder.markdown(draft + "▌")
                    #     message_placeholder.markdown(draft)
                    #     time.sleep(0.70)
                    #     message_placeholder.markdown("New draft is ready for editing!")
                else:
                    prompt, messages = get_draft(prompt=st.session_state.prompt
                                , product = st.session_state.product) 
                    
                    # Updated OpenAI API call (v1.0.0+)
                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in messages
                        ],
                    )
                    draft = completion.choices[0].message.content
                    
                    # UNCOMMENT THE CODE BELOW to implement the chatbot "typing" effect with new API
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        draft = ""
                        stream = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in messages
                            ],
                            stream=True,
                        )
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                draft += chunk.choices[0].delta.content
                                message_placeholder.markdown(draft + "▌")
                        message_placeholder.markdown(draft)
                        time.sleep(0.70)
                        message_placeholder.markdown("New draft is ready for editing!")
                    
                    st.session_state.messages.append({"role": "assistant", "content": draft})
            

            
        # Display the text for editing.
        # Don't use key parameter to avoid session state conflicts
        email_draft = st.text_area(
            label='Email Draft :email::writing_hand:'
            , value=draft
            , height=450 # In pixels
            )

        prompt = st.text_input(
            label='How would you like Maybot to edit the draft?', 
            placeholder='Make it sound less formal.' if st.session_state.product else 'Select a product first', 
            key="text", 
            on_change=submit_clear,
            disabled=(st.session_state.product is None)
        )

        # Display all historical prompts
        with st.expander(label='View past prompts', expanded=False):
            final_string=''
            for idx, msg in enumerate(st.session_state.userPrompts):
                final_string += f"{idx+1}. {msg}\n\n"
            st.write(final_string)
                
        
        # submit_email = st.button('Send Email')
        # if submit_email:
        #     send_mail(send_from=send_from, subject=subject, text=email_draft, send_to=send_to, username=username, password=password)
        #     st.write('email sent')


if __name__=='__main__':
    customer_profile()