import numpy as np, numpy.random
import streamlit as st
from openai import OpenAI  # Updated import
import os 
from dotenv import load_dotenv


####################
# OPENAI COMPONENT #
####################

try:
    api_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found in secrets or environment variables")

os.environ['OPENAI_API_KEY']=api_key
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

model= "gpt-3.5-turbo"
@st.cache_data
def test_conn(idx):
    """
    Helper function that tests if the connection to the openai API is valid
    """
    model= "gpt-3.5-turbo"
    # Updated API call (v1.0.0+)
    completion = client.chat.completions.create(
        model = model
        , messages = [
            {'role' : 'user', 'content': 'answer "yes" only'}
            
        ]
    )
    response=completion.choices[0].message
    content = response.content
    return content

def generate_prompt(email_text, prompt):
    '''
        This function generates the request body to be sent over to the API for query
    '''
    full_prompt = f"Written below in '' is an email to be sent to a customer, advertising a new product they may be interested in. "\
                f"\n'{email_text}'"\
                f"\n\nWith respect to this email draft, "\
                f"{prompt}"
    return {'role' : 'user', 'content' : full_prompt}

def construct_request(chat_history, email_text, prompt, customer_name, full_chain=False):
    """
        Given an input prompt and with access to the previous message history for this stream, queries the openai API and to obtain the output

        NOTE for now we do not use the full messages list as that increases the computation time & cost due to the increased number of tokens - updated 20231022

        Calls generate_prompt() to obtaint the prompt properly formatted

        Parameters
        ----------
        chat_history : list[dict]
            List of chat messages in the properly formatted prompt - response dictionary object. 
        email_text : string
            The current draft email. There is a need to include the full email text for every prompt (even if it's the 2nd or 3rd prompt) because a user could edit the draft text after receiving the output from the GPT model. 
        prompt : string
            The prompt as given by the user
        full_chain : boolean, default False
            Whether or not to include the full conversation history in the API request. Including the full history increases the cost of each query since more tokens are included in the request.
        
        Returns
        -------
        (dict, dict)
            Returns the formatted dictionary object for both the prompt and the API response
    """
    # Generate the properly formatted question prompt
    question_prompt = generate_prompt(
        email_text = email_text
        , prompt = prompt
    )

    # QUERY PARAMETERS. These are fixed parameters passed to the API with each request. 
    setup_context = {'role' : 'system', 
                    'content': f'You work at CBank drafting emails to potential customers. Your name is CareBot. Make sure to use the name of the customer: {customer_name}.'}

    if full_chain:
        messages = [setup_context]
        messages.extend(chat_history)
    else:
        messages = [setup_context]

    messages.append(question_prompt)
    return (question_prompt, messages)
    #DEPRECATED FOLLOWING UPDATE TO ACCEPT STREAMING#
    # # Obtain the response object
    # completion = openai.ChatCompletion.create(
    #     model = model
    #     , messages = messages
    # )
    # # obtain the response
    # response = completion.choices[0].message
    # return (question_prompt, response)

#########################
# RECCOMENDATION SYSTEM #
#########################

# A function for obtaining cross-selling distributions
# Placeholder that generates dummy values for now
@st.cache_data
def get_distribution(customer_traits):
    ### PLACE HOLDER ###
    # Randomly generate 4 numbers that sum up to 1
    items= ["Car Loan", "Credit Card", "Savings Account", "Mortgage"]
    probs = np.random.dirichlet(np.ones(len(items)),size=1).flatten()
    results = list(zip(items, probs))
    return results

# OTHER UTILITIES #

def generate_template(customer_name, product_selected, your_name="CareBot", your_title="CBank's Virtual assistant"
                        , your_bank='CBank', contact_info='+60 123456'):
    ''''
    Helper function used to generate template text when a cross-selling item is first selected
    '''
    text_templates = {
        'Travel Insurance': \
                        f"Subject: Secure Your Adventures with Travel Insurance"\
                        f"\n\nDear {customer_name},"\
                        f"\n\nWe trust your travel plans are taking shape, and we want to ensure your journeys are worry-free. Although you haven't specifically inquired, we'd like to introduce you to our Travel Insurance packages. Whether it's a quick weekend getaway or an epic adventure abroad, our policies offer comprehensive coverage for unforeseen events, giving you peace of mind while exploring the world."\
                        f"\n\n If you're interested in safeguarding your travels, please reply to this email, and we'll be happy to provide more information and help you choose the right plan for your upcoming adventures."\
                        f"\n\nBest regards,"\
                        f"\n\n{your_name}"\
                        f"\n{your_title}"\

        , 'Auto Loan' : \
                        f"Subject: Realise Your Automotive Dreams"\
                        f"\n\nDear {customer_name},"\
                        f"\n\nWe hope you're well. Have you considered upgrading your vehicle? While you haven't mentioned it, we believe our Auto Loans could help you get behind the wheel of your dream car. With competitive rates and flexible terms, our loans make it easier to finance your new or used vehicle."\
                        f"\n\nIf you're thinking about a change, please reply to this email, and our team will be happy to discuss your options and assist you in making that dream car a reality."\
                        f"\n\nBest regards,"\
                        f"\n\n{your_name}"\
                        f"\n{your_title}"\

        , 'Car Loan' : \
                        f"Subject: Realise Your Automotive Dreams"\
                        f"\n\nDear {customer_name},"\
                        f"\n\nAt {your_bank}, we value our customers' financial needs and continuously strive to provide solutions that align with your aspirations. We have noticed your recent transactions and financial history, and we understand that you may be considering purchasing a new vehicle. Our competitive car loan products that can help you turn your dreams into reality."\
                        f"\n\nWith flexible terms, competitive interest rates, and a quick approval process, our car loans are designed to make your car buying experience hassle-free. Whether you're in the market for a new or used vehicle, we're here to support you every step of the way. If you would like more information or have any questions, please don't hesitate to contact us. We're here to help you achieve your automotive goals."\
                        f"\n\nBest regards,"\
                        f"\n\n{your_name}"\
                        f"\n{your_title}"\

            , "Credit Card" : \
                                f"Subject: Exclusive Credit Card Offer Just for You"\
                                f"\n\nDear {customer_name},"\
                                f"\n\nWe hope you're enjoying your experience with our bank. We've identified that you might be a great fit for our upcoming credit card product, tailored to your financial needs. With competitive rates, exciting rewards, and exclusive benefits, it's designed to make your financial journey even more rewarding. "\
                                f"\n\nTo learn more and explore this opportunity, simply reply to this email, and our dedicated team will be delighted to assist you. Your financial well-being is our priority, and we look forward to the possibility of serving you with this new offering."\
                                f"\n\nWarm regards,"\
                                f"\n\n{your_name}"\
                                f"\n{your_title}"\

            , "Savings Account" : \
                                f"Subject: Discover a Better Way to Save with Our Personal Savings Account"\
                                f"\n\nDear {customer_name},"\
                                f"\n\nWe hope you've been enjoying your banking experience with us. We've noticed an opportunity that could significantly enhance your financial future - our new Personal Savings Account. Tailored to help you achieve your savings goals, it offers competitive interest rates, easy access to your funds, and a host of benefits." \
                                f"\n\nYou may not have heard about this account yet, but we believe it could be a valuable addition to your financial portfolio. If you'd like to learn more, simply reply to this email, and our dedicated team will be delighted to assist you in taking the next step towards a more secure financial future."\
                                f"\n\nWarm regards,"\
                                f"\n\n{your_name}"\
                                f"\n{your_title}"\

            , "Mortgage" : \
                            f"Subject: Explore Your Homeownership Dreams with Our Mortgage Solutions"\
                            f"\n\nDear {customer_name},"\
                            f"\n\nWe hope you're doing well. At {your_bank}, we understand that homeownership is a significant milestone. Even though you haven't mentioned it, we believe you may be interested in our range of mortgage solutions. Whether you're a first-time homebuyer, looking to refinance, or considering a new property investment, our mortgage experts can help you find the perfect financing option."\
                            f"\n\nTo discuss your homeownership aspirations and explore how we can assist you, please reply to this email. We're here to make your homeownership dreams a reality."\
                            f"\n\nWarm regards,"
                            f"\n\n{your_name}"\
                            f"\n{your_title}"\
    }
    if product_selected:
        return text_templates[product_selected] 
    else:
        return 'Select a product to draft an email'