import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import base64
from nltk.util import ngrams as nltk_ngrams

nltk.download('punkt', quiet=True)

score_to_category = {
    1: 'Very Dissatisfied',
    2: 'Dissatisfied',
    3: 'Neutral',
    4: 'Satisfied',
    5: 'Very Satisfied'
}


def initialize_state():
    # Initialize session states with default values if not already present
    keys = ['previous_dashboard', 'selected_role', 'selected_function', 'selected_location', 'uploaded_file']
    defaults = [None, [], [], [], None]
    for key, default in zip(keys, defaults):
        if key not in st.session_state:
            st.session_state[key] = default


def reset_filters():
    st.session_state['selected_role'] = []
    st.session_state['selected_function'] = []
    st.session_state['selected_location'] = []


st.set_page_config(layout="wide")
initialize_state()

# Add a file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader("Choose a file")
st.session_state['uploaded_file'] = uploaded_file

# Load and clean data
@st.cache_data(persist=True)
def load_data():
    if st.session_state['uploaded_file'] is not None:
        data = pd.read_excel(st.session_state['uploaded_file'])
    else:
        url = 'https://github.com/001202ZHENG/V1_Chatbot_Streamlit/raw/main/data/Voice%20of%20Customer_Second%20data%20set.xlsx'
        data = pd.read_excel(url)
    return data


data = load_data()

# General Page Layout
st.markdown(
    '''
    <style>
        .main .block-container {
            padding-top: 0.25rem;
            padding-right: 0.25rem;
            padding-left: 0.25rem;
            padding-bottom: 0.25rem;
        }
        h1 {
            margin-top: 0rem;
            margin-bottom: 0rem;
        }
        h3 {
            margin-top: 0rem;
            margin-bottom: 0rem;
        }
    </style>
    ''',
    unsafe_allow_html=True
)


# Header Function
def render_header(title, subtitle=None):
    style = style = """
    <style>
        h1.header, h3.subheader {
            background-color: #336699; /* Steel blue background */
            color: white; /* White text color */
            text-align: center;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 0;
            padding: 15px 0;
            height: auto
        }
        h1.header {
            margin-bottom: 0;
            font-size: 30px;
        }
        h3.subheader {
            font-size: 20px;
            font-weight: normal;
            margin-top: 0;
        }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    st.markdown(f'<h1 class="header">{title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<h3 class="subheader">{subtitle}</h3>', unsafe_allow_html=True)

st.sidebar.markdown('For detailed reason analysis/sentiment dashboard, please check out the [link](https://gucciouy5ardhonqumm6p4.streamlit.app)')

# Sidebar for dashboard selection
dashboard = st.sidebar.radio("Select Dashboard", ('General Survey Results',
                                                  'Section 1: Employee Experience',
                                                  'Section 2: Recruiting & Onboarding',
                                                  'Section 3: Performance & Talent',
                                                  'Section 4: Learning',
                                                  'Section 5: Compensation',
                                                  'Section 6: Payroll',
                                                  'Section 7: Time Management',
                                                  'Section 8: User Experience'
                                                  ))

if dashboard != st.session_state['previous_dashboard']:
    reset_filters()  # Reset filters if dashboard changed
    st.session_state['previous_dashboard'] = dashboard


@st.cache_data
def get_unique_values(column):
    return data[column].unique()


roles = get_unique_values('What is your role at the company ?')
functions = get_unique_values('What function are you part of ?')
locations = get_unique_values('Where are you located ?')

st.session_state['selected_role'] = st.sidebar.multiselect('Select Role', options=roles, default=None)
st.session_state['selected_function'] = st.sidebar.multiselect('Select Function', options=functions, default=None)
st.session_state['selected_location'] = st.sidebar.multiselect('Select Location', options=locations, default=None)


def apply_filters(data, roles, functions, locations):
    filtered = data
    if roles:
        filtered = filtered[filtered['What is your role at the company ?'].isin(roles)]
    if functions:
        filtered = filtered[filtered['What function are you part of ?'].isin(functions)]
    if locations:
        filtered = filtered[filtered['Where are you located ?'].isin(locations)]
    return filtered


# Use the function with both a title and a subtitle
if dashboard == 'General Survey Results':
    render_header("General Survey Results")
elif dashboard == 'Section 1: Employee Experience':
    render_header("Employee Experience: General HR Services Evaluation")
elif dashboard == 'Section 2: Recruiting & Onboarding':
    render_header("Recruiting & Onboarding")
elif dashboard == 'Section 3: Performance & Talent':
    render_header("Performance & Talent")
elif dashboard == 'Section 4: Learning':
    render_header("Learning")
elif dashboard == 'Section 5: Compensation':
    render_header("Compensation")
elif dashboard == 'Section 6: Payroll':
    render_header("Payroll")
elif dashboard == 'Section 7: Time Management':
    render_header("Time Management")
elif dashboard == 'Section 8: User Experience':
    render_header("User Experience")


def prepare_summaries(data):
    continent_to_country_code = {
        'Asia': 'KAZ',
        'Oceania': 'AUS',
        'North America': 'CAN',
        'South America': 'BRA',
        'Europe': 'DEU',
        'Africa': 'TCD'
    }
    country_code_to_continent = {v: k for k, v in continent_to_country_code.items()}
    location_summary = pd.DataFrame(data['Where are you located ?'].value_counts()).reset_index()
    location_summary.columns = ['Continent', 'Count']
    location_summary['Country_Code'] = location_summary['Continent'].map(continent_to_country_code)
    location_summary['Label'] = location_summary['Continent'].apply(
        lambda x: f"{x}: {location_summary.loc[location_summary['Continent'] == x, 'Count'].iloc[0]}")

    role_summary = pd.DataFrame(data['What is your role at the company ?'].value_counts()).reset_index()
    role_summary.columns = ['Role', 'Count']
    role_summary = role_summary.sort_values('Count', ascending=True)
    function_summary = pd.DataFrame(data['What function are you part of ?'].value_counts()).reset_index()
    function_summary.columns = ['Function', 'Count']
    function_summary = function_summary.sort_values('Count', ascending=True)
    return location_summary, role_summary, function_summary


############ GENERAL DASHBOARD STARTS ############
if dashboard == "General Survey Results":
    filtered_data = apply_filters(data, st.session_state['selected_role'], st.session_state['selected_function'],
                                  st.session_state['selected_location'])

    st.markdown(
        """
        <style>
        .top-bar {
            background-color: #f0f2f6;  /* Light grey background */
            text-align: left;
            display: flex;
            justify-content: flex-start;
            align-items: center;
            height: auto;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # The top bar with centered and styled text
    st.markdown(
        f'<div class="top-bar" style="font-weight: normal; font-size: 17px; padding: 10px 20px 10px 20px; color: #333333;"> The survey has  &nbsp;<strong>{len(data)}</strong>&nbsp; respondents in total, distributed among different locations, roles and function.</div>',
        unsafe_allow_html=True
    )

    # Prepare the summaries for the filtered data
    def prepare_summaries(data):
        continent_to_country_code = {
            'Asia': 'KAZ',
            'Oceania': 'AUS',
            'North America': 'CAN',
            'South America': 'BRA',
            'Europe': 'DEU',
            'Africa': 'TCD'
        }
        country_code_to_continent = {v: k for k, v in continent_to_country_code.items()}
        location_summary = pd.DataFrame(data['Where are you located ?'].value_counts()).reset_index()
        location_summary.columns = ['Continent', 'Count']
        location_summary['Country_Code'] = location_summary['Continent'].map(continent_to_country_code)
        location_summary['Label'] = location_summary['Continent'].apply(
            lambda x: f"{x}: {location_summary.loc[location_summary['Continent'] == x, 'Count'].iloc[0]}")

        role_summary = pd.DataFrame(data['What is your role at the company ?'].value_counts()).reset_index()
        role_summary.columns = ['Role', 'Count']
        role_summary = role_summary.sort_values('Count', ascending=True)
        function_summary = pd.DataFrame(data['What function are you part of ?'].value_counts()).reset_index()
        function_summary.columns = ['Function', 'Count']
        function_summary = function_summary.sort_values('Count', ascending=True)
        return location_summary, role_summary, function_summary

    location_summary, role_summary, function_summary = prepare_summaries(filtered_data)

    st.markdown(
        """
        <style>
        .text-container {
            font-size: 15px;
            padding: 10px 0px;
            color: #333333;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # A text container for filtering instructions
    st.markdown(
    f"""
    <div class="text-container" style="font-style: italic;">
    <br>
    Filter the data by selecting tags from the sidebar. The charts below will be updated to reflect the&nbsp;
    <strong>{len(filtered_data)}</strong>&nbsp;filtered respondents.
    </div>
    """,
    unsafe_allow_html=True
    )

    map_ratio = 0.5
    barcharts_ratio = 1 - map_ratio
    mark_color = '#336699'  # Steel Blue

    map_col, barcharts_col = st.columns([map_ratio, barcharts_ratio])
    # Map visualization
    with map_col:
        fig_continent = px.scatter_geo(location_summary,
                                       locations="Country_Code",
                                       size="Count",
                                       hover_name="Continent",
                                       text="Label",  # The text labels with continent names and counts
                                       color_discrete_sequence=[mark_color])

        fig_continent.update_geos(
            projection_type="natural earth",
            showcountries=True, countrycolor="lightgrey",
            showcoastlines=False, coastlinecolor="lightgrey",
            showland=True, landcolor="#F0F0F0",
            showocean=True, oceancolor="white",
            lataxis_showgrid=True,
            lonaxis_showgrid=True,
            lataxis_range=[-90, 90],
            lonaxis_range=[-180, 180]
        )

        # Update the layout for title and margins
        fig_continent.update_layout(
            title='by Continent',
            margin=dict(l=0, r=0, t=50, b=0),
            geo=dict(bgcolor='white')  # Set the background color of the geo part of the map
        )

        fig_continent.update_traces(
            marker=dict(size=location_summary['Count'] * 2, line=dict(width=0)),
            # Remove the white border by setting the line width to 0
            textposition='top center',
            textfont=dict(color='#333333', size=14)  # Set label font color and size
        )

        fig_continent.update_layout(hovermode=False)

        # Display the plot with full width
        st.plotly_chart(fig_continent, use_container_width=True)

    with barcharts_col:
        left_margin = 200  # Adjust this as necessary to align y-axes
        total_height = 460  # This is the total height for both bar charts, adjust as necessary.
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        # Horizontal bar chart for "by Role"
        fig_role = px.bar(role_summary, y='Role', x='Count', orientation='h')
        fig_role.update_layout(
            title="by Role",
            margin=dict(l=left_margin, r=0, t=50, b=0),
            height=role_chart_height,
            showlegend=False
        )
        fig_role.update_traces(marker_color=mark_color, text=role_summary['Count'], textposition='outside')
        fig_role.update_yaxes(showticklabels=True, title='')
        fig_role.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role, use_container_width=True)

        # Horizontal bar chart for "by Function"
        fig_function = px.bar(function_summary, y='Function', x='Count', orientation='h')
        fig_function.update_layout(
            title="by Function",
            margin=dict(l=left_margin, r=0, t=50, b=0),
            height=function_chart_height,
            showlegend=False
        )
        fig_function.update_traces(marker_color=mark_color, text=function_summary['Count'], textposition='outside')
        fig_function.update_yaxes(showticklabels=True, title='')
        fig_function.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function, use_container_width=True)


############ GENERAL DASHBOARD ENDS ############

##### SECTION FOR DEFINING FUNCTIONS ####
def score_distribution(data, column_index):
    # Extract the data series based on the column index
    data_series = data.iloc[:, column_index]

    # Calculate the percentage of each response
    value_counts = data_series.value_counts(normalize=True).sort_index() * 100

    # Ensure the value_counts includes all categories with zero counts for missing categories
    value_counts = value_counts.reindex(range(1, 6), fill_value=0)

    # Create the DataFrame

    # Calculate the median score
    raw_counts = data_series.value_counts().sort_index()
    scores = np.repeat(raw_counts.index, raw_counts.values)
    median_score = np.median(scores)

    return value_counts, median_score


def filter_by_satisfaction(data, satisfaction_level, column_index):
    if satisfaction_level == 'Select a satisfaction level':
        return data[data.iloc[:, column_index].notna()]
    else:
        return data[data.iloc[:, column_index] == satisfaction_options.index(satisfaction_level)]



def filter_by_comfort(data, comfort_level, column_index):
    if comfort_level != 'Select a comfort level':
        data = data[data.iloc[:, column_index] == comfort_options.index(comfort_level)]
    else:
        data = data[data.iloc[:, column_index].notna()]
    return data


############ SENTIMENT ANALYSIS FUNCTION STARTS ############
def generate_wordclouds(df, score_col_idx, reasons_col_idx, custom_stopwords):
    # Custom stopwords
    stopwords_set = set(STOPWORDS)
    stopwords_set.update(custom_stopwords)

    # Filter the DataFrame for scores 4 and 5
    df_high_scores = df[df.iloc[:, score_col_idx].isin([4, 5])]

    # Filter the DataFrame for scores 1, 2, and 3
    df_low_scores = df[df.iloc[:, score_col_idx].isin([1, 2, 3])]

    # Generate the text for word clouds
    text_high_scores = ' '.join(df_high_scores.iloc[:, reasons_col_idx].astype(str))
    text_low_scores = ' '.join(df_low_scores.iloc[:, reasons_col_idx].astype(str))

    # Generate the word clouds
    wordcloud_high_scores = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_set,
                                      collocations=False).generate(text_high_scores)
    wordcloud_low_scores = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_set,
                                     collocations=False).generate(text_low_scores)

    # Create columns for displaying the word clouds side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud for Low Scores</h3>",
            unsafe_allow_html=True)
        fig_low_scores, ax_low_scores = plt.subplots(figsize=(10, 5))
        ax_low_scores.imshow(wordcloud_low_scores, interpolation='bilinear')
        ax_low_scores.axis('off')
        st.pyplot(fig_low_scores)

    with col2:
        st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud for High Scores</h3>",
            unsafe_allow_html=True)
        fig_high_scores, ax_high_scores = plt.subplots(figsize=(10, 5))
        ax_high_scores.imshow(wordcloud_high_scores, interpolation='bilinear')
        ax_high_scores.axis('off')
        st.pyplot(fig_high_scores)

    


############ SENTIMENT ANALYSIS FUNCTION ENDS ############

# Give up this following Function for sentiment analysis dashboard because of low accuracy of NLTK in sentiment analyzer polarity scores

#def sentiment_dashboard(data_series, title):
    # Sidebar for control
    #st.sidebar.markdown("### Filter Options")
    #show_wordcloud = st.sidebar.checkbox("Show Word Cloud", value=True)
    # filter_negative = st.sidebar.checkbox("Show Negative Comments", value=False)
    # filter_positive = st.sidebar.checkbox("Show Positive Comments", value=False)

    # Initialize sentiment results and comment lists
    # sentiment_results = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    # negative_comments = []
    # positive_comments = []

    # Analyze sentiment and collect results
    # for sentence in data_series.dropna():
    #     sentiment_scores = sentiment_analyzer.polarity_scores(sentence)
    #     compound_score = sentiment_scores['compound']

    #     if compound_score <= -0.05:
    #         sentiment_results['Negative'] += 1
    #         negative_comments.append((sentence, compound_score))
    #     elif compound_score >= 0.05:
    #         sentiment_results['Positive'] += 1
    #         positive_comments.append((sentence, compound_score))
    #     else:
    #         sentiment_results['Neutral'] += 1

    # Display word cloud
    # if show_wordcloud:
    #     wordcloud = WordCloud(width=400, height=200, background_color='white').generate(' '.join(data_series.dropna()))
    #     plt.imshow(wordcloud, interpolation='bilinear')
    #     plt.axis("off")
    #     st.pyplot(plt)  # Display word cloud in Streamlit

    # Display top negative and positive comments
    # if filter_negative:
    #     st.markdown("### Top 5 Negative Comments")
    #     for comment, score in sorted(negative_comments, key=lambda x: x[1], reverse=True)[:5]:
    #         st.write(f"{comment} (Score: {score:.4f})")

    # if filter_positive:
    #     st.markdown("### Top 5 Positive Comments")
    #     for comment, score in sorted(positive_comments, key=lambda x: x[1], reverse=True)[:5]:
    #         st.write(f"{comment} (Score: {score:.4f})")

    # Create stacked bar chart for sentiment distribution
    # total = sum(sentiment_results.values())
    # proportions = {k: v / total for k, v in sentiment_results.items()}

    # fig = go.Figure()
    # cumulative_size = 0
    # for sentiment, proportion in proportions.items():
    #     color = 'lightgreen' if sentiment == 'Positive' else 'lightcoral' if sentiment == 'Negative' else 'lightgrey'
    #     fig.add_trace(go.Bar(x=[proportion], y=['Sentiment'], orientation='h', name=sentiment, base=cumulative_size,
    #                          marker=dict(color=color)))
    #     cumulative_size += proportion

    # Update layout and display chart in Streamlit
    # fig.update_layout(
    #     title="Sentiment Distribution",
    #     barmode='stack',
    #     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #     yaxis=dict(showgrid=False, zeroline=False),
    # )

    # st.plotly_chart(fig)  # Display the stacked bar chart


########## DEFINING FUNCTIONS ENDS ########

############ SECTION 1 STARTS ############
if dashboard == "Section 1: Employee Experience":

    q6ValuesCount, q6MedianScore = score_distribution(data, 11)
    q11ValuesCount, q11MedianScore = score_distribution(data, 13)

    # Question 4: What HR processes do you interact with the most in your day-to-day work ?
    q4_data = pd.DataFrame({
        'ID': data['ID'],
        'HR_Process': data['What HR processes do you interact with the most in your day-to-day work ?']
    })
    # Remove the last semicolon from each HR_Process value
    q4_data['HR_Process'] = q4_data['HR_Process'].str.rstrip(';')
    # Splitting the HR_Process values into separate lists of processes
    q4_data['HR_Process'] = q4_data['HR_Process'].str.split(';')
    # Explode the lists into separate rows while maintaining the corresponding ID
    q4_processed = q4_data.explode('HR_Process')
    # Reset index to maintain the original ID
    q4_processed.reset_index(drop=True, inplace=True)
    q4_count = q4_processed.groupby('HR_Process').size().reset_index(name='Count')

    # Question 5: In what areas do you think HR could improve its capabilities to enhance how they deliver services and support you ?
    q5_data = pd.DataFrame({
        'ID': data['ID'],
        'Improve_Area': data[
            'In what areas do you think HR could improve its capabilities to enhance how they deliver services and support you ?']
    })
    # Remove the last semicolon from each value
    q5_data['Improve_Area'] = q5_data['Improve_Area'].str.rstrip(';')
    # Splitting the values into separate lists of processes
    q5_data['Improve_Area'] = q5_data['Improve_Area'].str.split(';')
    # Explode the lists into separate rows while maintaining the corresponding ID
    q5_processed = q5_data.explode('Improve_Area')
    # Reset index to maintain the original ID
    q5_processed.reset_index(drop=True, inplace=True)
    q5_count = q5_processed.groupby('Improve_Area').size().reset_index(name='Count')

    # Question 4 and 5 combined
    # Merge the two dataset on function
    # Merge datasets by matching HR_Process and Improve_Area
    q4_q5_count = pd.merge(q4_count, q5_count, left_on='HR_Process', right_on='Improve_Area', how='outer')
    # Drop unnecessary columns
    q4_q5_count.drop(['Improve_Area'], axis=1, inplace=True)
    q4_q5_count.rename(
        columns={'HR_Process': 'HR Function', 'Count_x': 'HR_Process_Interacted', 'Count_y': 'Improvement_Areas'},
        inplace=True)
    q4_q5_count.sort_values('HR_Process_Interacted', ascending=False, inplace=True)
    # Separate 'None' row from the DataFrame
    none_row = q4_q5_count[q4_q5_count['HR Function'] == 'None']
    q4_q5_count = q4_q5_count[q4_q5_count['HR Function'] != 'None']

    # Sort 'HR_Process_Interacted' in descending order
    q4_q5_count.sort_values(by='HR_Process_Interacted', ascending=True, inplace=True)

    # Append 'None' row at the end
    q4_q5_count = pd.concat([none_row, q4_q5_count])
    # Reshape data into tidy format
    df_tidy = q4_q5_count.melt(id_vars='HR Function', var_name='Type', value_name='Count')

    # Question 7: How do you access HR Information ?
    q7_data = pd.DataFrame({'device': data["How do you access HR Information ?"]})
    q7_data['device'] = q7_data['device'].str.rstrip(';').str.split(';')
    q7_data = q7_data.explode('device')
    q7_data.dropna(inplace=True)
    # Count the occurrences of each device
    device_counts = q7_data['device'].value_counts().reset_index()
    device_counts.columns = ['device', 'count']
    # Calculate percentage
    device_counts['percentage'] = device_counts['count'] / len(q7_data) * 100


    # Question 10: Do you find the HR department responsive to your inquiries and concerns?
    q10_responsiveness_count = (data.iloc[:, 15] == 'Yes').sum()
    q10_responsiveness_pct = q10_responsiveness_count / len(data) * 100

    highest_hr_process_interacted = q4_q5_count.loc[
        q4_q5_count[q4_q5_count['HR Function'] != 'None']['HR_Process_Interacted'].idxmax(), "HR Function"]
    highest_improvement_areas = q4_q5_count.loc[
        q4_q5_count[q4_q5_count['HR Function'] != 'None']['Improvement_Areas'].idxmax(), "HR Function"]
    most_used_device = device_counts.iloc[0]['device']

    # Summary of all outputs in the bar container
    st.markdown(
    f"""
    <style>
    .top-bar {{
        background-color: #f0f2f6;  /* Light grey background */
        text-align: left;
        display: flex;
        justify-content: flex-start;
        align-items: center;
        height: auto;
        font-weight: normal;
        font-size: 17px;
        padding: 10px 20px;
        color: #333333;
        display: block;
        width: 100%;
        box-sizing: border-box;
    }}
    .top-bar ul, .top-bar li {{
        font-size: 17px;
        padding-left: 20px;
        margin: 0;
    }}
    </style>
    <div class="top-bar">
        This survey section is answered by all the <strong>{len(data)}</strong> survey participants:
        <ul>
            <li>{q10_responsiveness_pct:.0f}% of the respondents, {q10_responsiveness_count} employee(s), find the HR department responsive to their inquiries and concerns.</li>
            <li>The median satisfaction rating on overall HR services and support is {q6MedianScore}.</li>
            <li>The median satisfaction rating on the HR communication channels is {q11MedianScore}.</li>
            <li>The HR process that employees interact with the most is: {highest_hr_process_interacted}</li>
            <li>The HR process most commonly identified as needing improvement is: {highest_improvement_areas}</li>
            <li>The most common device used to access HR Information is: {most_used_device}</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
    )


    filtered_data = apply_filters(data, st.session_state['selected_role'], st.session_state['selected_function'],
                                  st.session_state['selected_location'])

    # A text container for filtering instructions
    st.markdown(
    f"""
    <div class="text-container" style="font-style: italic;">
    <br>
    Filter the data by selecting tags from the sidebar. The charts below will be updated to reflect the&nbsp;
    <strong>{len(filtered_data)}</strong>&nbsp;filtered respondents.
    </div>
    """,
    unsafe_allow_html=True
    )

    # Question 10: Do you find the HR department responsive to your inquiries and concerns?
    q10_responsiveness_count = (filtered_data.iloc[:, 15] == 'Yes').sum()
    q10_responsiveness_pct = q10_responsiveness_count / len(filtered_data) * 100

    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                {q10_responsiveness_pct:.0f}% of the respondents, {q10_responsiveness_count} employee(s), find the HR department responsive to their inquiries and concerns.</li>
                </div>
                """,
        unsafe_allow_html=True
    )

    # Question 4: What HR processes do you interact with the most in your day-to-day work ?
    q4_data = pd.DataFrame({
        'ID': filtered_data['ID'],
        'HR_Process': filtered_data['What HR processes do you interact with the most in your day-to-day work ?']
    })
    # Remove the last semicolon from each HR_Process value
    q4_data['HR_Process'] = q4_data['HR_Process'].str.rstrip(';')
    # Splitting the HR_Process values into separate lists of processes
    q4_data['HR_Process'] = q4_data['HR_Process'].str.split(';')
    # Explode the lists into separate rows while maintaining the corresponding ID
    q4_processed = q4_data.explode('HR_Process')
    # Reset index to maintain the original ID
    q4_processed.reset_index(drop=True, inplace=True)
    q4_count = q4_processed.groupby('HR_Process').size().reset_index(name='Count')

    # Question 5: In what areas do you think HR could improve its capabilities to enhance how they deliver services and support you ?
    q5_data = pd.DataFrame({
        'ID': filtered_data['ID'],
        'Improve_Area': filtered_data[
            'In what areas do you think HR could improve its capabilities to enhance how they deliver services and support you ?']
    })
    # Remove the last semicolon from each value
    q5_data['Improve_Area'] = q5_data['Improve_Area'].str.rstrip(';')
    # Splitting the values into separate lists of processes
    q5_data['Improve_Area'] = q5_data['Improve_Area'].str.split(';')
    # Explode the lists into separate rows while maintaining the corresponding ID
    q5_processed = q5_data.explode('Improve_Area')
    # Reset index to maintain the original ID
    q5_processed.reset_index(drop=True, inplace=True)
    q5_count = q5_processed.groupby('Improve_Area').size().reset_index(name='Count')

    # Question 4 and 5 combined
    # Merge the two dataset on function
    # Merge datasets by matching HR_Process and Improve_Area
    q4_q5_count = pd.merge(q4_count, q5_count, left_on='HR_Process', right_on='Improve_Area', how='outer')
    # Drop unnecessary columns
    q4_q5_count.drop(['Improve_Area'], axis=1, inplace=True)
    q4_q5_count.rename(
        columns={'HR_Process': 'HR Function', 'Count_x': 'HR_Process_Interacted', 'Count_y': 'Improvement_Areas'},
        inplace=True)
    q4_q5_count.sort_values('HR_Process_Interacted', ascending=False, inplace=True)
    # Separate 'None' row from the DataFrame
    none_row = q4_q5_count[q4_q5_count['HR Function'] == 'None']
    q4_q5_count = q4_q5_count[q4_q5_count['HR Function'] != 'None']

    # Sort 'HR_Process_Interacted' in descending order
    q4_q5_count.sort_values(by='HR_Process_Interacted', ascending=True, inplace=True)

    # Append 'None' row at the end
    q4_q5_count = pd.concat([none_row, q4_q5_count])
    # Reshape data into tidy format
    df_tidy = q4_q5_count.melt(id_vars='HR Function', var_name='Type', value_name='Count')

    # Question 7: How do you access HR Information ?
    q7_data = pd.DataFrame({'device': filtered_data["How do you access HR Information ?"]})
    q7_data['device'] = q7_data['device'].str.rstrip(';').str.split(';')
    q7_data = q7_data.explode('device')
    q7_data.dropna(inplace=True)
    # Count the occurrences of each device
    device_counts = q7_data['device'].value_counts().reset_index()
    device_counts.columns = ['device', 'count']
    # Calculate percentage
    device_counts['percentage'] = device_counts['count'] /len(q7_data) * 100

    satisfaction_ratio = 0.6
    barcharts_ratio = 1 - satisfaction_ratio
    satisfaction_col, barcharts_col = st.columns([satisfaction_ratio, barcharts_ratio])

    st.markdown("""
        <style>
        .chart-container {
            padding-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q6ValuesCount, q6MedianScore = score_distribution(filtered_data, 11)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q6ValuesCount.values})

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Overall Rating on HR Services and Support</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q6MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Satisfaction Level', color_discrete_map={
                'Very Dissatisfied': '#440154',  # Dark purple
                'Dissatisfied': '#3b528b',  # Dark blue
                'Neutral': '#21918c',  # Cyan
                'Satisfied': '#5ec962',  # Light green
                'Very Satisfied': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="overall_rating_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_options = ['Select a satisfaction level', 'Very Dissatisfied', 'Dissatisfied', 'Neutral',
                                'Satisfied', 'Very Satisfied']
        satisfaction_dropdown1 = st.selectbox('', satisfaction_options,
                                              key='satisfaction_dropdown1')

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown1, 11)

        location_summary1, role_summary1, function_summary1 = prepare_summaries(satisfaction_filtered_data1)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role1 = px.bar(role_summary1, y='Role', x='Count', orientation='h')
        fig_role1.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role1.update_traces(marker_color='#336699', text=role_summary1['Count'], textposition='outside')
        fig_role1.update_yaxes(showticklabels=True, title='')
        fig_role1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role1, use_container_width=True, key="roles_bar_chart1")

        fig_function1 = px.bar(function_summary1, y='Function', x='Count', orientation='h')
        fig_function1.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function1.update_traces(marker_color='#336699', text=function_summary1['Count'], textposition='outside')
        fig_function1.update_yaxes(showticklabels=True, title='')
        fig_function1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function1, use_container_width=True, key="functions_bar_chart1")

    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q11ValuesCount, q11MedianScore = score_distribution(filtered_data, 13)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q11ValuesCount.values})

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)        

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on HR Communication Channels</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q11MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Satisfaction Level', color_discrete_map={
                'Very Dissatisfied': '#440154',  # Dark purple
                'Dissatisfied': '#3b528b',  # Dark blue
                'Neutral': '#21918c',  # Cyan
                'Satisfied': '#5ec962',  # Light green
                'Very Satisfied': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="rating_hr_communication_channels_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_dropdown2 = st.selectbox('', satisfaction_options,
                                              key='satisfaction_dropdown2')

        satisfaction_filtered_data2 = filter_by_satisfaction(filtered_data, satisfaction_dropdown2, 13)

        location_summary2, role_summary2, function_summary2 = prepare_summaries(satisfaction_filtered_data2)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role2 = px.bar(role_summary2, y='Role', x='Count', orientation='h')
        fig_role2.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role2.update_traces(marker_color='#336699', text=role_summary2['Count'], textposition='outside')
        fig_role2.update_yaxes(showticklabels=True, title='')
        fig_role2.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role2, use_container_width=True, key="roles_bar_chart2")

        fig_function2 = px.bar(function_summary2, y='Function', x='Count', orientation='h')
        fig_function2.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function2.update_traces(marker_color='#336699', text=function_summary2['Count'], textposition='outside')
        fig_function2.update_yaxes(showticklabels=True, title='')
        fig_function2.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function2, use_container_width=True, key="functions_bar_chart2")

    # Define colors for each device
    colors = {'Computer': '#440154', 'Mobile': '#5ec962', 'Tablet': '#3b528b'}

    # Set up space for two visualizations
    fig_q4_ratio = 0.65
    fig_q7_ratio = 1 - fig_q4_ratio
    q4_col, q7_col = st.columns([fig_q4_ratio, fig_q7_ratio])

    with q4_col:
        # Plot for HR Processes in the first column
        fig_q4 = go.Figure(data=[
            go.Bar(
                name='Improvement Areas',
                y=df_tidy[df_tidy['Type'] == 'Improvement_Areas']['HR Function'],
                # make it horizontal bar chart to show texts completely
                x=df_tidy[df_tidy['Type'] == 'Improvement_Areas']['Count'],
                # make it horizontal bar chart to show texts completely
                marker_color='#3b528b',
                orientation='h'  # make it horizontal bar chart to show texts completely
            ),
            go.Bar(
                name='Employee Interaction',
                y=df_tidy[df_tidy['Type'] == 'HR_Process_Interacted']['HR Function'],
                # make it horizontal bar chart to show texts completely
                x=df_tidy[df_tidy['Type'] == 'HR_Process_Interacted']['Count'],
                # make it horizontal bar chart to show texts completely
                marker_color='#5ec962',
                orientation='h'  # make it horizontal bar chart to show texts completely
            )
        ])
        fig_q4.update_layout(
            title='HR Processes: Employee Interaction vs Improvement Areas',
            title_font=dict(size=17, family="Arial", color='#333333'),
            xaxis_title='Number of Respondents',
            yaxis_title='HR Process',
            barmode='group',
            annotations=[
                dict(
                    xref='paper', yref='paper', x=0, y=1.1,
                    xanchor='left', yanchor='top',
                    text="<i>Each respondent is able to select more than one HR process</i>",
                    font=dict(family='Arial', size=12, color='#707070'),
                    showarrow=False)
            ],
            legend=dict(
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.2,
                yanchor="top"
            ),
            margin=dict(l=22, r=20, t=70, b=70)
        )
        st.plotly_chart(fig_q4, use_container_width=True)

    # Plot for Device Usage in the second column
    with q7_col:
        fig_q7 = px.bar(device_counts, x='percentage', y='device', text='percentage', orientation='h', color='device',
                        color_discrete_map=colors)
        fig_q7.update_layout(
            title='Devices Used to Access HR Information',
            title_font=dict(size=17, family="Arial", color='#333333'),
            xaxis={'visible': False, 'showticklabels': False},
            yaxis_title=None,
            showlegend=False
        )
        fig_q7.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
        st.plotly_chart(fig_q7, use_container_width=True)

    # Question 9: Which reason(s) drive that score ?
    # Display the reasons for communication channel satisfaction
    st.markdown(
        '<h1 style="font-size:17px;font-family:Arial;color:#333333;">The Reasons for Ratings on Communication Channels</h1>',
        unsafe_allow_html=True)

    # Example usage
    communication_stopwords = ["communication", "channels", "HR", "information", "important", "informed", "stay",
                               "communicated", "employees", "company", "help", "communicates", "need", "everyone",
                               "makes"]

    # Run this code in a Streamlit app
    if __name__ == "__main__":
        st.markdown(
            "<h1 style='text-align: center; font-size: 24px; font-weight: normal;'>Word Cloud Visualization</h1>",
            unsafe_allow_html=True)
        generate_wordclouds(filtered_data, 13, 14, communication_stopwords)

    st.write('For detailed reason analysis/sentiment dashboard, please check out the [link](https://gucciouy5ardhonqumm6p4.streamlit.app)')

############ SECTION 1 ENDS ############


############ SECTION 2 STARTS ############
if dashboard == 'Section 2: Recruiting & Onboarding':
    filtered_data = apply_filters(data, st.session_state['selected_role'], st.session_state['selected_function'],
                                  st.session_state['selected_location'])

    ### Question11: How long have you been part of the company ?
    q11_data_available_count = (data.iloc[:, 16] == 'Less than a year').sum()
    q11_data_available_pct = q11_data_available_count / len(data) * 100

    q12ValuesCount, q12MedianScore = score_distribution(data, 17)

    q13a_data = pd.DataFrame({'negative_reasons': data.iloc[:, 18]})
    q13a_data['negative_reasons'] = q13a_data['negative_reasons'].str.rstrip(';').str.split(';')
    q13a_data = q13a_data.explode('negative_reasons')
    q13a_data.dropna(inplace=True)

    # Count the occurrences of each negative reason
    negative_reason_counts = q13a_data['negative_reasons'].value_counts().reset_index()
    negative_reason_counts.columns = ['Reason', 'Count']
    # Find the most common reason for dissatisfaction
    most_common_reason_for_dissatisfaction = negative_reason_counts.iloc[0]['Reason']

    q13b_data = pd.DataFrame({'positive_reasons': data.iloc[:, 19]})
    q13b_data['positive_reasons'] = q13b_data['positive_reasons'].str.rstrip(';').str.split(';')
    q13b_data = q13b_data.explode('positive_reasons')
    q13b_data.dropna(inplace=True)

    # Count the occurrences of each positive reason
    positive_reason_recruiting_counts = q13b_data['positive_reasons'].value_counts().reset_index()
    positive_reason_recruiting_counts.columns = ['positive_reasons', 'count']
    # Find the most common reason for dissatisfaction
    most_common_reason_for_satisfaction = positive_reason_recruiting_counts.iloc[0]['positive_reasons']

    q14_data = pd.DataFrame({'recruting process that required improvement': data.iloc[:, 20]})

    q14_data['recruting process that required improvement'] = q14_data[
        'recruting process that required improvement'].str.rstrip(';').str.split(';')
    q14_data = q14_data.explode('recruting process that required improvement')
    q14_data.dropna(inplace=True)

    # Count the occurrences of each aspect that required improvement
    aspect_recruiting_counts = q14_data['recruting process that required improvement'].value_counts().reset_index()
    aspect_recruiting_counts.columns = ['recruting process that required improvement', 'count']
    most_chosen_aspect = aspect_recruiting_counts.iloc[0]['recruting process that required improvement']

    q15ValuesCount, q15MedianScore = score_distribution(data, 21)

    q16a_data = pd.DataFrame({'negative_reasons': data.iloc[:, 22]})
    q16a_data['negative_reasons'] = q16a_data['negative_reasons'].str.rstrip(';').str.split(';')
    q16a_data = q16a_data.explode('negative_reasons')
    q16a_data.dropna(inplace=True)

    # Count the occurrences of each negative reason
    negative_reason_recruiting_counts = q16a_data['negative_reasons'].value_counts().reset_index()
    negative_reason_recruiting_counts.columns = ['negative_reasons', 'count']
    most_common_reason_for_dissatisfaction2 = negative_reason_recruiting_counts.iloc[0]['negative_reasons']

    q16b_data = pd.DataFrame({'positive_reasons': data.iloc[:, 23]})
    q16b_data['positive_reasons'] = q16b_data['positive_reasons'].str.rstrip(';').str.split(';')
    q16b_data = q16b_data.explode('positive_reasons')
    q16b_data.dropna(inplace=True)

    # Count the occurrences of each positive reason
    positive_reason_recruiting_counts = q16b_data['positive_reasons'].value_counts().reset_index()
    positive_reason_recruiting_counts.columns = ['positive_reasons', 'count']
    most_common_reason_for_satisfaction2 = positive_reason_recruiting_counts.iloc[0]['positive_reasons']

    q17_data = pd.DataFrame({'helpful_onboarding_process': data.iloc[:, 24]})
    q17_data['helpful_onboarding_process'] = q17_data['helpful_onboarding_process'].str.rstrip(';').str.split(';')
    q17_data = q17_data.explode('helpful_onboarding_process')
    q17_data.dropna(inplace=True)

    # Count the occurrences of each aspect that required improvement
    helpful_onboarding_counts = q17_data['helpful_onboarding_process'].value_counts().reset_index()
    helpful_onboarding_counts.columns = ['helpful_onboarding_process', 'count']
    most_helpful = helpful_onboarding_counts.iloc[0]['helpful_onboarding_process']

    # onboarding process to improve
    q18_data = pd.DataFrame({'onboarding_process_to_improve': data.iloc[:, 25]})
    q18_data['onboarding_process_to_improve'] = q18_data['onboarding_process_to_improve'].str.rstrip(';').str.split(';')
    q18_data = q18_data.explode('onboarding_process_to_improve')
    q18_data.dropna(inplace=True)

    # Count the occurrences of each aspect that required improvement
    aspect_onboarding_counts = q18_data['onboarding_process_to_improve'].value_counts().reset_index()
    aspect_onboarding_counts.columns = ['onboarding_process_to_improve', 'count']
    to_be_improved = aspect_onboarding_counts.iloc[0]['onboarding_process_to_improve']
    
    # Summary of all outputs in the bar container
    st.markdown(
    f"""
    <style>
    .top-bar {{
        background-color: #f0f2f6;  /* Light grey background */
        text-align: left;
        display: flex;
        justify-content: flex-start;
        align-items: center;
        height: auto;
        font-weight: normal;
        font-size: 17px;
        padding: 10px 20px;
        color: #333333;
        display: block;
        width: 100%;
        box-sizing: border-box;
    }}
    .top-bar ul, .top-bar li {{
        font-size: 17px;
        padding-left: 20px;
        margin: 0;
    }}
    </style>
    <div class="top-bar">
            This survey section is answered by <b>{q11_data_available_count}</b> employee(s), who have been with the company for less than a year:
        <ul>
        <li>The median satisfaction rating on the recruitment process is {q12MedianScore}.</li>
        <li>The most common reason for their dissatisfaction with the recruitment process is: {most_common_reason_for_dissatisfaction}.</li>
        <li>The most common reason for their satisfaction with the recruitment process is: {most_common_reason_for_satisfaction}.</li>
        <li>The most chosen aspect of the recruiting process that requires improvement is: {most_chosen_aspect}</li>
        <li>The median satisfaction rating on the onboarding process is: {q15MedianScore}.</li>
        <li>The most common reason for their dissatisfaction with the onboarding process is: {most_common_reason_for_dissatisfaction2}.</li>
        <li>the most common reason for their satisfaction with the onboarding process is: {most_common_reason_for_satisfaction2}.</li>
        <li>The part of the onboarding process considered most helpful is: {most_helpful}</li>
        <li>The part of the onboarding process most frequently requested for improvement is: {to_be_improved}</li>
        </ul>
        </div>
                """,
        unsafe_allow_html=True
    )

    # A text container for filtering instructions
    st.markdown(
        f"""
        <br>
        <div class="text-container" style="font-style: italic;">
        Filter the data by selecting tags from the sidebar. The charts below will be updated to reflect the&nbsp;
        <strong>{len(filtered_data)}</strong>&nbsp;filtered respondents.
        </div>
        """,
        unsafe_allow_html=True
    )


    ### Question12: How would rate the recruiting process ?
    satisfaction_ratio = 0.6
    barcharts_ratio = 1 - satisfaction_ratio
    satisfaction_col, barcharts_col = st.columns([satisfaction_ratio, barcharts_ratio])

    st.markdown("""
        <style>
        .chart-container {
            padding-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q12ValuesCount, q12MedianScore = score_distribution(filtered_data, 17)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q12ValuesCount.values})
        
        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on the Recruiting Process</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q12MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Satisfaction Level', color_discrete_map={
                'Very Dissatisfied': '#440154',  # Dark purple
                'Dissatisfied': '#3b528b',  # Dark blue
                'Neutral': '#21918c',  # Cyan
                'Satisfied': '#5ec962',  # Light green
                'Very Satisfied': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="overall_rating_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_options = ['Select a satisfaction level', 'Very Dissatisfied', 'Dissatisfied', 'Neutral',
                                'Satisfied', 'Very Satisfied']
        satisfaction_dropdown1 = st.selectbox('', satisfaction_options,
                                              key='satisfaction_dropdown1')

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown1, 17)

        location_summary1, role_summary1, function_summary1 = prepare_summaries(satisfaction_filtered_data1)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role1 = px.bar(role_summary1, y='Role', x='Count', orientation='h')
        fig_role1.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role1.update_traces(marker_color='#336699', text=role_summary1['Count'], textposition='outside')
        fig_role1.update_yaxes(showticklabels=True, title='')
        fig_role1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role1, use_container_width=True, key="roles_bar_chart1")

        fig_function1 = px.bar(function_summary1, y='Function', x='Count', orientation='h')
        fig_function1.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function1.update_traces(marker_color='#336699', text=function_summary1['Count'], textposition='outside')
        fig_function1.update_yaxes(showticklabels=True, title='')
        fig_function1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function1, use_container_width=True, key="functions_bar_chart1")

    ### Question13: What reason(s) drive that score ?
    col1, col2 = st.columns(2)

    # Display the reasons for negative ratings
    with col1:
        # Generate data for negative reasons
        q13a_data = pd.DataFrame({'negative_reasons': filtered_data.iloc[:, 18]})
        q13a_data['negative_reasons'] = q13a_data['negative_reasons'].str.rstrip(';').str.split(';')
        q13a_data = q13a_data.explode('negative_reasons')
        q13a_data.dropna(inplace=True)
        negative_reason_recruiting_counts = q13a_data['negative_reasons'].value_counts().reset_index()
        negative_reason_recruiting_counts.columns = ['negative_reasons', 'count']
        negative_reason_recruiting_counts['percentage'] = negative_reason_recruiting_counts['count'] / len(filtered_data) * 100

        # Create and display the treemap for negative reasons
        custom_color_scale = ['#2a4170', '#3b528b', '#4c669a', '#5d7ca6', '#6e92b8']
        fig10 = px.treemap(
            negative_reason_recruiting_counts,
            path=['negative_reasons'],
            values='count',
            color='count',
            color_continuous_scale=custom_color_scale,
            title='Reasons behind the Negative Ratings on the Recruiting Process'
        )
        fig10.update_layout(
            margin=dict(t=50, l=25, r=25, b=25),  # Adjust margin to allow more space
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig10, use_container_width=True)

    # Display the title for positive reasons
    with col2:
        # Generate data for positive reasons
        q13b_data = pd.DataFrame({'positive_reasons': filtered_data.iloc[:, 19]})
        q13b_data['positive_reasons'] = q13b_data['positive_reasons'].str.rstrip(';').str.split(';')
        q13b_data = q13b_data.explode('positive_reasons')
        q13b_data.dropna(inplace=True)
        positive_reason_recruiting_counts = q13b_data['positive_reasons'].value_counts().reset_index()
        positive_reason_recruiting_counts.columns = ['positive_reasons', 'count']
        positive_reason_recruiting_counts['percentage'] = positive_reason_recruiting_counts['count'] / len(filtered_data) * 100

        # Create and display the treemap for positive reasons
        custom_color_scale = ['#429c62', '#5ec962', '#7cd982', '#9de9a2', '#bef9c2']
        fig9 = px.treemap(
            positive_reason_recruiting_counts,
            path=['positive_reasons'],
            values='count',
            color='count',
            color_continuous_scale=custom_color_scale,
            title='Reasons behind the Positive Ratings on the Recruiting Process'
        )
        fig9.update_layout(
            margin=dict(t=50, l=25, r=25, b=25),
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig9, use_container_width=True)
        

    ### Question14: What aspect of the recruiting process took the most time and requires improvements ?
    q14_data = pd.DataFrame({'recruiting process that required improvement': filtered_data.iloc[:, 20]})
    q14_data['recruiting process that required improvement'] = q14_data['recruiting process that required improvement'].str.rstrip(';').str.split(';')
    q14_data = q14_data.explode('recruiting process that required improvement')
    q14_data.dropna(inplace=True)
    aspect_recruiting_counts = q14_data['recruiting process that required improvement'].value_counts().reset_index()
    aspect_recruiting_counts.columns = ['recruiting process that required improvement', 'count']
    aspect_recruiting_counts['percentage'] = aspect_recruiting_counts['count'] / len(filtered_data) * 100
    aspect_recruiting_counts = aspect_recruiting_counts.sort_values(by='percentage', ascending=True)

    # Create a horizontal bar chart
    fig8 = px.bar(
        aspect_recruiting_counts,
        y='recruiting process that required improvement',
        x='percentage',
        text='percentage',
        orientation='h',
        title='Aspects of the Recruiting Process that Require Improvements'
    )
    fig8.update_layout(
        xaxis={'visible': False},
        yaxis_title=None,
        showlegend=False,
        yaxis={'showgrid': False}
    )
    fig8.update_traces(
        marker_color='#336699',
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    st.plotly_chart(fig8, use_container_width=True)


    ### Question15: From 1 to 5, how would you rate the onboarding process ?
    satisfaction_ratio = 0.6
    barcharts_ratio = 1 - satisfaction_ratio
    satisfaction_col, barcharts_col = st.columns([satisfaction_ratio, barcharts_ratio])

    st.markdown("""
        <style>
        .chart-container {
            padding-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q15ValuesCount, q15MedianScore = score_distribution(filtered_data, 21)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q15ValuesCount.values})

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)


        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on the Onboarding Process</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q15MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Satisfaction Level', color_discrete_map={
                'Very Dissatisfied': '#440154',  # Dark purple
                'Dissatisfied': '#3b528b',  # Dark blue
                'Neutral': '#21918c',  # Cyan
                'Satisfied': '#5ec962',  # Light green
                'Very Satisfied': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="overall_rating_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_options = ['Select a satisfaction level', 'Very Dissatisfied', 'Dissatisfied', 'Neutral',
                                'Satisfied', 'Very Satisfied']
        satisfaction_dropdown15 = st.selectbox('', satisfaction_options,
                                               key='satisfaction_dropdown15')

        satisfaction_filtered_data15 = filter_by_satisfaction(filtered_data, satisfaction_dropdown15, 21)

        location_summary1, role_summary1, function_summary1 = prepare_summaries(satisfaction_filtered_data15)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role1 = px.bar(role_summary1, y='Role', x='Count', orientation='h')
        fig_role1.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role1.update_traces(marker_color='#336699', text=role_summary1['Count'], textposition='outside')
        fig_role1.update_yaxes(showticklabels=True, title='')
        fig_role1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role1, use_container_width=True, key="roles_bar_chart1")

        fig_function1 = px.bar(function_summary1, y='Function', x='Count', orientation='h')
        fig_function1.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function1.update_traces(marker_color='#336699', text=function_summary1['Count'], textposition='outside')
        fig_function1.update_yaxes(showticklabels=True, title='')
        fig_function1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function1, use_container_width=True, key="functions_bar_chart1")

    
    
    ### Question16: What reason(s) drive that score ?
    col3, col4 = st.columns(2)
    # Display the reasons for negative ratings
    with col3:
        # Generate data for negative reasons
        q16a_data = pd.DataFrame({'negative_reasons': filtered_data.iloc[:, 22]})
        q16a_data['negative_reasons'] = q16a_data['negative_reasons'].str.rstrip(';').str.split(';')
        q16a_data = q16a_data.explode('negative_reasons')
        q16a_data.dropna(inplace=True)
        negative_reason_recruiting_counts = q16a_data['negative_reasons'].value_counts().reset_index()
        negative_reason_recruiting_counts.columns = ['negative_reasons', 'count']
        negative_reason_recruiting_counts['percentage'] = negative_reason_recruiting_counts['count'] / len(
        filtered_data) * 100
        
        # Create and display the treemap for negative reasons
        custom_color_scale = ['#2a4170', '#3b528b', '#4c669a', '#5d7ca6', '#6e92b8']
        fig13 = px.treemap(
            negative_reason_recruiting_counts,
            path=['negative_reasons'],
            values='count',
            color='count',
            color_continuous_scale=custom_color_scale,
            title='Reasons behind the Negative Ratings on the Onboarding Process'
        )
        fig13.update_layout(
            margin=dict(t=50, l=25, r=25, b=25),  # Adjust margin to allow more space
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig13, use_container_width=True)
    
    with col4:
        # Generate data for positive reasons
        q16b_data = pd.DataFrame({'positive_reasons': filtered_data.iloc[:, 23]})
        q16b_data['positive_reasons'] = q16b_data['positive_reasons'].str.rstrip(';').str.split(';')
        q16b_data = q16b_data.explode('positive_reasons')
        q16b_data.dropna(inplace=True)
        positive_reason_recruiting_counts = q16b_data['positive_reasons'].value_counts().reset_index()
        positive_reason_recruiting_counts.columns = ['positive_reasons', 'count']
        positive_reason_recruiting_counts['percentage'] = positive_reason_recruiting_counts['count'] / len(
        filtered_data) * 100
 
        # Create and display the treemap for positive reasons
        custom_color_scale = ['#429c62', '#5ec962', '#7cd982', '#9de9a2', '#bef9c2']
        fig16 = px.treemap(
            positive_reason_recruiting_counts,
            path=['positive_reasons'],
            values='count',
            color='count',
            color_continuous_scale=custom_color_scale,
            title='Reasons behind the Positive Ratings on the Recruiting Process'
        )
        fig16.update_layout(
            margin=dict(t=50, l=25, r=25, b=25),
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig16, use_container_width=True)


    # Setup columns for the two bar charts
    col5, col6 = st.columns(2)

    # Question 18: What part of the Onboarding process could be improved ?
    with col5:
        q18_data = pd.DataFrame({'onboarding_process_to_improve': filtered_data.iloc[:, 25]})
        q18_data['onboarding_process_to_improve'] = q18_data['onboarding_process_to_improve'].str.rstrip(';').str.split(';')
        q18_data = q18_data.explode('onboarding_process_to_improve')
        q18_data.dropna(inplace=True)
        aspect_onboarding_counts = q18_data['onboarding_process_to_improve'].value_counts().reset_index()
        aspect_onboarding_counts.columns = ['onboarding_process_to_improve', 'count']
        aspect_onboarding_counts['percentage'] = aspect_onboarding_counts['count'] / len(filtered_data) * 100
        aspect_onboarding_counts = aspect_onboarding_counts.sort_values(by='percentage', ascending=True)
        
        fig18 = px.bar(aspect_onboarding_counts, y='onboarding_process_to_improve', x='percentage', text='percentage', orientation='h', title='Aspects of the Onboarding Process that Require Improvements', color_discrete_sequence=['#336699'])
        fig18.update_layout(
            xaxis={'visible': False},
            yaxis_title=None,
            showlegend=False,
            yaxis={'showgrid': False}
        )
        fig18.update_traces(
            marker_color='#336699',
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        st.plotly_chart(fig18, use_container_width=True)

    # Question 17: What part of the Onboarding process was particulary helpful ?
    with col6:
        q17_data = pd.DataFrame({'helpful_onboarding_process': filtered_data.iloc[:, 24]})
        q17_data['helpful_onboarding_process'] = q17_data['helpful_onboarding_process'].str.rstrip(';').str.split(';')
        q17_data = q17_data.explode('helpful_onboarding_process')
        q17_data.dropna(inplace=True)
        helpful_onboarding_counts = q17_data['helpful_onboarding_process'].value_counts().reset_index()
        helpful_onboarding_counts.columns = ['helpful_onboarding_process', 'count']
        helpful_onboarding_counts['percentage'] = helpful_onboarding_counts['count'] / len(filtered_data) * 100
        helpful_onboarding_counts = helpful_onboarding_counts.sort_values(by='percentage', ascending=True)
  
        fig17 = px.bar(helpful_onboarding_counts, y='helpful_onboarding_process', x='percentage', text='percentage', orientation='h', title='Helpful Aspects of the Onboarding Process', color_discrete_sequence=['#336699'])
        fig17.update_layout(
            xaxis={'visible': False},
            yaxis_title=None,
            showlegend=False,
            yaxis={'showgrid': False}
        )
        fig17.update_traces(
            marker_color='#5ec962',
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        st.plotly_chart(fig17, use_container_width=True)
        
############ SECTION 2 ENDS ############


############ SECTION 3 STARTS ############    
if dashboard == 'Section 3: Performance & Talent':
    filtered_data = apply_filters(data, st.session_state['selected_role'], st.session_state['selected_function'],
                                  st.session_state['selected_location'])

    q19_data_available_count = (data.iloc[:, 26])
    q19_data_available_pct = q19_data_available_count.sum() / len(data) * 100

    q19ValuesCount, q19MedianScore = score_distribution(data, 26)
    q21ValuesCount, q21MedianScore = score_distribution(data, 28)
    q22_data = pd.DataFrame({'reasons': data.iloc[:, 29]})
    q22_data['reasons'] = q22_data['reasons'].str.rstrip(';').str.split(';')
    q22_data = q22_data.explode('reasons')
    q22_data.dropna(inplace=True)

    reasons = q22_data['reasons'].value_counts().reset_index()
    reasons.columns = ['Reason', 'Count']
    most_chosen_reason = reasons.iloc[0]['Reason']

    q23_data_available_count = (data.iloc[:, 30] == 'Yes').sum()
    q23_data_available_pct = q23_data_available_count / len(data) * 100

    # Summary of all outputs in the bar container
    st.markdown(
        f"""
        <style>
        .top-bar {{
            background-color: #f0f2f6;  /* Light grey background */
            text-align: left;
            display: flex;
            justify-content: flex-start;
            align-items: center;
            height: auto;
            font-weight: normal;
            font-size: 17px;
            padding: 10px 20px;
            color: #333333;
            display: block;
            width: 100%;
            box-sizing: border-box;
        }}
        .top-bar ul, .top-bar li {{
            font-size: 17px;
            padding-left: 20px;
            margin: 0;
        }}
        </style>
        <div class="top-bar">
                This survey section is answered by all <b>{len(q19_data_available_count)}</b> survey participants<ul>
            <li>The median satisfaction rating on the company's performance evaluation and feedback process is {q19MedianScore}.</li>
            <li>The median score on how comfortable employees feel discussing career goals and development with managers is: {q21MedianScore}.</li>
            <li>The most chosen reason behind the comfort score is: {most_chosen_reason}.</li>
            <li>{q23_data_available_pct:.2f}% of the respondents, {q23_data_available_count} employee(s), are able to identify and tag their skills within the HRIS.</li>
            </ul>
            </div>
                    """,
        unsafe_allow_html=True
    )

    # A text container for filtering instructions
    st.markdown(
        f"""
        <div class="text-container" style="font-style: italic;">
        Filter the data by selecting tags from the sidebar. The charts below will be updated to reflect the&nbsp;
        <strong>{len(filtered_data)}</strong>&nbsp;filtered respondents.
        </div>
        """,
        unsafe_allow_html=True
    )

    ### Question19: From 1 to 5, how satisfied are you with the company's performance evaluation and feedback process ?
    satisfaction_ratio = 0.6
    barcharts_ratio = 1 - satisfaction_ratio
    satisfaction_col, barcharts_col = st.columns([satisfaction_ratio, barcharts_ratio])

    st.markdown("""
        <style>
        .chart-container {
            padding-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q19ValuesCount, q19MedianScore = score_distribution(filtered_data, 26)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q19ValuesCount.values})

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on Company's Performance Evaluation and Feedback Process</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q19MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Satisfaction Level', color_discrete_map={
                'Very Dissatisfied': '#440154',  # Dark purple
                'Dissatisfied': '#3b528b',  # Dark blue
                'Neutral': '#21918c',  # Cyan
                'Satisfied': '#5ec962',  # Light green
                'Very Satisfied': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="overall_rating_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_options = ['Select a satisfaction level', 'Very Dissatisfied', 'Dissatisfied', 'Neutral',
                                'Satisfied', 'Very Satisfied']
        satisfaction_dropdown1 = st.selectbox('', satisfaction_options,
                                              key='satisfaction_dropdown1')

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown1, 26)

        location_summary1, role_summary1, function_summary1 = prepare_summaries(satisfaction_filtered_data1)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role1 = px.bar(role_summary1, y='Role', x='Count', orientation='h')
        fig_role1.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role1.update_traces(marker_color='#336699', text=role_summary1['Count'], textposition='outside')
        fig_role1.update_yaxes(showticklabels=True, title='')
        fig_role1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role1, use_container_width=True, key="roles_bar_chart1")

        fig_function1 = px.bar(function_summary1, y='Function', x='Count', orientation='h')
        fig_function1.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function1.update_traces(marker_color='#336699', text=function_summary1['Count'], textposition='outside')
        fig_function1.update_yaxes(showticklabels=True, title='')
        fig_function1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function1, use_container_width=True, key="functions_bar_chart1")

    ### Question20: Which reason(s) drive that score ?

    # Display the reasons for performance evaluation and feedback process satisfaction
    st.markdown('<h1 style="font-size:17px;font-family:Arial;color:#333333;">The Reasons for Ratings on Performance Evaluation and Feedback Process</h1>', unsafe_allow_html=True)

    # Define custom stopwords for the word clouds
    performance_stopwords = ["performance", "evaluation", "feedback", "process", "talent", "employees", "company", "help", "need", "everyone", "makes"]

    # Run this code in a Streamlit app
    if __name__ == "__main__":
        st.markdown("<h1 style='text-align: center; font-size: 24px; font-weight: normal;'>Word Cloud Visualization</h1>", unsafe_allow_html=True)
        generate_wordclouds(filtered_data, 26, 27, performance_stopwords)

    st.write('For detailed reason analysis/sentiment dashboard, please check out the [link](https://gucciouy5ardhonqumm6p4.streamlit.app)')


    ### Question21: From 1 to 5, how comfortable do you feel discussing your career goals and development with your manager? 
    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Uncomfortable', 'Uncomfortable', 'Hesitant', 'Comfortable', 'Very Comfortable']
        q21ValuesCount, q21MedianScore = score_distribution(filtered_data, 28)

        ratings_df = pd.DataFrame({'Comfort Level': categories, 'Percentage': q21ValuesCount.values})

        #Define the order of the categories
        comfort_order = ['Very Comfortable', 'Comfortable', 'Hesitant', 'Uncomfortable', 'Very Uncomfortable']

        # Convert 'Comfort Level' to a categorical variable with the specified order
        ratings_df['Comfort Level'] = pd.Categorical(ratings_df['Comfort Level'], categories=comfort_order, ordered=True)

        # Sort the DataFrame by 'Comfort Level'
        ratings_df.sort_values('Comfort Level', inplace=True)

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Comfort Level in Discussing Career Goals         and Development with Manager</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median comfort score is                   {q21MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Comfort Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Comfort Level', color_discrete_map={
                'Very Uncomfortable': '#440154',  # Dark purple
                'Uncomfortable': '#3b528b',  # Dark blue
                'Hesitant': '#21918c',  # Cyan
                'Comfortable': '#5ec962',  # Light green
                'Very Comfortable': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="overall_rating_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        comfort_options = ['Select a comfort level', 'Very Uncomfortable', 'Uncomfortable', 'Hesitant',
                           'Comfortable', 'Very Comfortable']
        comfort_dropdown1 = st.selectbox('', comfort_options,
                                         key='comfort_dropdown1')

        comfort_filtered_data1 = filter_by_comfort(filtered_data, comfort_dropdown1, 28)

        location_summary1, role_summary1, function_summary1 = prepare_summaries(comfort_filtered_data1)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role1 = px.bar(role_summary1, y='Role', x='Count', orientation='h')
        fig_role1.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role1.update_traces(marker_color='#336699', text=role_summary1['Count'], textposition='outside')
        fig_role1.update_yaxes(showticklabels=True, title='')
        fig_role1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role1, use_container_width=True, key="roles_bar_chart1")

        fig_function1 = px.bar(function_summary1, y='Function', x='Count', orientation='h')
        fig_function1.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function1.update_traces(marker_color='#336699', text=function_summary1['Count'], textposition='outside')
        fig_function1.update_yaxes(showticklabels=True, title='')
        fig_function1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function1, use_container_width=True, key="functions_bar_chart1")

    ### Question22: Which reason(s) drive that score ?
    st.markdown(
    """
    <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
    Reasons that drive scores: 1 - Very Uncomfortable / 2 - Uncomfortable / 3 - Hesitant 
    </h2>
    """,
    unsafe_allow_html=True
    )
    
    q29_data = pd.DataFrame({'negative_reasons': filtered_data.iloc[:, 29]})
    q29_data = q29_data.explode('negative_reasons')
    q29_data.dropna(inplace=True)

    # Count the occurrences of each negative reason
    negative_reason_recruiting_counts = q29_data['negative_reasons'].value_counts().reset_index()
    negative_reason_recruiting_counts.columns = ['negative_reasons', 'count']

    # Calculate percentage
    negative_reason_recruiting_counts['percentage'] = negative_reason_recruiting_counts['count'] / len(
        filtered_data) * 100

    fig1 = px.bar(negative_reason_recruiting_counts, y='negative_reasons', x='percentage', text='percentage',
                  color='negative_reasons', color_discrete_sequence=['#3b528b'], orientation='h')

    fig1.update_traces(hovertemplate='<b>Reason:</b> %{y}<br><b>Count:</b> %{text}')

    # Set the y-axis title
    fig1.update_yaxes(title_text='Reasons for Discomfort')

    # Remove the legend
    fig1.update_layout(showlegend=False)

    # Show the chart
    st.plotly_chart(fig1, use_container_width=False)


    ### Question23: Are you able to identify and tag your skills within your HRIS?
    q23_data_available_count = (filtered_data.iloc[:, 30] == 'Yes').sum()
    q23_data_available_pct = q23_data_available_count / len(filtered_data) * 100
    
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Identify and tag your skills within the HRIS
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                {q23_data_available_pct:.2f}% of the respondents, {q23_data_available_count} employee(s), are able to identify and tag their skills within the HRIS.</li>
                </div>
                """,
        unsafe_allow_html=True
    )


############ SECTION 3 ENDS ############


############ SECTION 4 STARTS ############      
if dashboard == 'Section 4: Learning':
    filtered_data = apply_filters(data, st.session_state['selected_role'], st.session_state['selected_function'],
                                  st.session_state['selected_location'])

    q24_data_available_count = (data.iloc[:, 31])
    q24_data_available_pct = q24_data_available_count.sum() / len(data) * 100
    q24ValuesCount, q24MedianScore = score_distribution(data, 31)

    q25_data = pd.DataFrame({'format': data.iloc[:, 32]})
    q25_data['format'] = q25_data['format'].str.rstrip(';').str.split(';')
    q25_data = q25_data.explode('format')
    q25_data.dropna(inplace=True)

    formats = q25_data['format'].value_counts().reset_index()
    formats.columns = ['Format', 'Count']
    most_chosen_format = formats.iloc[0]['Format']

    q26_data_available_count = (data.iloc[:, 33] == 'Yes').sum()
    q26_data_available_pct = q26_data_available_count / len(data) * 100

    q27_data_available_count = (data.iloc[:, 34] == 'Yes').sum()
    q27_data_available_pct = q27_data_available_count / len(data) * 100

    st.markdown(
        f"""
            <style>
            .top-bar {{
                background-color: #f0f2f6;  /* Light grey background */
                text-align: left;
                display: flex;
                justify-content: flex-start;
                align-items: center;
                height: auto;
                font-weight: normal;
                font-size: 17px;
                padding: 10px 20px;
                color: #333333;
                display: block;
                width: 100%;
                box-sizing: border-box;
            }}
            .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
            }}
            </style>
            <div class="top-bar">
                    This survey section is answered by all <b>{len(q24_data_available_count)}</b> survey participants<ul>
                    <li>The median satisfaction rating on the current learning management system is {q24MedianScore}.</li>
                <li> The most preferred learning format is through {most_chosen_format}.</li>
                <li>{q26_data_available_pct:.2f}% of the respondents, {q26_data_available_count} employee(s), participated in training or development programs provided by HR.</li>
                <li>{q27_data_available_pct:.2f}% of the respondents, {q27_data_available_count} employee(s), received recommendations on training.</li>
                </ul>
                </div>
                        """,
        unsafe_allow_html=True
    )

    # A text container for filtering instructions
    st.markdown(
        f"""
        <div class="text-container" style="font-style: italic;">
        Filter the data by selecting tags from the sidebar. The charts below will be updated to reflect the&nbsp;
        <strong>{len(filtered_data)}</strong>&nbsp;filtered respondents.
        </div>
        """,
        unsafe_allow_html=True
    )

    satisfaction_ratio = 0.6
    barcharts_ratio = 1 - satisfaction_ratio
    satisfaction_col, barcharts_col = st.columns([satisfaction_ratio, barcharts_ratio])

    st.markdown("""
        <style>
        .chart-container {
            padding-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    ### Question24: From 1 to 5, how satisfied are you with your current learning management system ?
    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q24ValuesCount, q24MedianScore = score_distribution(filtered_data, 31)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q24ValuesCount.values})

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on Current Learning Management System</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q24MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Satisfaction Level', color_discrete_map={
                'Very Dissatisfied': '#440154',  # Dark purple
                'Dissatisfied': '#3b528b',  # Dark blue
                'Neutral': '#21918c',  # Cyan
                'Satisfied': '#5ec962',  # Light green
                'Very Satisfied': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="overall_rating_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_options = ['Select a satisfaction level', 'Very Dissatisfied', 'Dissatisfied', 'Neutral',
                                'Satisfied', 'Very Satisfied']
        satisfaction_dropdown1 = st.selectbox('', satisfaction_options,
                                              key='satisfaction_dropdown1')

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown1, 31)

        location_summary1, role_summary1, function_summary1 = prepare_summaries(satisfaction_filtered_data1)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role1 = px.bar(role_summary1, y='Role', x='Count', orientation='h')
        fig_role1.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role1.update_traces(marker_color='#336699', text=role_summary1['Count'], textposition='outside')
        fig_role1.update_yaxes(showticklabels=True, title='')
        fig_role1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role1, use_container_width=True, key="roles_bar_chart1")

        fig_function1 = px.bar(function_summary1, y='Function', x='Count', orientation='h')
        fig_function1.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function1.update_traces(marker_color='#336699', text=function_summary1['Count'], textposition='outside')
        fig_function1.update_yaxes(showticklabels=True, title='')
        fig_function1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function1, use_container_width=True, key="functions_bar_chart1")

    ### Question25: What are the learning format that you prefer ?
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Preferred Learning Format
        </h2>
        """,
        unsafe_allow_html=True
    )

    # Create a DataFrame with the learning format data
    q25_data = pd.DataFrame({'learning_format': filtered_data.iloc[:, 32]})
    q25_data['learning_format'] = q25_data['learning_format'].str.rstrip(';')
    q25_data.dropna(inplace=True)

    # Count the occurrences of each learning format
    learning_format_counts = q25_data['learning_format'].value_counts().reset_index()
    learning_format_counts.columns = ['learning_format', 'count']

    # Calculate percentage
    learning_format_counts['percentage'] = learning_format_counts['count'] / learning_format_counts['count'].sum() * 100
    
    # Define the preferred order of learning formats
    preferred_order = ['E-Learning', 'On site', 'Micro-Learning', 'Coaching']

    # Ensure the DataFrame respects the preferred order
    learning_format_counts['learning_format'] = pd.Categorical(
        learning_format_counts['learning_format'],
        categories=preferred_order,
        ordered=True
    )
    learning_format_counts.sort_values('learning_format', inplace=True)

    # Create a horizontal bar chart
    fig25 = px.bar(learning_format_counts, x='percentage', y='learning_format', text='percentage',
                   orientation='h',
                   color='learning_format', color_discrete_map={
            'E-Learning': '#440154',  # Dark purple
            'On site': '#3b528b',  # Dark blue
            'Micro-Learning': '#21918c',  # Cyan
            'Coaching': '#5ec962',  # Light green
        })

    # Remove legend and axes titles
    fig25.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                        height=300, margin=dict(l=20, r=20, t=30, b=20))

    # Format text on bars
    fig25.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    fig25.update_xaxes(range=[0, max(learning_format_counts['percentage']) * 1.1])

    # Improve layout aesthetics
    fig25.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

    # Show the chart
    st.plotly_chart(fig25, use_container_width=False)

    ### Question26: Have you participated in any training or development programs provided by HR?
    q26_data_available_count = (filtered_data.iloc[:, 33] == 'Yes').sum()
    q26_data_available_pct = q26_data_available_count / len(filtered_data) * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Participation in any Training or Development Programs Provided by HR
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                {q26_data_available_pct:.2f}% of the respondents, {q26_data_available_count} employee(s), participated in training or development programs provided by HR.</li>
                </div>
                """,
        unsafe_allow_html=True
    )



    ### Question27: Have you received any recommendations on training (either by the HR team or directly on your Learning    System) ?
    q27_data_available_count = (filtered_data.iloc[:, 34] == 'Yes').sum()
    q27_data_available_pct = q27_data_available_count / len(filtered_data) * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Recommendations on Training (either by the HR team or directly on Learning System)
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                {q27_data_available_pct:.2f}% of the respondents, {q27_data_available_count} employee(s), received recommendations on training.</li>
                </div>
                """,
        unsafe_allow_html=True
    )



    ### Question28: What could be improved or what kind of format is missing today ?
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Improvement or Missing Format 
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    #Extract key phrases from the text
    learning_stopwords = ["this","about", "of", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "help", "need", "everyone", "makes", "improved", "improvement", "missing", "format", "today", "no", "and","should","more", "training"]

    improvement_and_missing = filtered_data.iloc[:, 35]
    improvement_and_missing = improvement_and_missing.dropna()

    #generate text for simple word cloud
    improvement_and_missing_text = ' '.join(improvement_and_missing.astype(str))

    #generate word cloud
    improvement_and_missing_cloud = WordCloud(width=800, height=400, background_color='white', stopwords=learning_stopwords).generate(improvement_and_missing_text)

    # Display the word cloud using Streamlit
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    st.image(improvement_and_missing_cloud.to_array(), use_column_width=True)

    st.write('For detailed reason analysis/sentiment dashboard, please check out the [link](https://gucciouy5ardhonqumm6p4.streamlit.app)')

############ SECTION 4 ENDS ############


############ SECTION 5 STARTS ############
if dashboard == 'Section 5: Compensation':
    filtered_data = apply_filters(data, st.session_state['selected_role'], st.session_state['selected_function'],
                                  st.session_state['selected_location'])
    
    q29_data_available_count = (data.iloc[:, 36] == 'Yes').sum()
    
    q30_data_available_count = (data.iloc[:, 37] == 'Yes').sum()
    q30_data_available_pct = q30_data_available_count / q29_data_available_count * 100
    
    q32_data = pd.DataFrame({'compensation_manage': data.iloc[:, 39]})
    q32_data['compensation_manage'] = q32_data['compensation_manage'].str.rstrip(';')
    q32_data.dropna(inplace=True)

    compensation_manage_counts = q32_data['compensation_manage'].value_counts().reset_index()
    compensation_manage_counts.columns = ['compensation_manage', 'count']
    most_compensation_manage = compensation_manage_counts.iloc[0]['compensation_manage']
    
    q33ValuesCount, q33MedianScore = score_distribution(data, 40)
    
    q36_data_available_count = (data.iloc[:, 43] == 'Yes').sum()
    q36_data_available_pct = q36_data_available_count / q29_data_available_count * 100
    
    q37_data_available_count = (data.iloc[:, 44] == 'Yes').sum()
    q37_data_available_pct = q37_data_available_count / q29_data_available_count * 100

    q38ValuesCount, q38MedianScore = score_distribution(data, 45)
    
    q39_data = pd.DataFrame({'bonus_manage': data.iloc[:, 46]})
    q39_data['bonus_manage'] = q39_data['bonus_manage'].str.rstrip(';')
    q39_data.dropna(inplace=True)

    # Count the occurrences of each compensation format
    bonus_manage_counts = q39_data['bonus_manage'].value_counts().reset_index()
    bonus_manage_counts.columns = ['bonus_manage', 'count']
    most_bonus_manage = bonus_manage_counts.iloc[0]['bonus_manage']
    
    q40_data_available_count = (data.iloc[:, 47] == 'Yes').sum()
    q40_data_available_pct = q40_data_available_count / q29_data_available_count * 10


    # Summary of all outputs in the bar container
    st.markdown(
    f"""
    <style>
    .top-bar {{
        background-color: #f0f2f6;  /* Light grey background */
        text-align: left;
        display: flex;
        justify-content: flex-start;
        align-items: center;
        height: auto;
        font-weight: normal;
        font-size: 17px;
        padding: 10px 20px;
        color: #333333;
        display: block;
        width: 100%;
        box-sizing: border-box;
    }}
    .top-bar ul, .top-bar li {{
        font-size: 17px;
        padding-left: 20px;
        margin: 0;
    }}
    </style>
    <div class="top-bar">
            This survey section is answered by <b>{q29_data_available_count}</b> employee(s), who participated in the   compensation campaign:
        <ul>
        <li>{q30_data_available_pct:.2f}% of the respondents, {q30_data_available_count} employee(s), think that the data available in the compensation form enables him/her to make a fair decision regarding a promotion, a bonus or a raise.</li>
        <li>The most common way to manage/launch the compensation campaigns is: {most_compensation_manage}.</li>
        <li>The median satisfaction rating on the compensation campaign is: {q33MedianScore}.</li>
        <li>{q36_data_available_pct:.2f}% of the respondents, {q36_data_available_count} employee(s), have retroactivity on salary payments.</li>
        <li>{q37_data_available_pct:.2f}% of the respondents, {q37_data_available_count} employee(s), participated in variable pay/bonus campaign.</li>
        <li>The median satisfaction rating on the variable pay/bonus campaign is: {q38MedianScore}.</li>
        <li>The most common way to manage/launch bonus/variable pay campaigns is: {most_bonus_manage}.</li>
        <li>{q40_data_available_pct:.2f}% of the respondents, {q40_data_available_count} employee(s), have different dates for the variable pay campaign compared to the compensation campaign.</li>
        </ul>
        </div>
                """,
        unsafe_allow_html=True
    )
    
    
    
      # A text container for filtering instructions
    st.markdown(
        f"""
        <div class="text-container" style="font-style: italic;">
        Filter the data by selecting tags from the sidebar. The charts below will be updated to reflect the&nbsp;
        <strong>{len(filtered_data)}</strong>&nbsp;filtered respondents.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    ### Qustion29: Do you participate in the Compensation Campaign ?
    q29_data_available_count = (filtered_data.iloc[:, 36] == 'Yes').sum()
    q29_data_available_pct = q29_data_available_count / len(filtered_data) * 100
    
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Compensation Campaign Participation
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                {q29_data_available_pct:.2f}% of the respondents, {q29_data_available_count} employee(s), participated in the   compensation campaign.</li>
                </div>
                """,
        unsafe_allow_html=True
    )


 
    ### Qustion30: Do you think that the data available in the Compensation form enables you to make a fair decision regarding a promotion, a bonus or a raise ? (e.g : compa-ratio, variation between years, historical data on salary and bonus, …) 
    q30_data_available_count = (filtered_data.iloc[:, 37] == 'Yes').sum()
    q30_data_available_pct = q30_data_available_count / q29_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Data availability in the Compensation Form
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who participate the Compensation Campaign, {q30_data_available_pct:.2f}% of the respondents, {q30_data_available_count} employee(s), think that the data available in the Compensation form enables him/her to make a fair decision regarding a promotion, a bonus or a raise.</li>
                </div>
                """,
        unsafe_allow_html=True
    )
    


    ### Qustion31: What data is missing according to you ?
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Missing Data in Compensation Campaign
        </h2>
        """,
        unsafe_allow_html=True
    )

    #stopwords for data missing for compensation
    compensation_stopwords = ["compensation", "miss", "missing", "this","about", "of", ",", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "help", "need", "everyone", "makes", "improved", "improvement", "format", "today", "no", "and","should","more", "training", "data", "according", "you"]
    
    data_missing = filtered_data.iloc[:, 38]
    data_missing = data_missing.dropna()

    #generate all text for word cloud for data missing
    data_missing_text = ' '.join(data_missing.astype(str))
    
    #generate simple word cloud for data missing
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=compensation_stopwords).generate(data_missing_text)

    # Display the word cloud using Streamlit
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    st.image(wordcloud.to_array(), use_column_width=True)

    st.write('For detailed reason analysis/sentiment dashboard, please check out the [link](https://gucciouy5ardhonqumm6p4.streamlit.app)')

    ### Question32: Do you manage/launch your compensation campaigns nationally or in another way?
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Compensation Campaigns Management/Launch
        </h2>
        """,
        unsafe_allow_html=True
    )

    # Create a DataFrame with the compensation format data
    q32_data = pd.DataFrame({'compensation_manage': filtered_data.iloc[:, 39]})
    q32_data['compensation_manage'] = q32_data['compensation_manage'].str.rstrip(';')
    q32_data.dropna(inplace=True)

    # Count the occurrences of each compensation format
    compensation_manage_counts = q32_data['compensation_manage'].value_counts().reset_index()
    compensation_manage_counts.columns = ['compensation_manage', 'count']

    # Calculate percentage
    compensation_manage_counts['percentage'] = compensation_manage_counts['count'] / compensation_manage_counts[
        'count'].sum() * 100

    # Create a horizontal bar chart
    fig32 = px.bar(compensation_manage_counts, x='percentage', y='compensation_manage', text='percentage',
                   orientation='h',
                   color='compensation_manage', color_discrete_map={
            'National Campaign': '#440154',  # Dark purple
            'International Campaign': '#3b528b',  # Dark blue
            'Regional Campaign': '#21918c',  # Cyan
        })

    # Remove legend and axes titles
    fig32.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                        height=300, margin=dict(l=20, r=20, t=30, b=20))

    # Format text on bars
    fig32.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    fig32.update_xaxes(range=[0, max(compensation_manage_counts['percentage']) * 1.1])

    # Improve layout aesthetics
    fig32.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

    # Show the chart
    st.plotly_chart(fig32, use_container_width=False)

    ### Question33: How would you rate the overall satisfaction regarding the compensation campaign ?
    satisfaction_ratio = 0.6
    barcharts_ratio = 1 - satisfaction_ratio
    satisfaction_col, barcharts_col = st.columns([satisfaction_ratio, barcharts_ratio])

    st.markdown("""
        <style>
        .chart-container {
            padding-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q33ValuesCount, q33MedianScore = score_distribution(filtered_data, 40)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q33ValuesCount.values})

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on Compensation Campaign</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q33MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Satisfaction Level', color_discrete_map={
                'Very Dissatisfied': '#440154',  # Dark purple
                'Dissatisfied': '#3b528b',  # Dark blue
                'Neutral': '#21918c',  # Cyan
                'Satisfied': '#5ec962',  # Light green
                'Very Satisfied': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="overall_rating_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_options = ['Select a satisfaction level', 'Very Dissatisfied', 'Dissatisfied', 'Neutral',
                                'Satisfied', 'Very Satisfied']
        satisfaction_dropdown1 = st.selectbox('', satisfaction_options,
                                              key='satisfaction_dropdown1')

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown1, 40)

        location_summary1, role_summary1, function_summary1 = prepare_summaries(satisfaction_filtered_data1)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role1 = px.bar(role_summary1, y='Role', x='Count', orientation='h')
        fig_role1.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role1.update_traces(marker_color='#336699', text=role_summary1['Count'], textposition='outside')
        fig_role1.update_yaxes(showticklabels=True, title='')
        fig_role1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role1, use_container_width=True, key="roles_bar_chart1")

        fig_function1 = px.bar(function_summary1, y='Function', x='Count', orientation='h')
        fig_function1.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function1.update_traces(marker_color='#336699', text=function_summary1['Count'], textposition='outside')
        fig_function1.update_yaxes(showticklabels=True, title='')
        fig_function1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function1, use_container_width=True, key="functions_bar_chart1")

    ### Question36: Do you have retroactivity on salary payments ? (e.g. New salary announced in March but payed from January)
    q36_data_available_count = (filtered_data.iloc[:, 43] == 'Yes').sum()
    q36_data_available_pct = q36_data_available_count / q29_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Retroactivity on Salary Payments
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who participate the Compensation Campaign, {q36_data_available_pct:.2f}% of the respondents, {q36_data_available_count} employee(s), have retroactivity on salary payments.</li>
                </div>
                """,
        unsafe_allow_html=True
    )
    


    ### Question37: Do you participate in the variable pay/bonus campaign ?
    q37_data_available_count = (filtered_data.iloc[:, 44] == 'Yes').sum()
    q37_data_available_pct = q37_data_available_count / q29_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Participation in Variable Pay/Bonus Campaign 
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who participate the Compensation Campaign, {q37_data_available_pct:.2f}% of the respondents, {q37_data_available_count} employee(s), participated in variable pay/bonus campaign.</li>
                </div>
                """,
        unsafe_allow_html=True
    )
    


    ### Question38: How would you rate the overall satisfaction regarding the Variable Pay/Bonus campaign  ?
    satisfaction_ratio = 0.6
    barcharts_ratio = 1 - satisfaction_ratio
    satisfaction_col, barcharts_col = st.columns([satisfaction_ratio, barcharts_ratio])

    st.markdown("""
        <style>
        .chart-container {
            padding-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q38ValuesCount, q38MedianScore = score_distribution(filtered_data, 45)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q38ValuesCount.values})

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on Variable Pay/Bonus Campaign </h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q38MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Satisfaction Level', color_discrete_map={
                'Very Dissatisfied': '#440154',  # Dark purple
                'Dissatisfied': '#3b528b',  # Dark blue
                'Neutral': '#21918c',  # Cyan
                'Satisfied': '#5ec962',  # Light green
                'Very Satisfied': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="overall_rating_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_options = ['Select a satisfaction level', 'Very Dissatisfied', 'Dissatisfied', 'Neutral',
                                'Satisfied', 'Very Satisfied']
        satisfaction_dropdown38 = st.selectbox('', satisfaction_options,
                                               key='satisfaction_dropdown38')

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown38, 45)

        location_summary1, role_summary1, function_summary1 = prepare_summaries(satisfaction_filtered_data1)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role1 = px.bar(role_summary1, y='Role', x='Count', orientation='h')
        fig_role1.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role1.update_traces(marker_color='#336699', text=role_summary1['Count'], textposition='outside')
        fig_role1.update_yaxes(showticklabels=True, title='')
        fig_role1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role1, use_container_width=True, key="roles_bar_chart1")

        fig_function1 = px.bar(function_summary1, y='Function', x='Count', orientation='h')
        fig_function1.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function1.update_traces(marker_color='#336699', text=function_summary1['Count'], textposition='outside')
        fig_function1.update_yaxes(showticklabels=True, title='')
        fig_function1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function1, use_container_width=True, key="functions_bar_chart1")

    ### Question39: Do you manage/launch your bonus/variable pay campaigns nationally or in another way?
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
       Bonus/Variable Pay Campaigns Management/Launch
        </h2>
        """,
        unsafe_allow_html=True
    )

    # Create a DataFrame with the compensation format data
    q39_data = pd.DataFrame({'bonus_manage': filtered_data.iloc[:, 46]})
    q39_data['bonus_manage'] = q39_data['bonus_manage'].str.rstrip(';')
    q39_data.dropna(inplace=True)

    # Count the occurrences of each compensation format
    bonus_manage_counts = q39_data['bonus_manage'].value_counts().reset_index()
    bonus_manage_counts.columns = ['bonus_manage', 'count']

    # Calculate percentage
    bonus_manage_counts['percentage'] = bonus_manage_counts['count'] / bonus_manage_counts['count'].sum() * 100

    # Create a horizontal bar chart
    fig39 = px.bar(bonus_manage_counts, x='percentage', y='bonus_manage', text='percentage',
                   orientation='h',
                   color='bonus_manage', color_discrete_map={
            'National Campaign': '#440154',  # Dark purple
            'International Campaign': '#3b528b',  # Dark blue
            'Regional Campaign': '#21918c',  # Cyan
        })

    # Remove legend and axes titles
    fig39.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                        height=300, margin=dict(l=20, r=20, t=30, b=20))

    # Format text on bars
    fig39.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    fig39.update_xaxes(range=[0, max(bonus_manage_counts['percentage']) * 1.1])

    # Improve layout aesthetics
    fig39.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

    # Show the chart
    st.plotly_chart(fig39, use_container_width=False)

    ### Question40: Are the dates of your Variable Pay campaign different from the one for the Compensation Campaign ?
    q40_data_available_count = (filtered_data.iloc[:, 47] == 'Yes').sum()
    q40_data_available_pct = q40_data_available_count / q29_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Variable Pay Campaign Dates Different from Compensation Campaign Dates
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who participate the Compensation Campaign, {q40_data_available_pct:.2f}% of the respondents, {q40_data_available_count} employee(s), have different dates for the Variable Pay Campaign compared to the Compensation Campaign.</li>
                </div>
                """,
        unsafe_allow_html=True
    )


############ SECTION 5 ENDS ############


############ SECTION 6 STARTS ############
if dashboard == 'Section 6: Payroll':
    filtered_data = apply_filters(data, st.session_state['selected_role'], st.session_state['selected_function'],
                                  st.session_state['selected_location'])
    
    q41_data_available_count = (data.iloc[:, 48] == 'Yes').sum()
    
    q42ValuesCount, q42MedianScore = score_distribution(data, 49)
    
    q43_data_available_count = (data.iloc[:, 50] == 'Internal').sum()
    q43_data_available_pct = q43_data_available_count / q41_data_available_count * 100

    q44_data_available_count = (data.iloc[:, 51] == 'Yes').sum()
    q44_data_available_pct = q44_data_available_count / q41_data_available_count * 100

    q45_data_available_count = (data.iloc[:, 52] == 'Autonomous').sum()
    q45_data_available_pct = q45_data_available_count / q41_data_available_count * 100
    
    q47_data_available_count = (data.iloc[:, 54] == 'Yes').sum()
    q47_data_available_pct = q47_data_available_count / q41_data_available_count * 100

    q48_data_available_count = (data.iloc[:, 55] == 'Yes').sum()
    q48_data_available_pct = q48_data_available_count / q47_data_available_count * 100

    q49_data_available_count = (data.iloc[:, 56] == 'Yes').sum()
    q49_data_available_pct = q49_data_available_count / q41_data_available_count * 100
    
    q50_data_available_count = (data.iloc[:, 57] == 'Yes').sum()
    q50_data_available_pct = q50_data_available_count / q41_data_available_count * 100

    q51_data_available_count = (data.iloc[:, 58] == 'Yes').sum()
    q51_data_available_pct = q51_data_available_count / q41_data_available_count * 100


    # Summary of all outputs in the bar container
    st.markdown(
    f"""
    <style>
    .top-bar {{
        background-color: #f0f2f6;  /* Light grey background */
        text-align: left;
        display: flex;
        justify-content: flex-start;
        align-items: center;
        height: auto;
        font-weight: normal;
        font-size: 17px;
        padding: 10px 20px;
        color: #333333;
        display: block;
        width: 100%;
        box-sizing: border-box;
    }}
    .top-bar ul, .top-bar li {{
        font-size: 17px;
        padding-left: 20px;
        margin: 0;
    }}
    </style>
    <div class="top-bar">
            This survey section is answered by <b>{q41_data_available_count}</b> employee(s), who are part of the payroll team:
        <ul>
        <li>The median satisfaction rating on current payroll system is: {q42MedianScore}.</li>
        <li>{q43_data_available_pct:.2f}% of the respondents, {q43_data_available_count} employee(s), realize their payroll activities internally and others realize that it is outsourced.</li>
        <li>{q44_data_available_pct:.2f}% of the respondents, {q44_data_available_count} employee(s), answer that their system covers legal updates.</li>
        <li>{q45_data_available_pct:.2f}% of the respondents, {q45_data_available_count} employee(s), answer that they are autonomous and others rely on outside firms for updates.</li>
        <li>{q47_data_available_pct:.2f}% of the respondents, {q47_data_available_count} employee(s), answer that they have a global platform for consolidating all employees' country data.</li>
        <li>{q48_data_available_pct:.2f}% of the respondents, {q48_data_available_count} employee(s), answer that this platform automatically generate KPIs relating to the payroll (M/F headcount, salaries paid, contributions paid, etc.).</li>
        <li>{q49_data_available_pct:.2f}% of the respondents, {q49_data_available_count} employee(s), answer that mass entries be made in the tool.</li>
        <li>{q50_data_available_pct:.2f}% of the respondents, {q50_data_available_count} employee(s), answer that the payroll system is connected with time management system.</li>
        <li>{q51_data_available_pct:.2f}% of the respondents, {q51_data_available_count} employee(s), answer that the payroll system is connected with a CORE HR/Administrative solution.</li>
        </ul>
        </div>
                """,
        unsafe_allow_html=True
    )

    # A text container for filtering instructions
    st.markdown(
        f"""
        <div class="text-container" style="font-style: italic;">
        Filter the data by selecting tags from the sidebar. The charts below will be updated to reflect the&nbsp;
        <strong>{len(filtered_data)}</strong>&nbsp;filtered respondents.
        </div>
        """,
        unsafe_allow_html=True
    )

    ### Question41: Are you part of the payroll team ?
    q41_data_available_count = (filtered_data.iloc[:, 48] == 'Yes').sum()
    q41_data_available_pct = q41_data_available_count / len(filtered_data) * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Payroll Team
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                {q41_data_available_pct:.2f}% of the respondents, {q41_data_available_count} employee(s), are part of the payroll team.</li>
                </div>
                """,
        unsafe_allow_html=True
    )



    ### Question42: How satisfied are you with your current payroll system ?
    satisfaction_ratio = 0.6
    barcharts_ratio = 1 - satisfaction_ratio
    satisfaction_col, barcharts_col = st.columns([satisfaction_ratio, barcharts_ratio])

    st.markdown("""
        <style>
        .chart-container {
            padding-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q42ValuesCount, q42MedianScore = score_distribution(filtered_data, 49)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q42ValuesCount.values})

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on Current Payroll System</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q42MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Satisfaction Level', color_discrete_map={
                'Very Dissatisfied': '#440154',  # Dark purple
                'Dissatisfied': '#3b528b',  # Dark blue
                'Neutral': '#21918c',  # Cyan
                'Satisfied': '#5ec962',  # Light green
                'Very Satisfied': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="overall_rating_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_options = ['Select a satisfaction level', 'Very Dissatisfied', 'Dissatisfied', 'Neutral',
                                'Satisfied', 'Very Satisfied']
        satisfaction_dropdown38 = st.selectbox('', satisfaction_options,
                                               key='satisfaction_dropdown38')

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown38, 49)

        location_summary1, role_summary1, function_summary1 = prepare_summaries(satisfaction_filtered_data1)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role1 = px.bar(role_summary1, y='Role', x='Count', orientation='h')
        fig_role1.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role1.update_traces(marker_color='#336699', text=role_summary1['Count'], textposition='outside')
        fig_role1.update_yaxes(showticklabels=True, title='')
        fig_role1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role1, use_container_width=True, key="roles_bar_chart1")

        fig_function1 = px.bar(function_summary1, y='Function', x='Count', orientation='h')
        fig_function1.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function1.update_traces(marker_color='#336699', text=function_summary1['Count'], textposition='outside')
        fig_function1.update_yaxes(showticklabels=True, title='')
        fig_function1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function1, use_container_width=True, key="functions_bar_chart1")

    ### Question43: Do you realize your payroll activities internally or is it outsourced ?
    q43_data_available_count = (filtered_data.iloc[:, 50] == 'Internal').sum()
    q43_data_available_pct = q43_data_available_count / q41_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Payroll Activities
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the payroll team, {q43_data_available_pct:.2f}% of the respondents, {q43_data_available_count} employee(s), realize their payroll activities internally and others realize that it is outsourced.</li>
                </div>
                """,
        unsafe_allow_html=True
    )


    ### Question44: Does your system cover legal updates ?
    q44_data_available_count = (filtered_data.iloc[:, 51] == 'Yes').sum()
    q44_data_available_pct = q44_data_available_count / q41_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Legal Updates
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the payroll team, {q44_data_available_pct:.2f}% of the respondents, {q44_data_available_count} employee(s), answer that their system covers legal updates.</li>
                </div>
                """,
        unsafe_allow_html=True
    )

    ### Question45: Are you autonomous when it comes to updating simple data, or do you systematically rely on outside firms for updates?
    q45_data_available_count = (filtered_data.iloc[:, 52] == 'Autonomous').sum()
    q45_data_available_pct = q45_data_available_count / q41_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Autonomy When It Comes to Updating Simple Data
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the payroll team, {q45_data_available_pct:.2f}% of the respondents, {q45_data_available_count} employee(s), answer that they are autonomous and others rely on outside firms for updates.</li>
                </div>
                """,
        unsafe_allow_html=True
    )

    ### Question46: Can you share any specific features of your current system that you like/that made you choose it?
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Specific Features of the Current System that People like/that Made People Choose It
        </h2>
        """,
        unsafe_allow_html=True
    )

    #stopwords for specific features of the current system that you like/that made you choose it
    features_stopwords = ["payroll", "compensation", "miss", "missing", "this","about", "of", ",", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "system", "like", "choose", "help", "need", "everyone", "makes", "improved", "improvement", "format", "today", "no", "and","should","more", "training", "data", "according", "you"]
    
    specific_features = filtered_data.iloc[:, 53]

    #generate wordcloud since the repsonses are too few
    word_cloud = WordCloud(width=800, height=400, background_color='white', stopwords=features_stopwords).generate(' '.join(specific_features.dropna().astype(str)))

    # Display the word cloud using Streamlit
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    
    st.image(word_cloud.to_array(), use_column_width=True)
    
    #Generate more complex wordcloud if there are more repsonses
    specific_features = specific_features.dropna()
    
    st.markdown(
        "<h3 style='text-align: left; font-size: 20px; font-weight: normal;'>All specific features of the current system that people like/that made people choose it</h3>",
        unsafe_allow_html=True)
    st.write(specific_features)

    # Checkbox to decide whether to display the complete DataFrame
    if st.checkbox('Display complete specific features of the current system that people like/that made people choose it'):
        # Convert DataFrame to HTML and display it
        html = filtered_data.iloc[:,53].to_html(index=False)
        st.markdown(html, unsafe_allow_html=True)

    # Convert DataFrame to CSV and generate download link
    csv = filtered_data.iloc[:, 53].to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="specific_features.csv">Download specific_features CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.write('For detailed reason analysis/sentiment dashboard, please check out the [link](https://gucciouy5ardhonqumm6p4.streamlit.app)')


    ### Question47: If your payroll system is used in several countries, do you have a global platform for consolidating all your employees' country data?
    q47_data_available_count = (filtered_data.iloc[:, 54] == 'Yes').sum()
    q47_data_available_pct = q47_data_available_count / q41_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Global Platform for Multiple Countries
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the payroll team, {q47_data_available_pct:.2f}% of the respondents, {q47_data_available_count} employee(s), answer that they have a global platform for consolidating all employees' country data.</li>
                </div>
                """,
        unsafe_allow_html=True
    )


    ### Question48: If so, does this platform automatically generate KPIs relating to your payroll (M/F headcount, salaries paid, contributions paid, etc.)?
    q48_data_available_count = (filtered_data.iloc[:, 55] == 'Yes').sum()
    q48_data_available_pct = q48_data_available_count / q47_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Global Platform Function: Automatically Generate KPIs
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who have a global platform, {q48_data_available_pct:.2f}% of the respondents, {q48_data_available_count} employee(s), answer that this platform automatically generate KPIs relating to the payroll (M/F headcount, salaries paid, contributions paid, etc.).</li>
                </div>
                """,
        unsafe_allow_html=True
    )

    ### Question49: Can mass entries be made in the tool?
    q49_data_available_count = (filtered_data.iloc[:, 56] == 'Yes').sum()
    q49_data_available_pct = q49_data_available_count / q41_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Mass Entries
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the payroll team, {q49_data_available_pct:.2f}% of the respondents, {q49_data_available_count} employee(s), answer that that mass entries be made in the tool.</li>
                </div>
                """,
        unsafe_allow_html=True
    )


    ### Question50: Is your payroll connected with your time management system ?
    q50_data_available_count = (filtered_data.iloc[:, 57] == 'Yes').sum()
    q50_data_available_pct = q50_data_available_count / q41_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Payroll Connected with Time Management System
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the payroll team, {q50_data_available_pct:.2f}% of the respondents, {q50_data_available_count} employee(s), answer that the payroll system is connected with time management system.</li>
                </div>
                """,
        unsafe_allow_html=True
    )



    ### Question51: Is your payroll connected with a CORE HR/Administrative solution ?
    q51_data_available_count = (filtered_data.iloc[:, 58] == 'Yes').sum()
    q51_data_available_pct = q51_data_available_count / q41_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Payroll Connected with a CORE HR/Administrative Solution
        </h2>
        """,
        unsafe_allow_html=True
    )

    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the payroll team, {q51_data_available_pct:.2f}% of the respondents, {q51_data_available_count} employee(s), answer that the payroll system is connected with a CORE HR/Administrative solution.</li>
                </div>
                """,
        unsafe_allow_html=True
    )
############ SECTION 6 ENDS ############


############ SECTION 7 STARTS ############    
if dashboard == 'Section 7: Time Management':
    filtered_data = apply_filters(data, st.session_state['selected_role'], st.session_state['selected_function'],
                                  st.session_state['selected_location'])
    
    q52_data_available_count = (data.iloc[:, 59] == 'Yes').sum()
    
    q53_data_available_count = (data.iloc[:, 60] == 'Yes').sum()
    q53_data_available_pct = q53_data_available_count / q52_data_available_count * 100
    
    q54ValuesCount, q54MedianScore = score_distribution(data, 61)

    q55_data_available_count = (data.iloc[:, 62] == 'Yes').sum()
    q55_data_available_pct = q55_data_available_count / q52_data_available_count * 100
    
    q56_data_available_count = (data.iloc[:, 63] == 'Yes').sum()
    q56_data_available_pct = q56_data_available_count / q52_data_available_count * 100
    
    q57_data_available_count = (data.iloc[:, 64] == 'Yes').sum()
    q57_data_available_pct = q57_data_available_count / q52_data_available_count * 100
    
    q58_data_available_count = (data.iloc[:, 65] == 'Yes').sum()
    q58_data_available_pct = q58_data_available_count / q52_data_available_count * 100


    q60_data_available_count = (data.iloc[:, 67] == 'Yes').sum()
    q60_data_available_pct = q60_data_available_count / q52_data_available_count * 100

    q61_data_available_count = (data.iloc[:, 68] == 'Yes').sum()
    q61_data_available_pct = q61_data_available_count / q52_data_available_count * 100


    # Summary of all outputs in the bar container
    st.markdown(
    f"""
    <style>
    .top-bar {{
        background-color: #f0f2f6;  /* Light grey background */
        text-align: left;
        display: flex;
        justify-content: flex-start;
        align-items: center;
        height: auto;
        font-weight: normal;
        font-size: 17px;
        padding: 10px 20px;
        color: #333333;
        display: block;
        width: 100%;
        box-sizing: border-box;
    }}
    .top-bar ul, .top-bar li {{
        font-size: 17px;
        padding-left: 20px;
        margin: 0;
    }}
    </style>
    <div class="top-bar">
            This survey section is answered by <b>{q52_data_available_count}</b> employee(s), who are part of the time management team:
        <ul>
        <li>{q53_data_available_pct:.2f}% of the respondents,  {q53_data_available_count} employee(s),  answer that they currently have a time management system.</li>
        <li>The median satisfaction rating on time management system is: {q54MedianScore}.</li>
        <li>{q55_data_available_pct:.2f}% of the respondents,  {q55_data_available_count} employee(s),  answer that they have a self-service for their employees.</li>
        <li>{q56_data_available_pct:.2f}% of the respondents,  {q56_data_available_count} employee(s),  answer that the system allow employess to view their vacation counters (entitlement / taken / balance).</li>
        <li>{q57_data_available_pct:.2f}% of the respondents,  {q57_data_available_count} employee(s),  answer that the system cover all the shift scheduling functions needed.</li>
        <li>{q58_data_available_pct:.2f}% of the respondents,  {q58_data_available_count} employee(s),  answer that they have the capability to run all the reports needed.</li>
        <li>{q60_data_available_pct:.2f}% of the respondents,  {q60_data_available_count} employee(s),  answer that the system allows employees to take their own leave, with workflow validation by their manager or HR.</li>
        <li>{q61_data_available_pct:.2f}% of the respondents,  {q61_data_available_count} employee(s),  answer that the system automatically take retroactive items into account (e.g. application to April payroll of a salary increase with an effective date of January 1).</li>
        </ul>
        </div>
                """,
        unsafe_allow_html=True
    )

    # A text container for filtering instructions
    st.markdown(
        f"""
        <div class="text-container" style="font-style: italic;">
        Filter the data by selecting tags from the sidebar. The charts below will be updated to reflect the&nbsp;
        <strong>{len(filtered_data)}</strong>&nbsp;filtered respondents.
        </div>
        """,
        unsafe_allow_html=True
    )

    ### Question52: Are you part of the Time Management Team ?
    q52_data_available_count = (filtered_data.iloc[:, 59] == 'Yes').sum()
    q52_data_available_pct = q52_data_available_count / len(filtered_data) * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Time Management Team
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                {q52_data_available_pct:.2f}% of the respondents, {q52_data_available_count} employee(s), are part of the time management team.</li>
                </div>
                """,
        unsafe_allow_html=True
    )

    ### Question53: Do you currently have a time management system ?
    q53_data_available_count = (filtered_data.iloc[:, 60] == 'Yes').sum()
    q53_data_available_pct = q53_data_available_count / q52_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Time Management System
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the time management team, {q53_data_available_pct:.2f}% of the respondents,  {q53_data_available_count} employee(s),  answer that they currently have a time management system.</li>
                </div>
                """,
        unsafe_allow_html=True
    )

    ### Question54: How satisfied are you with your current time management system ?
    satisfaction_ratio = 0.6
    barcharts_ratio = 1 - satisfaction_ratio
    satisfaction_col, barcharts_col = st.columns([satisfaction_ratio, barcharts_ratio])

    st.markdown("""
        <style>
        .chart-container {
            padding-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    with satisfaction_col:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
        q54ValuesCount, q54MedianScore = score_distribution(filtered_data, 61)

        ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': q54ValuesCount.values})

        # Define the order of the categories
        satisfaction_order = ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']

        # Convert 'Satisfaction Level' to a categorical variable with the specified order
        ratings_df['Satisfaction Level'] = pd.Categorical(ratings_df['Satisfaction Level'], categories=satisfaction_order, ordered=True)

        # Sort the DataFrame by 'Satisfaction Level'
        ratings_df.sort_values('Satisfaction Level', inplace=True)

        # Display title and median score
        title_html = f"<h2 style='font-size: 17px; font-family: Arial; color: #333333;'>Rating on Current Time Management System</h2>"
        caption_html = f"<div style='font-size: 15px; font-family: Arial; color: #707070;'>The median satisfaction score is {q54MedianScore:.1f}</div>"
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(caption_html, unsafe_allow_html=True)

        # Create a horizontal bar chart with Plotly
        fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage', text='Percentage',
                     orientation='h',
                     color='Satisfaction Level', color_discrete_map={
                'Very Dissatisfied': '#440154',  # Dark purple
                'Dissatisfied': '#3b528b',  # Dark blue
                'Neutral': '#21918c',  # Cyan
                'Satisfied': '#5ec962',  # Light green
                'Very Satisfied': '#fde725'  # Bright yellow
            })

        # Remove legend and axes titles
        fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None, autosize=True,
                          height=300, margin=dict(l=20, r=20, t=30, b=20))

        # Format text on bars
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig.update_xaxes(range=[0, max(ratings_df['Percentage']) * 1.1])

        # Improve layout aesthetics
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig, use_container_width=True, key="overall_rating_bar_chart")
        st.markdown('</div>', unsafe_allow_html=True)

    with barcharts_col:
        satisfaction_options = ['Select a satisfaction level', 'Very Dissatisfied', 'Dissatisfied', 'Neutral',
                                'Satisfied', 'Very Satisfied']
        satisfaction_dropdown38 = st.selectbox('', satisfaction_options,
                                               key='satisfaction_dropdown38')

        satisfaction_filtered_data1 = filter_by_satisfaction(filtered_data, satisfaction_dropdown38, 61)

        location_summary1, role_summary1, function_summary1 = prepare_summaries(satisfaction_filtered_data1)
        left_margin = 150
        total_height = 310
        role_chart_height = total_height * 0.45
        function_chart_height = total_height * 0.55

        fig_role1 = px.bar(role_summary1, y='Role', x='Count', orientation='h')
        fig_role1.update_layout(title="by Role", margin=dict(l=left_margin, r=0, t=50, b=0),
                                height=role_chart_height, showlegend=False)
        fig_role1.update_traces(marker_color='#336699', text=role_summary1['Count'], textposition='outside')
        fig_role1.update_yaxes(showticklabels=True, title='')
        fig_role1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_role1, use_container_width=True, key="roles_bar_chart1")

        fig_function1 = px.bar(function_summary1, y='Function', x='Count', orientation='h')
        fig_function1.update_layout(title="by Function", margin=dict(l=left_margin, r=0, t=50, b=0),
                                    height=function_chart_height, showlegend=False)
        fig_function1.update_traces(marker_color='#336699', text=function_summary1['Count'], textposition='outside')
        fig_function1.update_yaxes(showticklabels=True, title='')
        fig_function1.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function1, use_container_width=True, key="functions_bar_chart1")

    ### Question55: Do you have a self-service for your employees ?
    q55_data_available_count = (filtered_data.iloc[:, 62] == 'Yes').sum()
    q55_data_available_pct = q55_data_available_count / q52_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Self-service for the Employees
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the time management team, {q55_data_available_pct:.2f}% of the respondents,  {q55_data_available_count} employee(s),  answer that they have a self-service for their employees.</li>
                </div>
                """,
        unsafe_allow_html=True
    )


    ### Question56: Does the system allow employees to view their vacation counters (entitlement / taken / balance)
    q56_data_available_count = (filtered_data.iloc[:, 63] == 'Yes').sum()
    q56_data_available_pct = q56_data_available_count / q52_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        System Function: View Vacation Counters
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the time management team, {q56_data_available_pct:.2f}% of the respondents,  {q56_data_available_count} employee(s),  answer that the system allow employess to view their vacation counters (entitlement / taken / balance).</li>
                </div>
                """,
        unsafe_allow_html=True
    )


    ### Question57: Does your system cover all the shift scheduling functions you need?
    q57_data_available_count = (filtered_data.iloc[:, 64] == 'Yes').sum()
    q57_data_available_pct = q57_data_available_count / q52_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        System Function: Cover the Shift Scheduling
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the time management team, {q57_data_available_pct:.2f}% of the respondents,  {q57_data_available_count} employee(s),  answer that the system cover all the shift scheduling functions needed.</li>
                </div>
                """,
        unsafe_allow_html=True
    )

    ### Question58: Do you have the capability to run all the report needed ?
    q58_data_available_count = (filtered_data.iloc[:, 65] == 'Yes').sum()
    q58_data_available_pct = q58_data_available_count / q52_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Capability: Run all the reports
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the time management team, {q58_data_available_pct:.2f}% of the respondents,  {q58_data_available_count} employee(s),  answer that they have the capability to run all the reports needed.</li>
                </div>
                """,
        unsafe_allow_html=True
    )

    ### Question59: According to you, what functionalities are missing from your current system ?
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Functionalities Missing from the Current System
        </h2>
        """,
        unsafe_allow_html=True
    )

    #stopwords for functionalities missing from the current system
    functionalities_stopwords = ["functionalities", "system", "payroll", "compensation", "miss", "missing", "this","about", "of", ",", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "system", "like", "choose", "help", "need", "everyone", "makes", "improved", "improvement", "format", "today", "no", "and","should","more", "training", "data", "according", "you"]
    
    functionalities_missing = filtered_data.iloc[:, 66]

    #generate wordcloud since the repsonses are too few
    word_cloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(functionalities_missing.dropna().astype(str)))

    # Display the word cloud using Streamlit
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    st.image(word_cloud.to_array(), use_column_width=True)
    
    #Generate more complex wordcloud if there are more repsonses
    #drop missing values first
    functionalities_missing = functionalities_missing.dropna()
    
    st.markdown(
        "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>All the functionalities missing from the current system:</h3>",
        unsafe_allow_html=True)
    st.write(functionalities_missing)

    # Checkbox to decide whether to display the complete DataFrame
    if st.checkbox('Display complete missing functionalities'):
        # Convert DataFrame to HTML and display it
        html = filtered_data.iloc[:, 66].to_html(index=False)
        st.markdown(html, unsafe_allow_html=True)

    # Convert DataFrame to CSV and generate download link
    csv = filtered_data.iloc[:, 66].to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="functionalities_missing.csv">Download functionalities_missing csv file</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.write('For detailed reason analysis/sentiment dashboard, please check out the [link](https://gucciouy5ardhonqumm6p4.streamlit.app)')


    ### Question60: Does the system allow employees to take their own leave, with workflow validation by their manager or HR?
    q60_data_available_count = (filtered_data.iloc[:, 67] == 'Yes').sum()
    q60_data_available_pct = q60_data_available_count / q52_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        System Function: Allow Employess to Take Their Own Leave
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the time management team, {q60_data_available_pct:.2f}% of the respondents,  {q60_data_available_count} employee(s),  answer that the system allows employees to take their own leave, with workflow validation by their manager or HR.</li>
                </div>
                """,
        unsafe_allow_html=True
    )


    ### Question61: Does your system automatically take retroactive items into account (e.g. application to April payroll of a salary increase with an effective date of January 1)?
    q61_data_available_count = (filtered_data.iloc[:, 68] == 'Yes').sum()
    q61_data_available_pct = q61_data_available_count / q52_data_available_count * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        System Function: Automatically Take Retroactive Items Into Account
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                Among the people who are part of the time management team, {q61_data_available_pct:.2f}% of the respondents,  {q61_data_available_count} employee(s),  answer that the system automatically take retroactive items into account (e.g. application to April payroll of a salary increase with an effective date of January 1).</li>
                </div>
                """,
        unsafe_allow_html=True
    )


############ SECTION 7 ENDS ############


############ SECTION 8 STARTS ############ 
if dashboard == 'Section 8: User Experience':
    filtered_data = apply_filters(data, st.session_state['selected_role'], st.session_state['selected_function'],
                                  st.session_state['selected_location'])
    
    q64_data_available_count = (data.iloc[:, 71] == 'Yes').sum()
    q64_data_available_pct = q64_data_available_count / len(data) * 100


    st.markdown(
    f"""
    <style>
    .top-bar {{
        background-color: #f0f2f6;  /* Light grey background */
        text-align: left;
        display: flex;
        justify-content: flex-start;
        align-items: center;
        height: auto;
        font-weight: normal;
        font-size: 17px;
        padding: 10px 20px;
        color: #333333;
        display: block;
        width: 100%;
        box-sizing: border-box;
    }}
    .top-bar ul, .top-bar li {{
        font-size: 17px;
        padding-left: 20px;
        margin: 0;
    }}
    </style>
    <div class="top-bar">
        This survey section is answered by all the <strong>{len(data)}</strong> survey participants:
        <ul>
            <li>{q64_data_available_pct:.2f}% of the respondents, {q64_data_available_count} employee(s), consider the time you spend on your HRIS to be time well spent.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
    )
    

    # A text container for filtering instructions
    st.markdown(
        f"""
        <div class="text-container" style="font-style: italic;">
        Filter the data by selecting tags from the sidebar. The charts below will be updated to reflect the&nbsp;
        <strong>{len(filtered_data)}</strong>&nbsp;filtered respondents.
        </div>
        """,
        unsafe_allow_html=True
    )

    ### Question62: In the context of your job, what are the most valuable activities your current HRIS enable you to do?
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        The Most Valuable Activities in the Current HRIS in the Context of the Job
        </h2>
        """,
        unsafe_allow_html=True
    )

    #stopwords for most valuable activities in the current HRIS
    HRIS_stopwords = ["I","and" ," and", "and ", " ;", "; ", " to", " for", "for ", "for", "to ", "my", "activities", "HRIS", "valuable", "system", "HR", "current", "functionalities", "system", "payroll", "compensation", "miss", "missing", "this","about", "of", ",", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "system", "like", "choose", "help", "need", "everyone", "makes", "improved", "improvement", "format", "today", "no", "and","should","more", "training", "data", "according", "you"]
    
    valuable_activities = filtered_data.iloc[:, 69]

    #drop missing values first
    valuable_activities = valuable_activities.dropna()

    #generate wordcloud since the repsonses are too few
    word_cloud_valuable = WordCloud(width=800, height=400, background_color='white', stopwords=HRIS_stopwords).generate(' '.join(valuable_activities.dropna().astype(str)))


    
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    
    st.image(word_cloud_valuable.to_array(), use_column_width=True)


    ### Question63: In the context of your job, what do your current HRIS fail to address?
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        What the Current HRIS Fail to Address in the Context of the Job
        </h2>
        """,
        unsafe_allow_html=True
    )
    #stopwords for most functions are missing in the current HRIS
    HRIS_stopwords2 = ["I", "my", "activities", "fail", "address", "missing", "HRIS", "valuable", "system", "HR", "current", "functionalities", "system", "payroll", "compensation", "miss", "missing", "this","about", "of", ",", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "system", "like", "choose", "help", "need", "everyone", "makes", "improved", "improvement", "format", "today", "no", "and","should","more", "training", "data", "according", "you"]

    functions_missing = filtered_data.iloc[:, 70]

    #generate wordcloud since the repsonses are too few
    word_cloud_functions = WordCloud(width=800, height=400, background_color='white', stopwords=HRIS_stopwords2).generate(' '.join(functions_missing.dropna().astype(str)))

    # Display the word cloud using Streamlit
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    st.image(word_cloud_functions.to_array(), use_column_width=True)

    #Generate more complex wordcloud if there are more repsonses

    #drop missing values first
    functions_missing = functions_missing.dropna()


    ### Question64: Do you consider the time you spend on your HRIS to be time well spent?
    q64_data_available_count = (filtered_data.iloc[:, 71] == 'Yes').sum()
    q64_data_available_pct = q64_data_available_count / len(filtered_data) * 100

    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        Time Spend on HRIS
        </h2>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
                <style>
                .top-bar {{
                    font-weight: normal;
                    font-size: 17px;
                    padding: 10px 20px;
                    color: #333333;
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                }}
                .top-bar ul, .top-bar li {{
                font-size: 17px;
                padding-left: 20px;
                margin: 0;
                }}
                </style>
                <div class="top-bar">
                {q64_data_available_pct:.2f}% of the respondents, {q64_data_available_count} employee(s), consider the time you spend on your HRIS to be time well spent.</li>
                </div>
                """,
        unsafe_allow_html=True
    )


    ### Question65: In 3 words, how would you describe your current user-experience with the HRIS ?
    st.markdown(
        """
        <h2 style='font-size: 17px; font-family: Arial; color: #333333;'>
        3 Words to Describe the Current User-Experience with the HRIS
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='text-align: center; font-size: 24px; font-weight: normal;'>Descriptions about User Experience with the Current HRIS</h1>", unsafe_allow_html=True)
    #get the data
    overall_experience = filtered_data.iloc[:, 72]

    #set the stopwords
    Overall_stopwords = [",", ";", " ;", "; ", "not very", "not", "very", " ; ", "I", "my", "activities", "fail", "address", "missing", "HRIS", "valuable", "system", "HR", "current", "functionalities", "system", "payroll", "compensation", "miss", "missing", "this","about", "of", ",", "to", "a", "what", "on", "could", "do", "we", "their", "the", "learning", "management", "system", "employees", "company", "system", "like", "choose", "help", "need", "everyone", "makes", "improved", "improvement", "format", "today", "no", "and","should","more", "training", "data", "according", "you"]

    # Function to extract n-grams from text
    def extract_ngrams(x, n):
        ngrams = []
        phrases = x.split(', ')
        for phrase in phrases:
            words = phrase.split(' ')
            ngrams.extend([' '.join(ng) for ng in nltk_ngrams(words, n)])
        return ngrams

    #drop missing values first
    overall_experience = overall_experience.dropna()

    # Concatenate all text data
    overall_text = ' '.join(overall_experience.astype(str))

    # Generate unigrams, bigrams, and trigrams
    unigrams_overall = extract_ngrams(overall_text, 1)
    bigrams_overall = extract_ngrams(overall_text, 2)
    trigrams_overall = extract_ngrams(overall_text, 3)

    # Count the frequency of each n-gram
    unigram_freq_overall = Counter(unigrams_overall)
    bigram_freq_overall = Counter(bigrams_overall)
    trigram_freq_overall = Counter(trigrams_overall)

    # Combine the frequencies
    combined_freq_overall = unigram_freq_overall + bigram_freq_overall + trigram_freq_overall

    # Generate the word cloud
    phrase_cloud_overall = WordCloud(width=800, height=400, background_color='white', stopwords = Overall_stopwords).generate_from_frequencies(combined_freq_overall)

    # Display the word cloud using Streamlit
    st.markdown(
            "<h3 style='text-align: center; font-size: 20px; font-weight: normal;'>Word Cloud</h3>",
            unsafe_allow_html=True)
    st.image(phrase_cloud_overall.to_array(), use_column_width=True)

    #check if the user wants to see the data
    if st.checkbox('Display complete description data in 3 words'):
        # Convert DataFrame to HTML and display it
        html = filtered_data.iloc[:, 72].to_html(index=False)
        st.markdown(html, unsafe_allow_html=True)

    # Convert DataFrame to CSV and generate download link
    csv = filtered_data.iloc[:, 72].to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="user_experience.csv">Download user_experience csv file</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.write('For detailed reason analysis/sentiment dashboard, please check out the [link](https://gucciouy5ardhonqumm6p4.streamlit.app)')

############ SECTION 8 ENDS ############