import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


def initialize_state():
    if 'previous_dashboard' not in st.session_state:
        st.session_state['previous_dashboard'] = None
    if 'selected_role' not in st.session_state:
        st.session_state['selected_role'] = []
    if 'selected_function' not in st.session_state:
        st.session_state['selected_function'] = []
    if 'selected_location' not in st.session_state:
        st.session_state['selected_location'] = []
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None


def reset_filters():
    st.session_state['selected_role'] = []
    st.session_state['selected_function'] = []
    st.session_state['selected_location'] = []


st.set_page_config(layout="wide")
initialize_state()


# Load and clean data
@st.cache_data
def load_data():
    file_name = 'https://github.com/001202ZHENG/V1_Chatbot_Streamlit/raw/main/data/Voice%20of%20Customer_Second%20data%20set.xlsx'

    data = pd.read_excel(file_name)
    data = data.rename(columns={
        'What is your role at the company ?': 'Role',
        'What function are you part of ?': 'Function',
        'Where are you located ?': 'Location'
    })
    return data


data = load_data()

# General Page Layout
st.markdown(
    '''
    <style>
        /* Remove Streamlit's default padding around the main content area */
        .main .block-container {
            padding-top: 0rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 2rem;
        }

        /* Adjust margins for headers to reduce white space */
        h1 {
            margin-top: 0.25rem;
            margin-bottom: 0.25rem;
        }
        h3 {
            margin-top: 0.25rem;
            margin-bottom: 0.25rem;
        }

        /* Additional global style overrides could be added here */
    </style>
    ''',
    unsafe_allow_html=True
)

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

selected_role = st.sidebar.multiselect('Select Role', options=data['Role'].unique(),
                                       default=st.session_state['selected_role'])
st.session_state['selected_role'] = selected_role

selected_function = st.sidebar.multiselect('Select Function', options=data['Function'].unique(),
                                           default=st.session_state['selected_function'])
st.session_state['selected_function'] = selected_function

selected_location = st.sidebar.multiselect('Select Location', options=data['Location'].unique(),
                                           default=st.session_state['selected_location'])
st.session_state['selected_location'] = selected_location


def apply_filters(data, roles, functions, locations):
    filtered = data
    if roles:
        filtered = filtered[filtered['Role'].isin(roles)]
    if functions:
        filtered = filtered[filtered['Function'].isin(functions)]
    if locations:
        filtered = filtered[filtered['Location'].isin(locations)]
    return filtered


# Header Function
def render_header(title, subtitle=None):
    # Define common styles
    style = """
    <style>
        h1 {
            text-align: center;
            color: #333333; /* Dark grey color */
            margin-bottom: 0; /* Smaller bottom margin to reduce space to subtitle */
        }
        h3 {
            text-align: center;
            color: #333333; /* Dark grey color */
            margin-top: 0; /* Smaller top margin to reduce space from title */
            margin-bottom: 0; /* Add a little space below the subtitle */
            font-weight: normal; /* Normal font weight for the subtitle */
        }
        .header {
            text-decoration: underline;
            font-size: 30px; /* Adjust size as needed */
        }
        .subheader {
            font-size: 20px; /* Adjust subtitle font size as needed */
        }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    st.markdown(f'<h1 class="header">{title}</h1>', unsafe_allow_html=True)

    if subtitle:
        st.markdown(f'<h3 class="subheader">{subtitle}</h3>', unsafe_allow_html=True)


# Use the function with both a title and a subtitle
if dashboard == 'General Survey Results':
    render_header("General Survey Results", "A summary of the respondent profiles and the general sentiment analysis")
elif dashboard == 'Section 1: Employee Experience':
    render_header("Employee Experience", "General HR Services Evaluation")
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

filtered_data = apply_filters(data, selected_role, selected_function, selected_location)

############ GENERAL DASHBOARD STARTS ############
if dashboard == "General Survey Results":
    # Data preparation for display of survey summary
    def prepare_summaries(data):
        # Create a dictionary to map continents to proxy countries
        continent_to_country_code = {
            'Asia': 'KAZ',
            'Oceania': 'AUS',
            'North America': 'CAN',
            'South America': 'BRA',
            'Europe': 'DEU',
            'Africa': 'TCD'
        }
        country_code_to_continent = {v: k for k, v in continent_to_country_code.items()}
        location_summary = data['Location'].value_counts().rename_axis('Continent').reset_index(name='Count')
        location_summary['Country_Code'] = location_summary['Continent'].map(continent_to_country_code)
        location_summary['Label'] = location_summary['Continent'].apply(
            lambda x: f"{x}: {location_summary.loc[location_summary['Continent'] == x, 'Count'].iloc[0]}")

        role_summary = data['Role'].value_counts().rename_axis('Role').reset_index(name='Count')
        function_summary = data['Function'].value_counts().rename_axis('Function').reset_index(name='Count')
        return location_summary, role_summary, function_summary


    location_summary, role_summary, function_summary = prepare_summaries(filtered_data)

    st.markdown(
        """
        <style>
        .top-bar {
            background-color: #f0f2f6;  /* Light grey background */
            text-align: center;  /* Center-align text */
            display: flex;
            justify-content: center;
            align-items: center;
            height: auto;  /* Let the height be dynamic */
        }
        </style>
        """, unsafe_allow_html=True
    )

    # The top bar with centered and styled text
    st.markdown(
        f'<div class="top-bar" style="font-weight: bold; font-size: 20px; padding: 10px 0; color: #333333;">{len(filtered_data)} total survey respondents</div>',
        unsafe_allow_html=True
    )

    map_ratio = 0.55
    barcharts_ratio = 1 - map_ratio
    mark_color = '#336699'

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
            margin=dict(l=left_margin, r=0, t=50, b=0),  # Set the left margin
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
            margin=dict(l=left_margin, r=0, t=50, b=0),  # Set the left margin, the same as for fig_role
            height=function_chart_height,
            showlegend=False
        )
        fig_function.update_traces(marker_color=mark_color, text=function_summary['Count'], textposition='outside')
        fig_function.update_yaxes(showticklabels=True, title='')
        fig_function.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function, use_container_width=True)


############ GENERAL DASHBOARD ENDS ############

##### THIS SECTION FOR SATISFACTION SCORES START ####
# MARIAS SCORE DISTRIBUTION FUNCTION
def score_distribution(column_index, title):
    # Extract the data series based on the column index
    data_series = data.iloc[:, column_index]

    # Calculate the percentage of each response
    value_counts = data_series.value_counts(normalize=True).sort_index() * 100
    categories = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
    ratings_df = pd.DataFrame({'Satisfaction Level': categories, 'Percentage': value_counts})

    # Calculate the median score
    scores = np.repeat([1, 2, 3, 4, 5], data_series.value_counts().sort_index())
    median_score = np.median(scores)

    # Create a horizontal bar chart with Plotly
    fig = px.bar(ratings_df, y='Satisfaction Level', x='Percentage',
                 title=title, text='Percentage',
                 orientation='h',
                 color='Satisfaction Level', color_discrete_map={
            'Very Dissatisfied': px.colors.diverging.RdYlGn[0],  # Red
            'Dissatisfied': px.colors.diverging.RdYlGn[1],  # Less red
            'Neutral': px.colors.diverging.RdYlGn[6],  # Neutral (middle)
            'Satisfied': px.colors.diverging.RdYlGn[8],  # Less green
            'Very Satisfied': px.colors.diverging.RdYlGn[10]  # Green
        })

    # Remove legend and axes titles
    fig.update_layout(showlegend=False, xaxis_visible=False, xaxis_title=None, yaxis_title=None)

    # Format text on bars
    fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')

    # Improve layout aesthetics
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

    # Use Streamlit to display the Plotly chart
    st.plotly_chart(fig)

    # Display median score graphically
    star_full = '★'
    star_empty = '☆'
    median_visual = star_full * int(median_score) + star_empty * (5 - int(median_score))
    st.markdown(f"**Median Score:** {median_visual}")


# Function to plot satisfaction proportions -- OLD
def plot_satisfaction_proportions(data_series, title):
    # Calculate satisfaction proportions
    score_counts = data_series.value_counts().sort_index().astype(int)
    total_satisfied = score_counts.get(4, 0) + score_counts.get(5, 0)
    total_dissatisfied = score_counts.get(1, 0) + score_counts.get(2, 0) + score_counts.get(3, 0)

    # Calculate proportions
    dissatisfied_proportions = [score_counts.get(i, 0) / total_dissatisfied if total_dissatisfied > 0 else 0 for i in
                                range(1, 4)]
    satisfied_proportions = [score_counts.get(i, 0) / total_satisfied if total_satisfied > 0 else 0 for i in
                             range(4, 6)]

    # Create the plotly figure for stacked bar chart
    fig = go.Figure()

    # Add 'Dissatisfied' segments
    cumulative_size = 0
    colors_dissatisfied = sns.color_palette("Blues_d", n_colors=3)
    for i, prop in enumerate(dissatisfied_proportions):
        fig.add_trace(go.Bar(
            x=[prop],
            y=['Dissatisfied'],
            orientation='h',
            name=f'{i + 1}',
            marker=dict(
                color=f'rgb({colors_dissatisfied[i][0] * 255},{colors_dissatisfied[i][1] * 255},{colors_dissatisfied[i][2] * 255})'),
            base=cumulative_size
        ))
        cumulative_size += prop

    # Add 'Satisfied' segments
    cumulative_size = 0
    colors_satisfied = sns.color_palette("Greens_d", n_colors=2)
    for i, prop in enumerate(satisfied_proportions):
        fig.add_trace(go.Bar(
            x=[prop],
            y=['Satisfied'],
            orientation='h',
            name=f'{i + 4}',
            marker=dict(
                color=f'rgb({colors_satisfied[i][0] * 255},{colors_satisfied[i][1] * 255},{colors_satisfied[i][2] * 255})'),
            base=cumulative_size
        ))
        cumulative_size += prop

    # Update layout and display in Streamlit
    fig.update_layout(
        title=title,
        barmode='stack',
        annotations=[
            dict(x=1.05, y=0, text=f'Total: {total_dissatisfied}', showarrow=False),
            dict(x=1.05, y=1, text=f'Total: {total_satisfied}', showarrow=False)
        ]
    )
    fig.update_xaxes(title_text="", visible=True, showticklabels=False)
    fig.update_yaxes(title_text="")

    st.plotly_chart(fig)  # Display the plot in Streamlit


##### THIS SECTION FOR SATISFACTION SCORES ENDS ####


##### THIS SECTION FOR SIDEBAR AND SENTIMENT ANALYSIS CHARTS START START START START ####
# Function to create Streamlit sentiment dashboard
# Initialize VADER sentiment analyzer
# Make sure the VADER lexicon is downloaded
nltk.download('vader_lexicon')
sentiment_analyzer = SentimentIntensityAnalyzer()


# Function for sentiment analysis dashboard
def sentiment_dashboard(data_series, title):
    # Sidebar for control
    st.sidebar.markdown("### Filter Options")
    show_wordcloud = st.sidebar.checkbox("Show Word Cloud", value=True)
    filter_negative = st.sidebar.checkbox("Show Negative Comments", value=False)
    filter_positive = st.sidebar.checkbox("Show Positive Comments", value=False)

    # Initialize sentiment results and comment lists
    sentiment_results = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    negative_comments = []
    positive_comments = []

    # Analyze sentiment and collect results
    for sentence in data_series.dropna():
        sentiment_scores = sentiment_analyzer.polarity_scores(sentence)
        compound_score = sentiment_scores['compound']

        if compound_score <= -0.05:
            sentiment_results['Negative'] += 1
            negative_comments.append((sentence, compound_score))
        elif compound_score >= 0.05:
            sentiment_results['Positive'] += 1
            positive_comments.append((sentence, compound_score))
        else:
            sentiment_results['Neutral'] += 1

    # Display word cloud
    if show_wordcloud:
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate(' '.join(data_series.dropna()))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)  # Display word cloud in Streamlit

    # Display top negative and positive comments
    if filter_negative:
        st.markdown("### Top 5 Negative Comments")
        for comment, score in sorted(negative_comments, key=lambda x: x[1], reverse=True)[:5]:
            st.write(f"{comment} (Score: {score:.4f})")

    if filter_positive:
        st.markdown("### Top 5 Positive Comments")
        for comment, score in sorted(positive_comments, key=lambda x: x[1], reverse=True)[:5]:
            st.write(f"{comment} (Score: {score:.4f})")

    # Create stacked bar chart for sentiment distribution
    total = sum(sentiment_results.values())
    proportions = {k: v / total for k, v in sentiment_results.items()}

    fig = go.Figure()
    cumulative_size = 0
    for sentiment, proportion in proportions.items():
        color = 'lightgreen' if sentiment == 'Positive' else 'lightcoral' if sentiment == 'Negative' else 'lightgrey'
        fig.add_trace(go.Bar(x=[proportion], y=['Sentiment'], orientation='h', name=sentiment, base=cumulative_size,
                             marker=dict(color=color)))
        cumulative_size += proportion

    # Update layout and display chart in Streamlit
    fig.update_layout(
        title="Sentiment Distribution",
        barmode='stack',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )

    st.plotly_chart(fig)  # Display the stacked bar chart


##### THIS SECTION FOR SIDEBAR AND SENTIMENT ANALYSIS CHARTS END END END END ####


############ SECTION 1 STARTS ############

if dashboard == "Section 1: Employee Experience":
    filtered_data = data[
        (data['Role'].isin(selected_role)) &
        (data['Function'].isin(selected_function)) &
        (data['Location'].isin(selected_location))
        ]

    st.markdown(
        """
        <style>
        .top-bar {
            background-color: #f0f2f6;  /* Light grey background */
            text-align: center;  /* Center-align text */
            display: flex;
            justify-content: center;
            align-items: center;
            height: auto;  /* Let the height be dynamic */
        }
        </style>
        """, unsafe_allow_html=True
    )

    # The top bar with centered and styled text
    st.markdown(
        f'<div class="top-bar" style="font-weight: bold; font-size: 20px; padding: 10px 0; color: #333333;">{len(filtered_data)} survey respondents</div>',
        unsafe_allow_html=True
    )

    # q10 how you find the HR services responsive
    q10_responsiveness_count = (filtered_data.iloc[:, 15] == 'Yes').sum()
    q10_responsiveness_pct = q10_responsiveness_count / len(filtered_data) * 100
    # st.write("Responsiveness to Inquiries and Concerns")
    st.write("%.2f" % q10_responsiveness_pct, "% of people, which are", q10_responsiveness_count,
             "person(s), find the HR department responsive to their inquiries and concerns.")

    # Question 6: From 1 to 5, how satisfied are you with the overall HR services and support provided by the company?
    score_distribution(11, "Overall Rating on HR Services and Support")

    # Question 8: From 1 to 5, how satisfied are you with the communication channels used to relay important HR information to employees?
    # score_distribution(13, "Rating on HR Communication Channels")

    # Extract the satisfaction scores column
    q6_data = pd.DataFrame({'satisfaction_score': filtered_data[
        "From 1 to 5, how satisfied are you with the overall HR services and support provided by the company?\xa0"]})

    # Count the occurrences of each score
    score_counts = q6_data['satisfaction_score'].value_counts().reset_index()
    score_counts.columns = ['satisfaction_score', 'count']

    # Create a dictionary to map scores to categories
    score_to_category = {
        1: 'Very Dissatisfied',
        2: 'Dissatisfied',
        3: 'Neutral',
        4: 'Satisfied',
        5: 'Very Satisfied'
    }

    # Create a new column 'satisfaction_category' by mapping the 'satisfaction_score' column to the categories
    score_counts['satisfaction_category'] = score_counts['satisfaction_score'].map(score_to_category)

    # Calculate percentage
    score_counts['percentage'] = score_counts['count'] / score_counts['count'].sum() * 100

    # Sort score_counts by 'satisfaction_score' in descending order
    score_counts = score_counts.sort_values('satisfaction_score', ascending=False)

    # Create a horizontal bar chart
    fig1 = px.bar(score_counts, x='percentage', y='satisfaction_category', text='count', orientation='h',
                  color='satisfaction_category',
                  color_discrete_map={
                      'Very Dissatisfied': '#C9190B',
                      'Dissatisfied': '#EC7A08',
                      'Neutral': '#F0AB00',
                      'Satisfied': '#519DE9',
                      'Very Satisfied': '#004B95'
                  })

    # Calculate median score
    median_score = q6_data['satisfaction_score'].median()

    # Determine the color based on the median score
    if median_score < 2:
        color = 'red'
    elif median_score < 3:
        color = 'orange'
    elif median_score < 4:
        color = 'yellow'
    else:
        color = 'green'

    # Display the median score in a text box
    st.markdown(f'<p style="color: {color};">Median Satisfaction Score: {median_score:.2f}</p>', unsafe_allow_html=True)

    st.plotly_chart(fig1, use_container_width=True)

    import plotly.graph_objects as go

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

    # Merge the two dataset on function
    # Merge datasets by matching HR_Process and Improve_Area
    q4_q5_count = pd.merge(q4_count, q5_count, left_on='HR_Process', right_on='Improve_Area', how='outer')

    # Drop unnecessary columns
    q4_q5_count.drop(['Improve_Area'], axis=1, inplace=True)

    q4_q5_count.rename(
        columns={'HR_Process': 'HR Function', 'Count_x': 'HR_Process_Interacted', 'Count_y': 'Improvement_Areas'},
        inplace=True)

    q4_q5_count.sort_values('HR_Process_Interacted', ascending=False, inplace=True)

    # Reshape data into tidy format
    df_tidy = q4_q5_count.melt(id_vars='HR Function', var_name='Type', value_name='Count')

    # Create grouped bar chart
    fig2 = go.Figure(data=[
        go.Bar(
            name='HR Process Interacted',
            x=df_tidy[df_tidy['Type'] == 'HR_Process_Interacted']['HR Function'],
            y=df_tidy[df_tidy['Type'] == 'HR_Process_Interacted']['Count'],
            marker_color='#3C3D99'
        ),
        go.Bar(
            name='Improvement Areas',
            x=df_tidy[df_tidy['Type'] == 'Improvement_Areas']['HR Function'],
            y=df_tidy[df_tidy['Type'] == 'Improvement_Areas']['Count'],
            marker_color='#F0AB00'
        )
    ])

    fig2.update_layout(
        title='Grouped Bar Chart by HR Function',
        xaxis_title='HR Function',
        yaxis_title='Count',
        barmode='group'
    )

    st.plotly_chart(fig2, use_container_width=True)

    # q7 how access to HR services
    q7_data = pd.DataFrame({'device': filtered_data["How do you access HR Information ?"]})
    q7_data['device'] = q7_data['device'].str.rstrip(';').str.split(';')
    q7_data = q7_data.explode('device')
    q7_data.dropna(inplace=True)

    # Count the occurrences of each device
    device_counts = q7_data['device'].value_counts().reset_index()
    device_counts.columns = ['device', 'count']

    # Calculate percentage
    device_counts['percentage'] = device_counts['count'] / len(filtered_data) * 100

    # Create a horizontal bar chart
    fig3 = px.bar(device_counts, x='percentage', y='device', text='count', orientation='h', color='device')

    # show the chart
    st.plotly_chart(fig3, use_container_width=True)

    # q10 how you find the HR services responsive
    q10_responsiveness_count = (filtered_data.iloc[:, 15] == 'Yes').sum()
    q10_responsiveness_pct = q10_responsiveness_count / len(filtered_data) * 100

    st.write("Responsiveness to Inquiries and Concerns")

    st.write("%.2f" % q10_responsiveness_pct, "% of people, which are", q10_responsiveness_count,
             "person(s), find the HR department responsive to their inquiries and concerns.")

    # q8 satisfaction about the channels
    q8_data = pd.DataFrame({'satisfaction_channel': filtered_data.iloc[:, 13]})

    # Count the occurrences of each satisfaction channel
    channel_counts = q8_data['satisfaction_channel'].value_counts().reset_index()
    channel_counts.columns = ['satisfaction_channel', 'count']

    # Calculate percentage
    channel_counts['percentage'] = channel_counts['count'] / channel_counts['count'].sum() * 100

    # Create a new column 'satisfaction_category' by mapping the 'satisfaction_channel' column to the categories
    channel_counts['satisfaction_category'] = channel_counts['satisfaction_channel'].map(score_to_category)

    # Calculate percentage
    channel_counts['percentage'] = channel_counts['count'] / channel_counts['count'].sum() * 100

    # Sort channel_counts by 'satisfaction_channel' in descending order
    channel_counts = channel_counts.sort_values('satisfaction_channel', ascending=False)

    # Create a horizontal bar chart
    fig4 = px.bar(channel_counts, x='percentage', y='satisfaction_category', text='count', orientation='h',
                  color='satisfaction_category',
                  color_discrete_map={
                      'Very Dissatisfied': '#C9190B',
                      'Dissatisfied': '#EC7A08',
                      'Neutral': '#F0AB00',
                      'Satisfied': '#519DE9',
                      'Very Satisfied': '#004B95'
                  })

    # Calculate median score
    median_score1 = q8_data['satisfaction_channel'].median()

    # Determine the color based on the median score
    if median_score1 < 2:
        color = 'red'
    elif median_score1 < 3:
        color = 'orange'
    elif median_score1 < 4:
        color = 'yellow'
    else:
        color = 'green'

    # Display the median score in a text box
    st.markdown(f'<p style="color: {color};">Median Channel Satisfaction Score: {median_score1:.2f}</p>',
                unsafe_allow_html=True)

    st.plotly_chart(fig4, use_container_width=True)

############ SECTION 1 ENDS ############

############ SECTION 2 STARTS ############
if dashboard == 'Section 2: Recruiting & Onboarding':
    plot_satisfaction_proportions(data['From 1 to 5, how would you rate the onboarding process ?'],
                                  'Proportion of Onboarding Process Satisfaction Scores')
    sentiment_dashboard(data['Which reason(s) drive that score ?'], 'Sentiment Analysis Dashboard')

    filtered_data = data[
        (data['Role'].isin(selected_role)) &
        (data['Function'].isin(selected_function)) &
        (data['Location'].isin(selected_location))
        ]

    column_16 = filtered_data.iloc[:, 16]

    # Count the number of 'Less than a year' and 'More than a year' responses
    less_than_a_year_count = (column_16 == 'Less than a year').sum()
    more_than_a_year_count = (column_16 == 'More than a year').sum()

    # Get the total number of rows in the DataFrame
    total_rows = filtered_data.shape[0]

    # Display the sentence
    st.markdown(
        f"The dashboard section above is based on {less_than_a_year_count} out of {total_rows} respondents, who have been less than a year.")

    # Extract the satisfaction scores column
    q9_data = pd.DataFrame({'satisfaction_score': filtered_data["How would rate the recruiting process ?"]})

    # Count the occurrences of each score
    q9_score_counts = q9_data['satisfaction_score'].value_counts().reset_index()
    q9_score_counts.columns = ['satisfaction_score', 'count']

    # Create a dictionary to map scores to categories
    score_to_category = {
        1: 'Very Dissatisfied',
        2: 'Dissatisfied',
        3: 'Neutral',
        4: 'Satisfied',
        5: 'Very Satisfied'
    }

    # Create a new column 'satisfaction_category' by mapping the 'satisfaction_score' column to the categories
    q9_score_counts['satisfaction_category'] = q9_score_counts['satisfaction_score'].map(score_to_category)

    # Calculate percentage
    q9_score_counts['percentage'] = q9_score_counts['count'] / q9_score_counts['count'].sum() * 100

    # Sort score_counts by 'satisfaction_score' in descending order
    q9_score_counts = q9_score_counts.sort_values('satisfaction_score', ascending=False)

    # Create a horizontal bar chart
    fig5 = px.bar(q9_score_counts, x='percentage', y='satisfaction_category', text='count', orientation='h',
                  color='satisfaction_category',
                  color_discrete_map={
                      'Very Dissatisfied': '#C9190B',
                      'Dissatisfied': '#EC7A08',
                      'Neutral': '#F0AB00',
                      'Satisfied': '#519DE9',
                      'Very Satisfied': '#004B95'
                  })

    # Calculate median score
    q9_median_score = q9_data['satisfaction_score'].median()

    # Determine the color based on the median score
    if q9_median_score < 2:
        color = 'red'
    elif q9_median_score < 3:
        color = 'orange'
    elif q9_median_score < 4:
        color = 'yellow'
    else:
        color = 'green'

    # Display the median score in a text box
    st.markdown(f'<p style="color: {color};">Median Satisfaction Score: {q9_median_score:.2f}</p>',
                unsafe_allow_html=True)

    st.plotly_chart(fig5, use_container_width=True)

    # negative reasons for recruiting process
    q18_data = pd.DataFrame({'negative_reasons': filtered_data.iloc[:, 18]})
    q18_data['negative_reasons'] = q18_data['negative_reasons'].str.rstrip(';').str.split(';')
    q18_data = q18_data.explode('negative_reasons')
    q18_data.dropna(inplace=True)

    # Count the occurrences of each negative reason
    negative_reason_recruiting_counts = q18_data['negative_reasons'].value_counts().reset_index()
    negative_reason_recruiting_counts.columns = ['negative_reasons', 'count']

    # Calculate percentage
    negative_reason_recruiting_counts['percentage'] = negative_reason_recruiting_counts['count'] / len(
        filtered_data) * 100

    # Create a vertical bar chart
    fig6 = px.bar(negative_reason_recruiting_counts, x='negative_reasons', y='percentage', text='count',
                  color='negative_reasons', color_discrete_sequence=['#FFA500'])

    # make a tree map on the negative reasons
    fig10 = px.treemap(negative_reason_recruiting_counts, path=['negative_reasons'], values='count', color='count',
                       color_continuous_scale='RdBu')

    # Show the chart
    st.plotly_chart(fig6, use_container_width=True)
    st.plotly_chart(fig10, use_container_width=True)

    # positive reasons for recruiting process
    q19_data = pd.DataFrame({'positive_reasons': filtered_data.iloc[:, 19]})
    q19_data['positive_reasons'] = q19_data['positive_reasons'].str.rstrip(';').str.split(';')
    q19_data = q19_data.explode('positive_reasons')
    q19_data.dropna(inplace=True)

    # Count the occurrences of each positive reason
    positive_reason_recruiting_counts = q19_data['positive_reasons'].value_counts().reset_index()
    positive_reason_recruiting_counts.columns = ['positive_reasons', 'count']

    # Calculate percentage
    positive_reason_recruiting_counts['percentage'] = positive_reason_recruiting_counts['count'] / len(
        filtered_data) * 100

    # Create a vertical bar chart
    fig7 = px.bar(positive_reason_recruiting_counts, x='positive_reasons', y='percentage', text='count',
                  color='positive_reasons', color_discrete_sequence=['#519DE9'])

    # make a tree map on the positive reasons
    fig9 = px.treemap(positive_reason_recruiting_counts, path=['positive_reasons'], values='count', color='count',
                      color_continuous_scale='RdBu')

    # Show the chart
    st.plotly_chart(fig7, use_container_width=True)
    st.plotly_chart(fig9, use_container_width=True)

    # recruting process that took longest time and require improvement
    q20_data = pd.DataFrame({'recruting process that required improvement': filtered_data.iloc[:, 20]})

    q20_data['recruting process that required improvement'] = q20_data[
        'recruting process that required improvement'].str.rstrip(';').str.split(';')
    q20_data = q20_data.explode('recruting process that required improvement')
    q20_data.dropna(inplace=True)

    # Count the occurrences of each aspect that required improvement
    aspect_recruiting_counts = q20_data['recruting process that required improvement'].value_counts().reset_index()
    aspect_recruiting_counts.columns = ['recruting process that required improvement', 'count']

    # Calculate percentage
    aspect_recruiting_counts['percentage'] = aspect_recruiting_counts['count'] / len(filtered_data) * 100

    # Create a vertical bar chart
    fig8 = px.bar(aspect_recruiting_counts, x='recruting process that required improvement', y='percentage',
                  text='count', color='recruting process that required improvement',
                  color_discrete_sequence=['#FF7F7F'])

    # make a tree map on the aspect that required improvement
    fig11 = px.treemap(aspect_recruiting_counts, path=['recruting process that required improvement'], values='count',
                       color='count', color_continuous_scale='RdBu')

    # Show the chart
    st.plotly_chart(fig8, use_container_width=True)
    st.plotly_chart(fig11, use_container_width=True)

    # score analysis on onboarding process
    q21_data = pd.DataFrame({'onboarding_score': filtered_data.iloc[:, 21]})

    # Count the occurrences of each score
    q21_score_counts = q21_data['onboarding_score'].value_counts().reset_index()
    q21_score_counts.columns = ['onboarding_score', 'count']

    # Create a new column 'onboarding_category' by mapping the 'onboarding_score' column to the categories
    q21_score_counts['onboarding_satisfactory_category'] = q21_score_counts['onboarding_score'].map(score_to_category)

    # Calculate percentage
    q21_score_counts['percentage'] = q21_score_counts['count'] / q21_score_counts['count'].sum() * 100

    # Sort score_counts by 'onboarding_score' in descending order
    q21_score_counts = q21_score_counts.sort_values('onboarding_score', ascending=False)

    # Create a horizontal bar chart
    fig12 = px.bar(q21_score_counts, x='percentage', y='onboarding_satisfactory_category', text='count',
                   orientation='h', color='onboarding_satisfactory_category',
                   color_discrete_map={
                       'Very Dissatisfied': '#C9190B',
                       'Dissatisfied': '#EC7A08',
                       'Neutral': '#F0AB00',
                       'Satisfied': '#519DE9',
                       'Very Satisfied': '#004B95'
                   })

    # Calculate median score
    q21_median_score = q21_data['onboarding_score'].median()

    # Determine the color based on the median score
    if q21_median_score < 2:
        color = 'red'
    elif q21_median_score < 3:
        color = 'orange'
    elif q21_median_score < 4:
        color = 'yellow'
    else:
        color = 'green'

    # Display the median score in a text box
    st.markdown(f'<p style="color: {color};">Median Onboarding Score: {q21_median_score:.2f}</p>',
                unsafe_allow_html=True)

    st.plotly_chart(fig12, use_container_width=True)

    # negative reasons for onboarding process
    q22_data = pd.DataFrame({'negative_reasons': filtered_data.iloc[:, 22]})
    q22_data['negative_reasons'] = q22_data['negative_reasons'].str.rstrip(';').str.split(';')
    q22_data = q22_data.explode('negative_reasons')
    q22_data.dropna(inplace=True)

    # Count the occurrences of each negative reason
    negative_reason_recruiting_counts = q22_data['negative_reasons'].value_counts().reset_index()
    negative_reason_recruiting_counts.columns = ['negative_reasons', 'count']

    # Calculate percentage
    negative_reason_recruiting_counts['percentage'] = negative_reason_recruiting_counts['count'] / len(
        filtered_data) * 100

    # Create a vertical bar chart
    fig13 = px.bar(negative_reason_recruiting_counts, x='negative_reasons', y='percentage', text='count',
                   color='negative_reasons', color_discrete_sequence=['#FFA500'])

    # make a tree map on the negative reasons
    fig14 = px.treemap(negative_reason_recruiting_counts, path=['negative_reasons'], values='count', color='count',
                       color_continuous_scale='RdBu')

    # Show the chart
    st.plotly_chart(fig13, use_container_width=True)
    st.plotly_chart(fig14, use_container_width=True)

    # positive reasons for onboarding process
    q23_data = pd.DataFrame({'positive_reasons': filtered_data.iloc[:, 23]})
    q23_data['positive_reasons'] = q23_data['positive_reasons'].str.rstrip(';').str.split(';')
    q23_data = q23_data.explode('positive_reasons')
    q23_data.dropna(inplace=True)

    # Count the occurrences of each positive reason
    positive_reason_recruiting_counts = q23_data['positive_reasons'].value_counts().reset_index()
    positive_reason_recruiting_counts.columns = ['positive_reasons', 'count']

    # Calculate percentage
    positive_reason_recruiting_counts['percentage'] = positive_reason_recruiting_counts['count'] / len(
        filtered_data) * 100

    # Create a vertical bar chart
    fig15 = px.bar(positive_reason_recruiting_counts, x='positive_reasons', y='percentage', text='count',
                   color='positive_reasons', color_discrete_sequence=['#519DE9'])

    # make a tree map on the positive reasons
    fig16 = px.treemap(positive_reason_recruiting_counts, path=['positive_reasons'], values='count', color='count',
                       color_continuous_scale='RdBu')

    # Show the chart
    st.plotly_chart(fig15, use_container_width=True)
    st.plotly_chart(fig16, use_container_width=True)

    # helpful onboarding process
    q24_data = pd.DataFrame({'helpful_onboarding_process': filtered_data.iloc[:, 24]})
    q24_data['helpful_onboarding_process'] = q24_data['helpful_onboarding_process'].str.rstrip(';').str.split(';')
    q24_data = q24_data.explode('helpful_onboarding_process')
    q24_data.dropna(inplace=True)

    # Count the occurrences of each aspect that required improvement
    helpful_onboarding_counts = q24_data['helpful_onboarding_process'].value_counts().reset_index()
    helpful_onboarding_counts.columns = ['helpful_onboarding_process', 'count']

    # Calculate percentage
    helpful_onboarding_counts['percentage'] = helpful_onboarding_counts['count'] / len(filtered_data) * 100

    # Create a vertical bar chart
    fig17 = px.bar(helpful_onboarding_counts, x='helpful_onboarding_process', y='percentage', text='count',
                   color='helpful_onboarding_process', color_discrete_sequence=['#519DE9'])

    # make a tree map on the aspect that required improvement
    fig18 = px.treemap(helpful_onboarding_counts, path=['helpful_onboarding_process'], values='count', color='count',
                       color_continuous_scale='RdBu')

    # Show the chart
    st.plotly_chart(fig17, use_container_width=True)
    st.plotly_chart(fig18, use_container_width=True)

    # onboarding process to improve
    q25_data = pd.DataFrame({'onboarding_process_to_improve': filtered_data.iloc[:, 25]})
    q25_data['onboarding_process_to_improve'] = q25_data['onboarding_process_to_improve'].str.rstrip(';').str.split(';')
    q25_data = q25_data.explode('onboarding_process_to_improve')
    q25_data.dropna(inplace=True)

    # Count the occurrences of each aspect that required improvement
    aspect_onboarding_counts = q25_data['onboarding_process_to_improve'].value_counts().reset_index()
    aspect_onboarding_counts.columns = ['onboarding_process_to_improve', 'count']

    # Calculate percentage
    aspect_onboarding_counts['percentage'] = aspect_onboarding_counts['count'] / len(filtered_data) * 100

    # Create a vertical bar chart
    fig19 = px.bar(aspect_onboarding_counts, x='onboarding_process_to_improve', y='percentage', text='count',
                   color='onboarding_process_to_improve', color_discrete_sequence=['#FF7F7F'])

    # make a tree map on the aspect that required improvement
    fig20 = px.treemap(aspect_onboarding_counts, path=['onboarding_process_to_improve'], values='count', color='count',
                       color_continuous_scale='RdBu')

    # Show the chart
    st.plotly_chart(fig19, use_container_width=True)
    st.plotly_chart(fig20, use_container_width=True)

if dashboard == 'Section 3: Performance & Talent':
    filtered_data = data[
        (data['Role'].isin(selected_role)) &
        (data['Function'].isin(selected_function)) &
        (data['Location'].isin(selected_location))
        ]

    import altair as alt
    from altair import expr, datum
    import plotly.express as px
    import plotly.graph_objects as go

    # Extract the satisfaction scores column
    q26_data = pd.DataFrame({'performance_satisfaction': filtered_data.iloc[:, 26]})

    # Count the occurrences of each score
    performance_satisfaction_counts = q26_data['performance_satisfaction'].value_counts().reset_index()
    performance_satisfaction_counts.columns = ['performance_satisfaction', 'count']

    # create a dictionary to map scores to categories
    score_to_category = {
        1: 'Very Dissatisfied',
        2: 'Dissatisfied',
        3: 'Neutral',
        4: 'Satisfied',
        5: 'Very Satisfied'
    }

    # Create a new column 'performance_satisfaction_category' by mapping the 'performance_satisfaction' column to the categories
    performance_satisfaction_counts['performance_satisfaction_category'] = performance_satisfaction_counts[
        'performance_satisfaction'].map(score_to_category)

    # Calculate percentage
    performance_satisfaction_counts['percentage'] = performance_satisfaction_counts['count'] / \
                                                    performance_satisfaction_counts['count'].sum() * 100

    # Sort performance_satisfaction_counts by 'performance_satisfaction' in descending order
    performance_satisfaction_counts = performance_satisfaction_counts.sort_values('performance_satisfaction',
                                                                                  ascending=False)

    # Create a horizontal bar chart
    fig26 = px.bar(performance_satisfaction_counts, x='percentage', y='performance_satisfaction_category', text='count',
                   orientation='h', color='performance_satisfaction_category',
                   color_discrete_map={
                       'Very Dissatisfied': '#C9190B',
                       'Dissatisfied': '#EC7A08',
                       'Neutral': '#F0AB00',
                       'Satisfied': '#519DE9',
                       'Very Satisfied': '#004B95'
                   })

    # Calculate median score
    median_score_26 = q26_data['performance_satisfaction'].median()

    # Determine the color based on the median score
    if median_score_26 < 2:
        color = 'red'
    elif median_score_26 < 3:
        color = 'orange'
    elif median_score_26 < 4:
        color = 'yellow'
    else:
        color = 'green'

    # Display the median score in a text box
    st.markdown(f'<p style="color: {color};">Median Performance Satisfaction Score: {median_score_26:.2f}</p>',
                unsafe_allow_html=True)

    st.plotly_chart(fig26, use_container_width=True)

    # Extract the satisfaction scores column
    q28_data = pd.DataFrame({'career_goal_satisfaction': filtered_data.iloc[:, 28]})

    # Count the occurrences of each score
    career_goal_satisfaction_counts = q28_data['career_goal_satisfaction'].value_counts().reset_index()
    career_goal_satisfaction_counts.columns = ['career_goal_satisfaction', 'count']

    # Create a new column 'career_goal_satisfaction_category' by mapping the 'career_goal_satisfaction' column to the categories
    career_goal_satisfaction_counts['career_goal_satisfaction_category'] = career_goal_satisfaction_counts[
        'career_goal_satisfaction'].map(score_to_category)

    # Calculate percentage
    career_goal_satisfaction_counts['percentage'] = career_goal_satisfaction_counts['count'] / \
                                                    career_goal_satisfaction_counts['count'].sum() * 100

    # Sort career_goal_satisfaction_counts by 'career_goal_satisfaction' in descending order
    career_goal_satisfaction_counts = career_goal_satisfaction_counts.sort_values('career_goal_satisfaction',
                                                                                  ascending=False)

    # Create a horizontal bar chart
    fig28 = px.bar(career_goal_satisfaction_counts, x='percentage', y='career_goal_satisfaction_category', text='count',
                   orientation='h', color='career_goal_satisfaction_category',
                   color_discrete_map={
                       'Very Dissatisfied': '#C9190B',
                       'Dissatisfied': '#EC7A08',
                       'Neutral': '#F0AB00',
                       'Satisfied': '#519DE9',
                       'Very Satisfied': '#004B95'
                   })

    # Calculate median score
    median_score_28 = q28_data['career_goal_satisfaction'].median()

    # Determine the color based on the median score
    if median_score_28 < 2:
        color = 'red'
    elif median_score_28 < 3:
        color = 'orange'
    elif median_score_28 < 4:
        color = 'yellow'
    else:
        color = 'green'

    # Display the median score in a text box
    st.markdown(f'<p style="color: {color};">Median Career Goal Satisfaction Score: {median_score_28:.2f}</p>',
                unsafe_allow_html=True)

    st.plotly_chart(fig28, use_container_width=True)

    # negative reasons for career goals
    q29_data = pd.DataFrame({'negative_reasons': filtered_data.iloc[:, 29]})
    q29_data['negative_reasons'] = q29_data['negative_reasons'].str.rstrip(';').str.split(';')
    q29_data = q29_data.explode('negative_reasons')
    q29_data.dropna(inplace=True)

    # Count the occurrences of each negative reason
    negative_reason_recruiting_counts = q29_data['negative_reasons'].value_counts().reset_index()
    negative_reason_recruiting_counts.columns = ['negative_reasons', 'count']

    # Calculate percentage
    negative_reason_recruiting_counts['percentage'] = negative_reason_recruiting_counts['count'] / len(
        filtered_data) * 100

    # Create a horizontal bar chart
    fig29 = px.bar(negative_reason_recruiting_counts, x='percentage', y='negative_reasons', text='count',
                   orientation='h', color='negative_reasons', color_discrete_sequence=['#FFA500'])

    # Show the chart
    st.plotly_chart(fig29, use_container_width=True)

    # available to tag skills in HRIS
    q30_data_available_count = (filtered_data.iloc[:, 30] == 'Yes').sum()
    q30_data_available_pct = q30_data_available_count / len(filtered_data) * 100

    st.write("identify and tag your skills within the HRIS")

    st.write(
        f"{q30_data_available_pct:.2f}% of people, which are {q30_data_available_count} person(s), are able to identify and tag their skills within the HRIS.")

if dashboard == 'Section 4: Learning':
    filtered_data = data[
        (data['Role'].isin(selected_role)) &
        (data['Function'].isin(selected_function)) &
        (data['Location'].isin(selected_location))
        ]

    # Extract the satisfaction scores column
    q31_data = pd.DataFrame({'learning_satisfaction': filtered_data.iloc[:, 31]})

    # Count the occurrences of each score
    learning_satisfaction_counts = q31_data['learning_satisfaction'].value_counts().reset_index()
    learning_satisfaction_counts.columns = ['learning_satisfaction', 'count']

    # Create a dictionary to map scores to categories
    score_to_category = {
        1: 'Very Dissatisfied',
        2: 'Dissatisfied',
        3: 'Neutral',
        4: 'Satisfied',
        5: 'Very Satisfied'
    }

    # Create a new column 'learning_satisfaction_category' by mapping the 'learning_satisfaction' column to the categories
    learning_satisfaction_counts['learning_satisfaction_category'] = learning_satisfaction_counts[
        'learning_satisfaction'].map(score_to_category)

    # Calculate percentage
    learning_satisfaction_counts['percentage'] = learning_satisfaction_counts['count'] / learning_satisfaction_counts[
        'count'].sum() * 100

    # Sort learning_satisfaction_counts by 'learning_satisfaction' in descending order
    learning_satisfaction_counts = learning_satisfaction_counts.sort_values('learning_satisfaction', ascending=False)

    # Create a horizontal bar chart
    fig31 = px.bar(learning_satisfaction_counts, x='percentage', y='learning_satisfaction_category', text='count',
                   orientation='h', color='learning_satisfaction_category',
                   color_discrete_map={
                       'Very Dissatisfied': '#C9190B',
                       'Dissatisfied': '#EC7A08',
                       'Neutral': '#F0AB00',
                       'Satisfied': '#519DE9',
                       'Very Satisfied': '#004B95'
                   })

    # Calculate median score
    median_score_31 = q31_data['learning_satisfaction'].median()

    # Determine the color based on the median score
    if median_score_31 < 2:
        color = 'red'
    elif median_score_31 < 3:
        color = 'orange'
    elif median_score_31 < 4:
        color = 'yellow'
    else:
        color = 'green'

    # Display the median score in a text box
    st.markdown(f'<p style="color: {color};">Median Learning Satisfaction Score: {median_score_31:.2f}</p>',
                unsafe_allow_html=True)

    st.plotly_chart(fig31, use_container_width=True)

    # learning format distribution
    q32_data = pd.DataFrame({'learning_format': filtered_data.iloc[:, 32]})
    q32_data['learning_format'] = q32_data['learning_format'].str.rstrip(';')
    q32_data.dropna(inplace=True)

    # Count the occurrences of each learning format
    learning_format_counts = q32_data['learning_format'].value_counts().reset_index()
    learning_format_counts.columns = ['learning_format', 'count']

    # Calculate percentage
    learning_format_counts['percentage'] = learning_format_counts['count'] / learning_format_counts['count'].sum() * 100

    # Create a horizontal bar chart
    fig32 = px.bar(learning_format_counts, x='percentage', y='learning_format', text='count', orientation='h',
                   color='learning_format')

    # Show the chart
    st.plotly_chart(fig32, use_container_width=True)

    # training/development program participation
    q33_data_available_count = (filtered_data.iloc[:, 33] == 'Yes').sum()
    q33_data_available_pct = q33_data_available_count / len(filtered_data) * 100

    st.write("Training/Development Program Participation (something wrong with the data)")

    st.write(
        f"{q33_data_available_pct:.2f}% of people, which are {q33_data_available_count} person(s), are able to identify and tag their skills within the HRIS.")

    # Whether receive recommendation on training
    q34_data_available_count = (filtered_data.iloc[:, 34] == 'Yes').sum()
    q34_data_available_pct = q34_data_available_count / len(filtered_data) * 100

    st.write("Whether to Receive Recommendation on training (something wrong with the data)")

    st.write(
        f"{q34_data_available_pct:.2f}% of people, which are {q34_data_available_count} person(s), received some recommendations on training.")

if dashboard == 'Section 5: Compensation':
    filtered_data = data[
        (data['Role'].isin(selected_role)) &
        (data['Function'].isin(selected_function)) &
        (data['Location'].isin(selected_location))
        ]

    q36_compensation_count = (filtered_data['Do you participate in the Compensation Campaign ?'] == 'Yes').sum()
    q36_compensation_pct = q36_compensation_count / len(filtered_data) * 100

    st.write("Compensation Campaign Participation")

    st.write("%.2f" % q36_compensation_pct, "% of people, which are", q36_compensation_count,
             "person(s), participate in the Compensation Campaign.")

    q37_data_available_count = (filtered_data[
                                    'Do you think that the data available in the Compensation form enables you to make a fair decision regarding a promotion, a bonus or a raise ? (e.g : compa-ratio, variation between years, historica...'] == 'Yes').sum()
    q37_data_available_pct = q37_data_available_count / q36_compensation_count * 100

    st.write("Data Available in Compensation Form")

    st.write("Among the people who participate the Compensation Campaign "
             "%.2f" % q37_data_available_pct, "% of people, which are", q37_data_available_count,
             "\nperson(s), think that the data available in the Compensation form enables him/her to make \na fair decision regarding a promotion, a bonus or a raise.")

    st.write("Data Missing")

    q38_data_reliable = filtered_data['What data is missing according to you ?'].dropna()
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # Replace spaces with underscores
    phrases = q38_data_reliable.str.replace(' ', '_')

    # Generate word cloud
    wordcloud = WordCloud(width=1000, height=500).generate(' '.join(phrases))

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud)
    plt.axis("off")

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Convert the data to a DataFrame
    q39_campaign_manage = pd.DataFrame(
        {'Campaign': filtered_data["Do you manage/launch your compensation campaigns nationally or in another way?\n"]})

    # Drop NaN values
    q39_campaign_manage.dropna(inplace=True)

    # Count occurrences of each campaign type
    campaign_manage_counts = q39_campaign_manage['Campaign'].value_counts().reset_index()
    campaign_manage_counts.columns = ['Campaign', 'Count']
    campaign_manage_counts['Percentage'] = campaign_manage_counts['Count'] / len(q39_campaign_manage) * 100

    # Sort the DataFrame by count
    campaign_manage_counts = campaign_manage_counts.sort_values(by='Count', ascending=False)

    # Create the bar chart using Plotly
    fig33 = px.bar(campaign_manage_counts, x='Campaign', y='Count', text='Percentage',
                   title="Do you manage/launch your compensation campaigns nationally or in another way?")
    fig33.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    # Customize the layout
    fig33.update_layout(
        xaxis_title="Campaign",
        yaxis_title="Count",
        showlegend=False
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig33)

    # Extract the column
    q40_compensation_satisfaction = pd.DataFrame({'satisfaction_score': filtered_data.iloc[:, 40]}).dropna()

    # Count the occurrences of each value
    value_counts = q40_compensation_satisfaction['satisfaction_score'].value_counts().reset_index()
    value_counts.columns = ['satisfaction_score', 'count']

    # Create a dictionary to map scores to categories
    score_to_category = {
        1: 'Very Dissatisfied',
        2: 'Dissatisfied',
        3: 'Neutral',
        4: 'Satisfied',
        5: 'Very Satisfied'
    }

    # Create a new column 'category' by mapping the 'column_40' column to the categories
    value_counts['category'] = value_counts['satisfaction_score'].map(score_to_category)

    # Calculate percentage
    value_counts['percentage'] = value_counts['count'] / value_counts['count'].sum() * 100

    # Create a horizontal bar chart
    fig34 = px.bar(value_counts, x='percentage', y='category', text='count', orientation='h', color='category')

    # Add interactivity to the bar chart only
    fig34.update_traces(texttemplate='%{text:.2s}', textposition='inside', selector=dict(type='bar'))
    fig34.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

    st.plotly_chart(fig34, use_container_width=True)

    # column 43 available count retroactivity on salary payments
    q43_data_available_count = (filtered_data.iloc[:, 43] == 'Yes').sum()
    q43_data_available_pct = q43_data_available_count / q36_compensation_count * 100

    st.write("Retroactivity on Salary Payments")

    st.write("Among the people who participate the Compensation Campaign "
             "%.2f" % q43_data_available_pct, "% of people, which are", q43_data_available_count,
             "person(s), have retroactivity on salary payments.")

    # column 44 available count participation in the variable pay/bonus campaign
    q44_data_available_count = (filtered_data.iloc[:, 44] == 'Yes').sum()
    q44_data_available_pct = q44_data_available_count / q36_compensation_count * 100

    st.write("Variable Pay/Bonus Campaign Participation")
    st.write("Among the people who participate the Compensation Campaign "
             "%.2f" % q44_data_available_pct, "% of people, which are", q44_data_available_count,
             "person(s), participate in the Variable Pay/Bonus Campaign.")

    # Extract satisfaction scores column for variable pay/bonus campaign
    q45_data = pd.DataFrame({'variable_pay_satisfaction': filtered_data.iloc[:, 45]})

    # Count the occurrences of each score
    variable_pay_satisfaction_counts = q45_data['variable_pay_satisfaction'].value_counts().reset_index()
    variable_pay_satisfaction_counts.columns = ['variable_pay_satisfaction', 'count']

    # Create a new column 'category' by mapping the 'variable_pay_satisfaction' column to the categories
    variable_pay_satisfaction_counts['category'] = variable_pay_satisfaction_counts['variable_pay_satisfaction'].map(
        score_to_category)

    # Calculate percentage
    variable_pay_satisfaction_counts['percentage'] = variable_pay_satisfaction_counts['count'] / \
                                                     variable_pay_satisfaction_counts['count'].sum() * 100

    # Create a horizontal bar chart
    fig35 = px.bar(variable_pay_satisfaction_counts, x='percentage', y='category', text='count', orientation='h',
                   color='category')

    # Add interactivity to the bar chart only
    fig35.update_traces(texttemplate='%{text:.2s}', textposition='inside', selector=dict(type='bar'))
    fig35.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

    st.plotly_chart(fig35, use_container_width=True)

    # Convert the data to a DataFrame
    q46_campaign_manage = pd.DataFrame({'bonus/variable pay campaign': filtered_data.iloc[:, 46]})

    # Drop NaN values
    q46_campaign_manage.dropna(inplace=True)

    # Count occurrences of each campaign type
    campaign_manage_counts = q46_campaign_manage['bonus/variable pay campaign'].value_counts().reset_index()
    campaign_manage_counts.columns = ['bonus/variable pay campaign', 'Count']
    campaign_manage_counts['Percentage'] = campaign_manage_counts['Count'] / len(q46_campaign_manage) * 100

    # Sort the DataFrame by count
    campaign_manage_counts = campaign_manage_counts.sort_values(by='Count', ascending=False)

    # Create the bar chart using Plotly
    fig36 = px.bar(campaign_manage_counts, x='bonus/variable pay campaign', y='Count', text='Percentage',
                   title="Do you manage/launch your bonus/variable pay campaigns nationally or in another way?")
    fig36.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    # Customize the layout
    fig36.update_layout(
        xaxis_title="Campaign",
        yaxis_title="Count",
        showlegend=False
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig36)

    # the dates of your Variable Pay campaign different from the one for the Compensation Campaign
    q47_data_available_count = (filtered_data.iloc[:, 47] == 'Yes').sum()
    q47_data_available_pct = q47_data_available_count / q36_compensation_count * 100

    st.write("Variable Pay Campaign Dates Different from Compensation Campaign Dates")

    st.write("Among the people who participate the Compensation Campaign "
             "%.2f" % q47_data_available_pct, "% of people, which are", q47_data_available_count,
             "person(s), have different dates for the Variable Pay Campaign compared to the Compensation Campaign.")

if dashboard == 'Section 6: Payroll':
    filtered_data = data[
        (data['Role'].isin(selected_role)) &
        (data['Function'].isin(selected_function)) &
        (data['Location'].isin(selected_location))
        ]

    # Payroll team
    q48_payroll_count = (filtered_data.iloc[:, 48] == 'Yes').sum()
    q48_payroll_pct = q48_payroll_count / len(filtered_data) * 100

    st.write("Payroll Team")

    st.write("%.2f" % q48_payroll_pct, "% of people, which are", q48_payroll_count,
             "person(s), are part of the payroll team.")

    # Extract the satisfaction scores column
    q49_data = pd.DataFrame({'Payroll_satisfaction': filtered_data.iloc[:, 49]})

    # Count the occurrences of each score
    payroll_satisfaction_counts = q49_data['Payroll_satisfaction'].value_counts().reset_index()
    payroll_satisfaction_counts.columns = ['Payroll_satisfaction', 'count']

    # Create a dictionary to map scores to categories
    score_to_category = {
        1: 'Very Dissatisfied',
        2: 'Dissatisfied',
        3: 'Neutral',
        4: 'Satisfied',
        5: 'Very Satisfied'
    }

    # Create a new column 'Payroll_satisfaction_category' by mapping the 'Payroll_satisfaction' column to the categories
    payroll_satisfaction_counts['Payroll_satisfaction_category'] = payroll_satisfaction_counts[
        'Payroll_satisfaction'].map(score_to_category)

    # Calculate percentage
    payroll_satisfaction_counts['percentage'] = payroll_satisfaction_counts['count'] / payroll_satisfaction_counts[
        'count'].sum() * 100

    # Sort payroll_satisfaction_counts by 'Payroll_satisfaction' in descending order
    payroll_satisfaction_counts = payroll_satisfaction_counts.sort_values('Payroll_satisfaction', ascending=False)

    # Create a horizontal bar chart
    fig37 = px.bar(payroll_satisfaction_counts, x='percentage', y='Payroll_satisfaction_category', text='count',
                   orientation='h', color='Payroll_satisfaction_category',
                   color_discrete_map={
                       'Very Dissatisfied': '#C9190B',
                       'Dissatisfied': '#EC7A08',
                       'Neutral': '#F0AB00',
                       'Satisfied': '#519DE9',
                       'Very Satisfied': '#004B95'
                   })

    # Calculate median score
    median_score_49 = q49_data['Payroll_satisfaction'].median()

    # Determine the color based on the median score
    if median_score_49 < 2:
        color = 'red'
    elif median_score_49 < 3:
        color = 'orange'
    elif median_score_49 < 4:
        color = 'yellow'
    else:
        color = 'green'

    # Display the median score in a text box
    st.markdown(f'<p style="color: {color};">Median Payroll Satisfaction Score: {median_score_49:.2f}</p>',
                unsafe_allow_html=True)

    st.plotly_chart(fig37, use_container_width=True)

    # internally or outsourced
    q50_data = filtered_data.iloc[:, 50]

    # Count the occurrences of each value
    q50_counts = q50_data.value_counts().reset_index()
    q50_counts.columns = ['internally or outsourced', 'count']

    # Calculate percentage
    q50_counts['percentage'] = q50_counts['count'] / q50_counts['count'].sum() * 100

    # Create a horizontal bar chart
    fig38 = px.bar(q50_counts, x='percentage', y='internally or outsourced', text='count', orientation='h',
                   color='internally or outsourced',
                   title='Do you realize your payroll activities internally or is it outsourced ?')

    # show the chart
    st.plotly_chart(fig38, use_container_width=True)

    # cover legal updates
    q51_legal_count = (filtered_data.iloc[:, 51] == 'Yes').sum()
    q51_legal_pct = q51_legal_count / q48_payroll_count * 100

    st.write("Cover Legal Updates")
    st.write("%.2f" % q51_legal_pct, "% of people, which are", q51_legal_count,
             "person(s), have the systems cover legal updates.")

    # autonomous or outsourced
    q52_data = filtered_data.iloc[:, 52]

    # Count the occurrences of each value
    q52_counts = q52_data.value_counts().reset_index()
    q52_counts.columns = ['autonomous or outsourced', 'count']

    # Calculate percentage
    q52_counts['percentage'] = q52_counts['count'] / q52_counts['count'].sum() * 100

    # Create a horizontal bar chart
    fig39 = px.bar(q52_counts, x='percentage', y='autonomous or outsourced', text='count', orientation='h',
                   color='autonomous or outsourced',
                   title='Are you autonomous when it comes to updating simple data, or do you systematically rely on outside firms for updates?')

    # show the chart
    st.plotly_chart(fig39, use_container_width=True)

    # global platform or not
    q54_global_count = (filtered_data.iloc[:, 54] == 'Yes').sum()
    q54_global_pct = q54_global_count / q48_payroll_count * 100
    st.write("Global Platform")
    st.write("If the payroll system is used in several countries," "%.2f" % q54_global_pct, "% of people, which are",
             q54_global_count, "person(s), have a global platform for consolidating all the employees' country data.")

    # automatically generate KPIs relating to the payroll
    q55_auto_count_yes = (filtered_data.iloc[:, 55] == 'Yes').sum()
    q55_auto_pct_yes = q55_auto_count_yes / q54_global_count * 100

    q55_auto_count_no = (filtered_data.iloc[:, 55] == 'No').sum()
    q55_auto_pct_no = q55_auto_count_no / q54_global_count * 100

    q55_auto_count_not_concerned = (filtered_data.iloc[:, 55] == 'Not concerned').sum()
    q55_auto_pct_not_concerned = q55_auto_count_not_concerned / q54_global_count * 100
    st.write("Automatically Generate KPIs")
    st.write(
        "Among the people who have a global platform for consolidating all the employees' country data," "%.2f" % q55_auto_pct_yes,
        "% of people, which are", q55_auto_count_yes,
        "person(s), automatically generate KPIs relating to the payroll." "\n" "%.2f" % q55_auto_pct_no,
        "% of people, which are", q55_auto_count_no,
        "person(s), do not automatically generate KPIs relating to the payroll." "\n" "%.2f" % q55_auto_pct_not_concerned,
        "% of people, which are", q55_auto_count_not_concerned,
        "person(s), are not concerned about automatically generating KPIs relating to the payroll.")

    # mass entries ability
    q56_mass_count = (filtered_data.iloc[:, 56] == 'Yes').sum()
    q56_mass_pct = q56_mass_count / q48_payroll_count * 100
    st.write("Mass Entries Ability")
    st.write("%.2f" % q56_mass_pct, "% of people, which are", q56_mass_count,
             "person(s), have the tool to make mass entries.")

    # connectivity with the core HR function
    q57_connectivity_count_yes = (filtered_data.iloc[:, 57] == 'Yes').sum()
    q57_connectivity_pct_yes = q57_connectivity_count_yes / q48_payroll_count * 100

    q57_connectivity_count_no = (filtered_data.iloc[:, 57] == 'No').sum()
    q57_connectivity_pct_no = q57_connectivity_count_no / q48_payroll_count * 100

    q57_connectivity_count_not_core = (filtered_data.iloc[:, 57] == 'I do not have this type of system currently').sum()
    q57_connectivity_pct_not_core = q57_connectivity_count_not_core / q48_payroll_count * 100

    st.write("Connectivity with the Core HR/Administration Function")
    st.write("In the payroll team," "%.2f" % q57_connectivity_pct_yes, "% of people, which are",
             q57_connectivity_count_yes,
             "person(s), have connectivity with the core HR function." "\n" "%.2f" % q57_connectivity_pct_no,
             "% of people, which are", q57_connectivity_count_no,
             "person(s), do not have connectivity with the core HR function." "\n" "%.2f" % q57_connectivity_pct_not_core,
             "% of people, which are", q57_connectivity_count_not_core,
             "person(s), do not have this type of system currently.")

    # connectivity with the core HR function
    q58_connectivity_count_yes = (filtered_data.iloc[:, 58] == 'Yes').sum()
    q58_connectivity_pct_yes = q58_connectivity_count_yes / q48_payroll_count * 100

    q58_connectivity_count_no = (filtered_data.iloc[:, 58] == 'No').sum()
    q58_connectivity_pct_no = q58_connectivity_count_no / q48_payroll_count * 100

    q58_connectivity_count_not_core = (filtered_data.iloc[:, 58] == 'I do not have this type of system currently').sum()
    q58_connectivity_pct_not_core = q58_connectivity_count_not_core / q48_payroll_count * 100

    st.write("Connectivity with the Core HR/Administration Function")
    st.write("In the payroll team," "%.2f" % q58_connectivity_pct_yes, "% of people, which are",
             q58_connectivity_count_yes,
             "person(s), have connectivity with the core HR function." "\n" "%.2f" % q58_connectivity_pct_no,
             "% of people, which are", q58_connectivity_count_no,
             "person(s), do not have connectivity with the core HR function." "\n" "%.2f" % q58_connectivity_pct_not_core,
             "% of people, which are", q58_connectivity_count_not_core,
             "person(s), do not have this type of system currently.")

if dashboard == 'Section 7: Time Management':
    filtered_data = data[
        (data['Role'].isin(selected_role)) &
        (data['Function'].isin(selected_function)) &
        (data['Location'].isin(selected_location))
        ]

    # part of time management team
    q59_time_management_count = (filtered_data.iloc[:, 59] == 'Yes').sum()
    q59_time_management_pct = q59_time_management_count / len(filtered_data) * 100

    st.write("Time Management Team")
    st.write("%.2f" % q59_time_management_pct, "% of people, which are", q59_time_management_count,
             "person(s), are part of the time management team.")

    # do you have a time management system
    q60_data = filtered_data.iloc[:, 60]
    q60_yes_count = (q60_data == 'Yes').sum()
    q60_yes_pct = q60_yes_count / q59_time_management_count * 100

    st.write("Time Management System")
    st.write("%.2f" % q60_yes_pct, "% of people, which are", q60_yes_count, "person(s), have a time management system.")

    # time management satisfaction
    q61_data = pd.DataFrame({'time_management_satisfaction': filtered_data.iloc[:, 61]})
    q61_data.dropna(inplace=True)

    # Count the occurrences of each score
    time_management_satisfaction_counts = q61_data['time_management_satisfaction'].value_counts().reset_index()
    time_management_satisfaction_counts.columns = ['time_management_satisfaction', 'count']

    # Create a dictionary to map scores to categories
    score_to_category = {
        1: 'Very Dissatisfied',
        2: 'Dissatisfied',
        3: 'Neutral',
        4: 'Satisfied',
        5: 'Very Satisfied'
    }

    # Create a new column 'category' by mapping the 'time_management_satisfaction' column to the categories
    time_management_satisfaction_counts['category'] = time_management_satisfaction_counts[
        'time_management_satisfaction'].map(score_to_category)

    # Calculate percentage
    time_management_satisfaction_counts['percentage'] = time_management_satisfaction_counts['count'] / \
                                                        time_management_satisfaction_counts['count'].sum() * 100

    # Sort time management by 'time_management_satisfaction' in descending order
    time_management_satisfaction_counts = time_management_satisfaction_counts.sort_values(
        'time_management_satisfaction', ascending=False)

    # Create a horizontal bar chart
    fig40 = px.bar(time_management_satisfaction_counts, x='percentage', y='time_management_satisfaction', text='count',
                   orientation='h', color='category',
                   color_discrete_map={
                       'Very Dissatisfied': '#C9190B',
                       'Dissatisfied': '#EC7A08',
                       'Neutral': '#F0AB00',
                       'Satisfied': '#519DE9',
                       'Very Satisfied': '#004B95'
                   })

    # Calculate median score
    median_score_61 = q61_data['time_management_satisfaction'].median()

    # Determine the color based on the median score
    if median_score_61 < 2:
        color = 'red'
    elif median_score_61 < 3:
        color = 'orange'
    elif median_score_61 < 4:
        color = 'yellow'
    else:
        color = 'green'

    # Display the median score in a text box
    st.markdown(f'<p style="color: {color};">Median Time Management Satisfaction Score: {median_score_61:.2f}</p>',
                unsafe_allow_html=True)

    st.plotly_chart(fig40, use_container_width=True)

    # self-service
    q62_yes = (filtered_data.iloc[:, 62] == 'Yes').sum()
    q62_yes_pct = q62_yes / q59_time_management_count * 100

    st.write("Self-Service")
    st.write("%.2f" % q62_yes_pct, "% of people, which are", q62_yes,
             "person(s), have a self-service for the employees.")

    # vacation counters
    q63_yes = (filtered_data.iloc[:, 63] == 'Yes').sum()
    q63_yes_pct = q63_yes / q59_time_management_count * 100

    st.write("Vacation Counters")
    st.write("%.2f" % q63_yes_pct, "% of people, which are", q63_yes,
             "person(s), have the system allow employees to view their vacation counters (entitlement / taken / balance).")

    # cover shift scheduling shifts
    q64_yes = (filtered_data.iloc[:, 64] == 'Yes').sum()
    q64_yes_pct = q64_yes / q59_time_management_count * 100

    st.write("Cover Shift Scheduling Shifts")
    st.write("%.2f" % q64_yes_pct, "% of people, which are", q64_yes,
             "person(s), have the system cover all the shift scheduling functions needed.")

    # capability to run all the report needed
    q65_yes = (filtered_data.iloc[:, 65] == 'Yes').sum()
    q65_yes_pct = q65_yes / q59_time_management_count * 100

    st.write("Capability to Run All the Reports Needed")
    st.write("%.2f" % q65_yes_pct, "% of people, which are", q65_yes,
             "person(s), have the capability to run all the report needed.")

    # allow employees to take their own leave
    q67_yes = (filtered_data.iloc[:, 67] == 'Yes').sum()
    q67_yes_pct = q67_yes / q59_time_management_count * 100

    st.write("Allow Employees to Take Their Own Leave")
    st.write("%.2f" % q67_yes_pct, "% of people, which are", q67_yes,
             "person(s), have the system allow employees to take their own leave, with workflow validation by their manager or HR.")

    # automatically take retroactive items into account
    q68_yes = (filtered_data.iloc[:, 68] == 'Yes').sum()
    q68_yes_pct = q68_yes / q59_time_management_count * 100

    st.write("Automatically Take Retroactive Items into Account")
    st.write("%.2f" % q68_yes_pct, "% of people, which are", q68_yes,
             "person(s), have the system automatically take retroactive items into account (e.g. application to April payroll of a salary increase with an effective date of January 1).")

if dashboard == 'Section 8: User Experience':
    filtered_data = data[
        (data['Role'].isin(selected_role)) &
        (data['Function'].isin(selected_function)) &
        (data['Location'].isin(selected_location))
        ]

    # time well spent
    q71_yes = (filtered_data.iloc[:, 71] == 'Yes').sum()
    q71_yes_pct = q71_yes / len(filtered_data) * 100

    st.write("Time Well Spent")
    st.write("%.2f" % q71_yes_pct, "% of people, which are", q71_yes,
             "person(s), think that the time spent on the system is well spent.")

    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
    import torch
    import numpy as np


    def load_models():
        # Load the tokenizers and models
        tokenizer_1 = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        model_1 = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

        tokenizer_2 = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
        model_2 = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-emotion")

        return tokenizer_1, model_1, tokenizer_2, model_2


    def predict_emotions_hybrid(df, text_columns):
        tokenizer_1, model_1, tokenizer_2, model_2 = load_models()

        emotion_labels_1 = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        emotion_labels_2 = ["anger", "joy", "optimism", "sadness"]

        for column in text_columns:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' does not exist in DataFrame")
            df[column] = df[column].fillna("")

        for column in text_columns:
            # Predictions from the first model
            encoded_texts_1 = tokenizer_1(df[column].tolist(), padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs_1 = model_1(**encoded_texts_1)
                probabilities_1 = torch.nn.functional.softmax(outputs_1.logits, dim=-1)

            # Predictions from the second model
            encoded_texts_2 = tokenizer_2(df[column].tolist(), padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs_2 = model_2.generate(input_ids=encoded_texts_2['input_ids'],
                                             attention_mask=encoded_texts_2['attention_mask'])
                predicted_labels_2 = [tokenizer_2.decode(output, skip_special_tokens=True) for output in outputs_2]
                probabilities_2 = torch.tensor(
                    [[1 if label == emotion else 0 for emotion in emotion_labels_2] for label in predicted_labels_2])

            # Adjust probabilities_2 to match the length of emotion_labels_1 by filling missing labels with zero probabilities
            adjusted_probabilities_2 = torch.zeros(probabilities_2.size(0), len(emotion_labels_1))
            for i, emotion in enumerate(emotion_labels_2):
                if emotion in emotion_labels_1:
                    adjusted_probabilities_2[:, emotion_labels_1.index(emotion)] = probabilities_2[:, i]

            # Average the probabilities
            averaged_probabilities = (probabilities_1 + adjusted_probabilities_2) / 2
            predicted_emotions = [emotion_labels_1[probability.argmax()] for probability in averaged_probabilities]

            df[f'{column}_predicted_emotion'] = predicted_emotions

        return df


    # Load the DataFrame from the Excel file
    df = pd.read_excel('/content/data.xlsx')

    # Specify the columns to analyze
    columns_to_analyze = [
        'What could be improved or what kind of format is missing today ?',
        'In the context of your job, what are the most valuable activities your current HRIS enable you to do?',
        'In the context of your job, what do your current HRIS fail to address?',
        'In 3 words, how would you describe your current user-experience with the HRIS ?'
    ]

    # Run the function
    df_with_emotions = predict_emotions_hybrid(df, columns_to_analyze)

    # Display the DataFrame with predicted emotions
    df_with_emotions.head()






