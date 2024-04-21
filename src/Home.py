import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Sets the page to wide layout.
st.set_page_config(layout="wide")

# Load and clean data
@st.cache_data
def load_data():
    # the path needs to change to a GitHub path
    file_name = 'data.xlsx'
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, file_name)
    data = pd.read_excel(file_path) 
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
                                             'Section 8: User Experience',
                                             'Chatbot'
                                             ))

# Sidebar for tags selection
selected_role = st.sidebar.multiselect('Select Role', options=data['Role'].unique(), default=data['Role'].unique())
selected_function = st.sidebar.multiselect('Select Function', options=data['Function'].unique(), default=data['Function'].unique())
selected_location = st.sidebar.multiselect('Select Location', options=data['Location'].unique(), default=data['Location'].unique())


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
    render_header("General Employee Experience")
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
        location_summary['Label'] = location_summary['Continent'].apply(lambda x: f"{x}: {location_summary.loc[location_summary['Continent'] == x, 'Count'].iloc[0]}")

        role_summary = data['Role'].value_counts().rename_axis('Role').reset_index(name='Count')
        function_summary = data['Function'].value_counts().rename_axis('Function').reset_index(name='Count')
        return location_summary, role_summary, function_summary
    location_summary, role_summary, function_summary = prepare_summaries(data)

    filtered_data = data[
        (data['Role'].isin(selected_role)) &
        (data['Function'].isin(selected_function)) &
        (data['Location'].isin(selected_location))
    ]
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
            marker=dict(size=location_summary['Count']*2, line=dict(width=0)),  # Remove the white border by setting the line width to 0
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
        fig_role.update_traces(marker_color= mark_color, text=role_summary['Count'], textposition='outside')
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
        fig_function.update_traces(marker_color= mark_color, text=function_summary['Count'], textposition='outside')
        fig_function.update_yaxes(showticklabels=True, title='')
        fig_function.update_xaxes(showticklabels=False, title='')
        st.plotly_chart(fig_function, use_container_width=True)

if dashboard == "Recruiting & Onboarding":
    def plot_satisfaction_proportions(data_series, title):
        # Count the occurrences of each score
        score_counts = data_series.value_counts().sort_index().astype(int)

        # Calculate the total counts for 'Satisfied' and 'Dissatisfied' categories
        total_satisfied = score_counts.get(4, 0) + score_counts.get(5, 0)
        total_dissatisfied = score_counts.get(1, 0) + score_counts.get(2, 0) + score_counts.get(3, 0)

        # Calculate proportions for each score category
        dissatisfied_proportions = [score_counts.get(i, 0) / total_dissatisfied if total_dissatisfied > 0 else 0 for i in range(1, 4)]
        satisfied_proportions = [score_counts.get(i, 0) / total_satisfied if total_satisfied > 0 else 0 for i in range(4, 6)]

        # Create the stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 2))

        # Positions of the bars on the y-axis
        bar_positions = [0, 1]

        # Cumulative size for each segment to get the start position of the next segment
        cumulative_size_dissatisfied = 0
        cumulative_size_satisfied = 0

        # Plot each score segment for 'Dissatisfied'
        for i in range(3):
            ax.barh(bar_positions[0], dissatisfied_proportions[i], left=cumulative_size_dissatisfied, edgecolor='white', color=sns.color_palette("Blues_d", n_colors=3)[i])
            cumulative_size_dissatisfied += dissatisfied_proportions[i]

        # Plot each score segment for 'Satisfied'
        for i in range(2):
            ax.barh(bar_positions[1], satisfied_proportions[i], left=cumulative_size_satisfied, edgecolor='white', color=sns.color_palette("Greens_d", n_colors=2)[i])
            cumulative_size_satisfied += satisfied_proportions[i]

        # Add labels and a title
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Dissatisfied', 'Satisfied'])
        ax.set_title(title)

        # Remove x-axis ticks for clarity
        ax.set_xticks([])

        # Add annotations for each segment
        for i, prop in enumerate(dissatisfied_proportions):
            ax.text(prop / 2 + sum(dissatisfied_proportions[:i]), 0, f'{i+1} ({prop:.0%})', va='center', ha='center', color='white')

        for i, prop in enumerate(satisfied_proportions):
            ax.text(prop / 2 + sum(satisfied_proportions[:i]), 1, f'{i+4} ({prop:.0%})', va='center', ha='center', color='white')

        # Add annotation for the total count of dissatisfied
        ax.text(1.05, 0, f'Total: {total_dissatisfied}', va='center', ha='left', color='black', fontsize=10)

        # Add annotation for the total count of satisfied
        ax.text(1.05, 1, f'Total: {total_satisfied}', va='center', ha='left', color='black', fontsize=10)

        plt.show()
        st.pyplot(fig)
    plot_satisfaction_proportions(data['From 1 to 5, how would you rate the onboarding process ?'], 'Proportion of Onboarding Process Satisfaction Scores')
