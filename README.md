[![GitHub Link](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/001202ZHENG/V1_Chatbot_Streamlit)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-orange?logo=streamlit)](https://v1chatbotapp-2syadnkccp6nyevahkmkbm.streamlit.app/)

# VoC Project

## Overview
Welcome to the VoC Chatbot project! This repository contains code for a dashboard visualizing HR related survey data and an AI-powered chatbot designed to process and analyze survey data, providing valuable insights. 

## Project Background
In the evolving landscape of Human Resources, capturing and understanding employee sentiment is crucial for fostering a positive work environment and making informed decisions. Traditional survey analysis methods can be time-consuming and insufficient for capturing the depth of employee sentiments. Our project aims to develop an automated tool that integrates sentiment analysis, visualization technologies, and generative AI to address these challenges.

## Links Running Online
- **Streamlit Application**: 
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-orange?logo=streamlit)](https://v1chatbotapp-2syadnkccp6nyevahkmkbm.streamlit.app/)

## Features
1. **Visualization Tool**:
   - Utilizes Streamlit for visualizing survey data.
   - Provides comprehensive exploration across multiple dimensions (e.g., geographic distribution, satisfaction levels, sentiment analysis).
   - Enables dynamic content updates and drill-down capabilities.

2. **Sentiment Analysis**:
   - Employs NLP techniques to transform survey textual data into structured insights.
   - Provides sentiment analysis results to gauge overall workforce sentiment.

3. **Generative AI**:
   - Uses OpenAI models to generate responses to user queries.
   - Supports dynamic analysis based on user prompts.

## Usage
Once the application is running, you can upload survey data files (CSV, PDF, TXT, XLSX) via the sidebar. The chatbot will process the data and provide insights based on your queries. You can interact with the chatbot to explore data trends, perform sentiment analysis, etc.


## Project Structure

```
├── data
├── embeddings
├── src
│   ├── modules
│   │   ├── robby_sheet
│   │   │   ├── chatbot.py
│   │   │   ├── embedder.py
│   │   │   ├── history.py
│   │   │   ├── layout.py
│   │   │   ├── sidebar.py
│   │   │   ├── utils.py
│   ├── pages
│   │   ├── 1__VoC-Chat.py
│   │   ├── Home.py
├── LICENSE
├── README.md
├── requirements.txt
├── setup.sh
```

### Explanation

- **data/**: Directory to store input data files.
- **embeddings/**: Directory to store document embeddings.
- **src/**: Contains all the source code for the project.
  - **modules/**: Contains the core functionality and helper classes for the chatbot.
    - **robby_sheet/**:
      - **chatbot.py**: Contains the `Chatbot` class for handling conversations and retrieving information.
      - **embedder.py**: Contains the `Embedder` class for creating and retrieving document embeddings.
      - **history.py**: Manages chat history and user interactions.
      - **layout.py**: Defines the layout and UI components of the application.
      - **sidebar.py**: Handles sidebar options and configurations.
      - **utils.py**: Utility functions for loading API keys, handling file uploads, and setting up the chatbot.
  - **pages/**: Contains the Streamlit pages for the application.
    - **1__VoC-Chat.py**: Main page that calls and implements the chatbot functionality.
    - **Home.py**: Contains code related to visualization of the data.
- **LICENSE**: License file for the project.
- **README.md**: This README file.
- **requirements.txt**: Lists the dependencies required for the project.
- **setup.sh**: Setup script for initializing the project.

## Installation
To set up the VoC Chatbot on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/001202ZHENG/V1_Chatbot_Streamlit.git
   cd V1_Chatbot_Streamlit
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key**:
   - Create a `.env` file in the root directory of the project and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

4. **Run the Streamlit application**:
   ```bash
   streamlit run pages/1__VoC-Chat.py
   ```
   
## Acknowledgements
This project was developed as part of a collaboration between students from ESSEC Business School and Deloitte HR Transformation team. We are grateful for the support and guidance provided by our professors and the Deloitte team.
