import os 
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from sqlalchemy import create_engine
import re
from PIL import Image

# Database connection configuration
DATABASE_URI = 'postgresql://postgres:Smallholder19@localhost:5432/moisture_results'

# Define paths for the images
MAP_IMAGE_PATH = "data/map_image/Screenshot_2-11-2024_73055_browser.dataspace.copernicus.eu.jpeg"
CLASSIFICATION_IMAGE_PATH = "data/map_image/2024-07-03-00_00_2024-07-03-23_59_Sentinel-2_L2A_Scene_classification_map_.png"
MASK_OUTPUT_FOLDER = "output_results"

# Define coordinates of area of interest
COORDINATES = [
    [[35.632233, -15.492857], [35.376801, -15.492857], 
     [35.376801, -15.352397], [35.632233, -15.352397], 
     [35.632233, -15.492857]]
]

# Function to fetch and process data
def fetch_data():
    engine = create_engine(DATABASE_URI)
    query = """
    SELECT
        image_name,
        dry_percentage,
        normal_percentage,
        wet_percentage,
        condition_summary
    FROM moisture_results;
    """
    df = pd.read_sql(query, engine)
    df['date'] = df['image_name'].apply(lambda x: re.search(r"\d{4}-\d{2}-\d{2}", x).group() if re.search(r"\d{4}-\d{2}-\d{2}", x) else None)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values(by='date')
    df.set_index('date', inplace=True)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    monthly_df = numeric_df.resample('M').mean()
    return monthly_df, df

# Function to get the latest mask image from the output folder
def get_latest_mask_image():
    try:
        mask_files = [f for f in os.listdir(MASK_OUTPUT_FOLDER) if f.endswith('.png')]
        if mask_files:
            latest_image_file = max(mask_files, key=lambda x: os.path.getctime(os.path.join(MASK_OUTPUT_FOLDER, x)))
            latest_image_path = os.path.join(MASK_OUTPUT_FOLDER, latest_image_file)
            return latest_image_path
    except Exception as e:
        print(f"Error fetching latest mask image: {e}")
    return None

# Streamlit app layout
st.set_page_config(layout="wide")

# Centered buttons
st.markdown(
    "<style>div.row-widget.stButton {text-align: center;}</style>", 
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Soil Moisture Dashboard for Zomba")
options = st.sidebar.radio("Select a view:", ("Recent Moisture Readings", "Historical Readings"))

# Fetch the data
monthly_df, raw_df = fetch_data()

# Display initial images
if st.button("Area of Interest Classification Map"):
    st.subheader("Map of Area of Interest")
    st.image(MAP_IMAGE_PATH, caption="Map of Area of Interest", use_column_width=True)
    
    st.subheader("Classification Image")
    st.image(CLASSIFICATION_IMAGE_PATH, caption="Classification of Features", use_column_width=True)

# Main page content
if options == "Recent Moisture Readings":
    st.title("Recent Moisture Readings")
    if not monthly_df.empty:
        latest_reading = monthly_df.iloc[-1]
        st.subheader(f"Latest Monthly Average: {latest_reading.name.strftime('%B %Y')}")
        
        # Display percentage values with styling
        st.markdown(f"""
        <div style="display: flex; gap: 20px; justify-content: center;">
            <div style="background-color: #ffcccc; padding: 20px; border-radius: 8px; width: 200px; text-align: center;">
                <h4 style="color: red;">Dry Percentage</h4>
                <p style="font-size: 24px; font-weight: bold;">{latest_reading['dry_percentage']:.2f}%</p>
            </div>
            <div style="background-color: #fff2cc; padding: 20px; border-radius: 8px; width: 200px; text-align: center;">
                <h4 style="color: #FFD700;">Normal Percentage</h4>
                <p style="font-size: 24px; font-weight: bold;">{latest_reading['normal_percentage']:.2f}%</p>
            </div>
            <div style="background-color: #cce5ff; padding: 20px; border-radius: 8px; width: 200px; text-align: center;">
                <h4 style="color: blue;">Wet Percentage</h4>
                <p style="font-size: 24px; font-weight: bold;">{latest_reading['wet_percentage']:.2f}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Display pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Dry', 'Normal', 'Wet'],
            values=[latest_reading['dry_percentage'], latest_reading['normal_percentage'], latest_reading['wet_percentage']],
            marker=dict(colors=['red', 'yellow', 'blue'])
        )])
        fig.update_layout(title="Moisture Conditions - Latest Reading")
        st.plotly_chart(fig)

        # Display recommendation based on moisture conditions
        if latest_reading['wet_percentage'] < 9.5 and latest_reading['dry_percentage'] > 51 and latest_reading['normal_percentage'] < 38:
            recommendation_html = """
            <div style="text-align: center; font-size: 20px; margin-top: 20px;">
                <strong style="color: red;">Alert:</strong> Water levels are low.<br>
                <strong style="color: blue;">Recommendation:</strong> Irrigation required in crop fields.
            </div>
            """
            st.markdown(recommendation_html, unsafe_allow_html=True)

        # Button to display the latest analyzed mask image
        if st.button("Show Latest Analyzed Mask Output"):
            latest_mask_image = get_latest_mask_image()
            if latest_mask_image:
                st.subheader("Latest Mask Output")
                st.image(Image.open(latest_mask_image), caption="Most Recently Analyzed Mask Output", use_column_width=True)
            else:
                st.write("No recent mask output image available.")
    else:
        st.write("No recent moisture readings available.")

elif options == "Historical Readings":
    st.title("Historical Monthly Trends")

    # Grouped Bar Chart for Monthly Averages
    st.subheader("Moisture Condition Trends - Monthly Averages (Grouped Bar Chart)")
    grouped_fig = go.Figure()

    grouped_fig.add_trace(go.Bar(
        x=monthly_df.index.strftime('%b %Y'), 
        y=monthly_df['dry_percentage'], 
        name='Dry Percentage', 
        marker_color='red', 
        width=0.4  
    ))
    grouped_fig.add_trace(go.Bar(
        x=monthly_df.index.strftime('%b %Y'), 
        y=monthly_df['normal_percentage'], 
        name='Normal Percentage', 
        marker_color='yellow', 
        width=0.4  
    ))
    grouped_fig.add_trace(go.Bar(
        x=monthly_df.index.strftime('%b %Y'), 
        y=monthly_df['wet_percentage'], 
        name='Wet Percentage', 
        marker_color='blue', 
        width=0.4  
    ))
    grouped_fig.update_layout(
        barmode='group',
        title="Monthly Average - Moisture Conditions",
        xaxis_title="Month",
        yaxis_title="Percentage (%)",
        xaxis_tickangle=-45,
        legend_title="Condition",
        bargap=0.15,
        bargroupgap=0.1,
    )
    st.plotly_chart(grouped_fig)

    # Adding animation for time-series data
    st.subheader("Animated Monthly Moisture Levels")
    animation_fig = px.bar(
        monthly_df.reset_index(), 
        x='date', 
        y=['dry_percentage', 'normal_percentage', 'wet_percentage'], 
        labels={'value': 'Percentage (%)', 'variable': 'Condition'},
        animation_frame='date',
        range_y=[0, 100]
    )
    animation_fig.update_layout(
        title="Animated Monthly Moisture Conditions Over Time",
        xaxis_title="Month",
        yaxis_title="Percentage (%)"
    )
    st.plotly_chart(animation_fig)

    # Display Data Table for Historical Readings with Conditional Formatting
    st.subheader("Detailed Historical Data Table")
    styled_df = raw_df.reset_index()[['date', 'dry_percentage', 'normal_percentage', 'wet_percentage', 'condition_summary']]
    styled_df = styled_df.style.background_gradient(cmap="coolwarm", subset=['dry_percentage', 'normal_percentage', 'wet_percentage'])
    st.dataframe(styled_df)

    # Download button for exporting data
    st.download_button(
        label="Download Filtered Data as CSV",
        data=raw_df.to_csv(index=True),
        file_name="historical_moisture_readings.csv",
        mime="text/csv"
    )
