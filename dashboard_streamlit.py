# # # import streamlit as st
# # # import pandas as pd
# # # import plotly.express as px
# # # from streamlit_folium import st_folium
# # # import folium
# # # from prophet import Prophet

# # # # ---
# # # # Dashboard: MTA Ridership Explorer
# # # # Dependencies:
# # # #   pip install streamlit plotly prophet streamlit-folium folium
# # # # ---

# # # @st.cache_data
# # # def load_data(path):
# # #     df = pd.read_csv(path, parse_dates=['transit_timestamp'])
# # #     return df

# # # @st.cache_data
# # # def prep_forecast(df_station):
# # #     # Prepare time series for Prophet
# # #     ts = (
# # #         df_station
# # #         .set_index('transit_timestamp')['ridership']
# # #         .resample('D').sum()
# # #         .reset_index()
# # #         .rename(columns={'transit_timestamp': 'ds', 'ridership': 'y'})
# # #     )
# # #     m = Prophet(daily_seasonality=True, weekly_seasonality=True)
# # #     m.fit(ts)
# # #     future = m.make_future_dataframe(periods=7)
# # #     forecast = m.predict(future)
# # #     return ts, forecast

# # # # Load master dataset
# # # df_master = load_data("master_df.csv")

# # # # Sidebar: station selection
# # # stations = df_master['station_complex'].unique()
# # # selected_station = st.sidebar.selectbox("Select a station", stations)

# # # df_station = df_master[df_master['station_complex'] == selected_station]

# # # # Main: station panels
# # # st.title(f"Ridership Dashboard: {selected_station}")

# # # # 1. Daily and Weekly Ridership
# # # st.header("Ridership Over Time")
# # # # Daily
# # # daily = df_station.set_index('transit_timestamp')['ridership'].resample('D').sum().reset_index()
# # # fig_daily = px.line(daily, x='transit_timestamp', y='ridership', title='Daily Ridership')
# # # st.plotly_chart(fig_daily, use_container_width=True)
# # # # Weekly
# # # weekly = df_station.set_index('transit_timestamp')['ridership'].resample('W').sum().reset_index()
# # # fig_weekly = px.line(weekly, x='transit_timestamp', y='ridership', title='Weekly Ridership')
# # # st.plotly_chart(fig_weekly, use_container_width=True)

# # # # 2. Map View
# # # st.header("Station Location")
# # # lat = df_station['latitude'].mean()
# # # lon = df_station['longitude'].mean()
# # # m = folium.Map(location=(lat, lon), zoom_start=14)
# # # folium.Marker((lat, lon), popup=selected_station).add_to(m)
# # # st_folium(m, width=700)

# # # # 3. Forecast Next Week
# # # st.header("7-day Forecast")
# # # with st.spinner("Training forecasting model..."):
# # #     ts, forecast = prep_forecast(df_station)
# # # fig_fc = px.line(forecast, x='ds', y='yhat', title='Forecast')
# # # # show actual vs forecast
# # # fig_fc.add_scatter(x=ts['ds'], y=ts['y'], mode='markers', name='Actual')
# # # st.plotly_chart(fig_fc, use_container_width=True)

# # # # Borough-level tabs
# # # st.header("Borough-level Summary")
# # # boroughs = df_master['borough'].unique()
# # # tabs = st.tabs(df_master['borough'].unique().tolist())

# # # # tabs = st.tabs(boroughs)
# # # for tab, b in zip(tabs, boroughs):
# # #     with tab:
# # #         df_b = df_master[df_master['borough'] == b]
# # #         st.markdown(f"**{b}**: Stations: {df_b['station_complex'].nunique()}  ")
# # #         # aggregate ridership by station
# # #         top = (
# # #             df_b.groupby('station_complex')['ridership']
# # #             .sum().nlargest(5).reset_index()
# # #         )
# # #         fig_top = px.bar(top, x='station_complex', y='ridership', title=f'Top 5 Stations in {b}')
# # #         st.plotly_chart(fig_top, use_container_width=True)








# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from streamlit_folium import st_folium
# import folium
# from prophet import Prophet

# # ---
# # Dashboard: MTA Ridership Explorer
# # Dependencies:
# #   pip install streamlit plotly prophet streamlit-folium folium
# # ---

# @st.cache_data
# def load_data(path):
#     df = pd.read_csv(path, parse_dates=['transit_timestamp'])
#     return df

# @st.cache_data
# def prep_forecast(df_station):
#     # Prepare time series for Prophet
#     ts = (
#         df_station
#         .set_index('transit_timestamp')['ridership']
#         .resample('D').sum()
#         .reset_index()
#         .rename(columns={'transit_timestamp': 'ds', 'ridership': 'y'})
#     )
#     m = Prophet(daily_seasonality=True, weekly_seasonality=True)
#     m.fit(ts)
#     future = m.make_future_dataframe(periods=7)
#     forecast = m.predict(future)
#     return ts, forecast

# # Load master dataset
# df_master = load_data("master_df.csv")

# # Sidebar: station selection
# stations = df_master['station_complex'].unique()
# selected_station = st.sidebar.selectbox("Select a station", stations)

# df_station = df_master[df_master['station_complex'] == selected_station]

# # Main: station panels
# st.title(f"Ridership Dashboard: {selected_station}")

# # 1. Daily and Weekly Ridership
# st.header("Ridership Over Time")
# # Daily
# daily = df_station.set_index('transit_timestamp')['ridership'].resample('D').sum().reset_index()
# fig_daily = px.line(daily, x='transit_timestamp', y='ridership', title='Daily Ridership')
# fig_daily.update_layout(xaxis_title='Date', yaxis_title='Ridership')
# st.plotly_chart(fig_daily, use_container_width=True)
# # Weekly
# weekly = df_station.set_index('transit_timestamp')['ridership'].resample('W').sum().reset_index()
# fig_weekly = px.line(weekly, x='transit_timestamp', y='ridership', title='Weekly Ridership')
# fig_weekly.update_layout(xaxis_title='Date', yaxis_title='Ridership')
# st.plotly_chart(fig_weekly, use_container_width=True)

# # 2. Map View
# st.header("Station Location")
# lat = df_station['latitude'].mean()
# lon = df_station['longitude'].mean()
# m = folium.Map(location=(lat, lon), zoom_start=14)
# folium.CircleMarker((lat, lon), radius=8, popup=selected_station, color='blue', fill=True).add_to(m)
# st_folium(m, width=700)

# # 3. Forecast Next Week
# st.header("7-day Forecast")
# with st.spinner("Training forecasting model..."):
#     ts, forecast = prep_forecast(df_station)
# fig_fc = px.line(forecast, x='ds', y='yhat', title='7-day Forecast')
# # show actual vs forecast
# fig_fc.add_scatter(x=ts['ds'], y=ts['y'], mode='markers', name='Actual')
# fig_fc.update_layout(
#     xaxis_title='Transit TimeStamp',
#     yaxis_title='No of people'
# )
# st.plotly_chart(fig_fc, use_container_width=True)

# # Borough-level tabs
# st.header("Borough-level Summary")
# boroughs = df_master['borough'].unique()
# tabs = st.tabs(boroughs.tolist())
# for tab, b in zip(tabs, boroughs):
#     with tab:
#         df_b = df_master[df_master['borough'] == b]
#         st.markdown(f"**{b}**: {df_b['station_complex'].nunique()} stations")
#         # allow station-specific analysis per borough
#         stations_b = df_b['station_complex'].unique()
#         sel = st.multiselect(f"Select station(s) in {b}", stations_b, default=stations_b[:3])
#         if sel:
#             df_sel = df_b[df_b['station_complex'].isin(sel)]
#             # plot daily ridership for selected stations
#             df_plot = (
#                 df_sel.set_index('transit_timestamp')
#                 .groupby('station_complex')['ridership']
#                 .resample('D').sum()
#                 .reset_index()
#             )
#             fig = px.line(df_plot, x='transit_timestamp', y='ridership', color='station_complex',
#                           title=f'Daily Ridership Trends in {b}')
#             fig.update_layout(xaxis_title='Date', yaxis_title='Ridership')
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info('Select at least one station to view its ridership trends.')

# # 4. Additional Insights
# st.header("Additional Insights")
# # Ridership by Hour of Day
# hourly = df_master.groupby('hour')['ridership'].sum().reset_index()
# fig_hour = px.bar(hourly, x='hour', y='ridership', title='Total Ridership by Hour')
# fig_hour.update_layout(xaxis_title='Hour of Day', yaxis_title='Ridership')
# st.plotly_chart(fig_hour, use_container_width=True)

# # Payment Method Distribution
# pay_dist = df_master['payment_method'].value_counts().reset_index()
# pay_dist.columns = ['payment_method', 'count']
# fig_pay = px.pie(pay_dist, names='payment_method', values='count', title='Payment Method Breakdown')
# st.plotly_chart(fig_pay, use_container_width=True)

# # Ridership vs Distance to Central
# scatter = df_master.sample(min(len(df_master), 2000))  # sample for performance
# fig_scatter = px.scatter(
#     scatter, x='distance_to_central', y='ridership', color='borough',
#     title='Ridership vs Distance to Central'
# )
# fig_scatter.update_layout(xaxis_title='Distance to Central (m)', yaxis_title='Ridership')
# st.plotly_chart(fig_scatter, use_container_width=True)




# ##########################
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from streamlit_folium import st_folium
# import folium
# from prophet import Prophet
# from statsmodels.tsa.seasonal import seasonal_decompose
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# import json

# # --- CACHING DATA LOADING & MODEL FITS ---------------------------------------

# @st.cache_data
# def load_data(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path, parse_dates=['transit_timestamp'])
#     df['hour'] = df['transit_timestamp'].dt.hour
#     return df

# @st.cache_data
# def long_term_prophet(df: pd.DataFrame, years: int = 5):
#     ts = (
#         df
#         .set_index('transit_timestamp')['ridership']
#         .resample('D').sum()
#         .reset_index()
#         .rename(columns={'transit_timestamp':'ds','ridership':'y'})
#     )
#     m = Prophet(daily_seasonality=False,
#                 weekly_seasonality=True,
#                 yearly_seasonality=True)
#     m.fit(ts)
#     future = m.make_future_dataframe(periods=365*years)
#     forecast = m.predict(future)
#     return ts, forecast

# @st.cache_data
# def xgb_long_forecast(df: pd.DataFrame, years: int = 5):
#     df2 = (
#         df
#         .set_index('transit_timestamp')['ridership']
#         .resample('D').sum()
#         .rename('y')
#     )
#     df2 = df2.to_frame()
#     df2['dayofweek']  = df2.index.dayofweek
#     df2['dayofyear']  = df2.index.dayofyear
#     df2['lag_1']      = df2['y'].shift(1)
#     df2['rolling7']   = df2['y'].rolling(7).mean()
#     df2 = df2.dropna()
#     X = df2[['dayofweek','dayofyear','lag_1','rolling7']]
#     y = df2['y']
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
#     model = xgb.XGBRegressor(tree_method='hist', n_estimators=200)
#     model.fit(X_train, y_train)
#     # iterative forecast
#     last = df2.copy()
#     future_idx = pd.date_range(last.index[-1] + pd.Timedelta(1, 'D'),
#                                periods=365*years, freq='D')
#     preds = []
#     for day in future_idx:
#         feat = pd.DataFrame({
#             'dayofweek': [day.dayofweek],
#             'dayofyear': [day.dayofyear],
#             'lag_1': [ last['y'].iloc[-1] ],
#             'rolling7': [ last['y'].iloc[-7:].mean() ]
#         })
#         p = model.predict(feat)[0]
#         preds.append(p)
#         last.loc[day] = [p, day.dayofweek, day.dayofyear, p, last['y'].shift(1).rolling(7).mean().iloc[-1]]
#     future = pd.DataFrame({'ds': future_idx, 'yhat': preds})
#     hist = df2.reset_index().rename(columns={'index':'ds','y':'y'})
#     return hist, future

# # --- APP LAYOUT --------------------------------------------------------------

# st.set_page_config(layout='wide')
# st.title("🚇 MTA Ridership Explorer")

# # --- SIDEBAR CONTROLS --------------------------------------------------------

# df_master = load_data("master_df.csv")

# min_date = df_master['transit_timestamp'].dt.date.min()
# max_date = df_master['transit_timestamp'].dt.date.max()
# date_range = st.sidebar.date_input("Date range", [min_date, max_date],
#                                    min_value=min_date, max_value=max_date)

# filtered = df_master[
#     (df_master['transit_timestamp'].dt.date >= date_range[0]) &
#     (df_master['transit_timestamp'].dt.date <= date_range[1])
# ]

# stations = filtered['station_complex'].unique()
# sel_stations = st.sidebar.multiselect("Select station(s)", stations,
#                                       default=[stations[0]])
# compare_mode = len(sel_stations) > 1

# # export filtered data
# csv = filtered.to_csv(index=False)
# st.sidebar.download_button("Download filtered CSV", csv, "ridership.csv")

# # load borough shapes
# with open("boroughs.geojson") as f:
#     boroughs_geo = json.load(f)

# # --- METRIC CARDS ------------------------------------------------------------

# st.header("Key Metrics")
# cols = st.columns(len(sel_stations))
# for col, stn in zip(cols, sel_stations):
#     df_s = filtered[filtered['station_complex']==stn]
#     total = int(df_s['ridership'].sum())
#     avg_daily = df_s.set_index('transit_timestamp')['ridership'] \
#                     .resample('D').sum().mean()
#     col.metric(f"{stn}", f"Total: {total:,}", f"Avg/day: {avg_daily:,.0f}")

# # --- TIME SERIES TRENDS ------------------------------------------------------

# st.subheader("Ridership Over Time")
# df_plot = (
#     filtered[filtered['station_complex'].isin(sel_stations)]
#     .set_index('transit_timestamp')
#     .groupby('station_complex')['ridership']
#     .resample('D').sum()
#     .reset_index()
# )
# fig = px.line(df_plot, x='transit_timestamp', y='ridership',
#               color='station_complex',
#               title="Daily Ridership")
# fig.update_layout(xaxis_title='Date', yaxis_title='Ridership')
# st.plotly_chart(fig, use_container_width=True)

# # --- CHOROPLETH BY BOROUGH ---------------------------------------------------

# st.subheader("Total Ridership by Borough")
# agg = filtered.groupby('borough')['ridership'].sum().reset_index()
# m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
# folium.Choropleth(
#     geo_data=boroughs_geo,
#     data=agg,
#     columns=['borough','ridership'],
#     key_on='feature.properties.boro_name',
#     fill_opacity=0.7,
#     line_opacity=0.2,
#     legend_name='Ridership'
# ).add_to(m)
# st_folium(m, width=700)

# # --- SEASONAL DECOMPOSITION --------------------------------------------------

# st.subheader("Seasonal Decomposition (Selected Station)")
# if len(sel_stations)==1:
#     ts = (
#         filtered[filtered['station_complex']==sel_stations[0]]
#         .set_index('transit_timestamp')['ridership']
#         .resample('D').sum()
#     )
#     dec = seasonal_decompose(ts, model='additive', period=365)
#     for name, series in zip(["trend","seasonal","resid"], [dec.trend,dec.seasonal,dec.resid]):
#         dfc = series.reset_index().rename(columns={'transit_timestamp':'ds', 0:'value'})
#         figc = px.line(dfc, x='ds', y=series.name, title=name.capitalize())
#         figc.update_layout(xaxis_title='Date', yaxis_title=name.capitalize())
#         st.plotly_chart(figc, use_container_width=True)
# else:
#     st.info("Pick a single station for decomposition.")

# # --- 7-DAY & 5-YEAR FORECAST (PROPHET) ---------------------------------------

# st.subheader("Forecasting")
# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("*7-day* (short-term)")
#     ts7, fc7 = long_term_prophet(filtered[filtered['station_complex']==sel_stations[0]], years=0)
#     fig7 = px.line(fc7, x='ds', y='yhat', title="7-day Prophet Forecast")
#     fig7.add_scatter(x=ts7['ds'], y=ts7['y'], mode='markers', name='Actual')
#     fig7.update_layout(xaxis_title='Date', yaxis_title='Ridership')
#     st.plotly_chart(fig7, use_container_width=True)

# with col2:
#     st.markdown("*5-year* (long-term)")
#     ts5, fc5 = long_term_prophet(filtered[filtered['station_complex']==sel_stations[0]], years=5)
#     fig5 = px.line(fc5, x='ds', y='yhat', title="5-year Prophet Forecast")
#     fig5.add_scatter(x=ts5['ds'], y=ts5['y'], mode='markers', name='Actual')
#     fig5.update_layout(xaxis_title='Date', yaxis_title='Ridership')
#     st.plotly_chart(fig5, use_container_width=True)

# # --- OPTIONAL: 5-YEAR FORECAST (XGBOOST) --------------------------------------

# st.subheader("5-year XGBoost Forecast")
# hist, xgb_fc = xgb_long_forecast(filtered[filtered['station_complex']==sel_stations[0]], years=5)
# figx = px.line(xgb_fc, x='ds', y='yhat', title="XGBoost Forecast")
# figx.add_scatter(x=hist['ds'], y=hist['y'], mode='markers', name='Actual')
# figx.update_layout(xaxis_title='Date', yaxis_title='Ridership')
# st.plotly_chart(figx, use_container_width=True)

# # --- ADDITIONAL INSIGHTS ------------------------------------------------------

# st.subheader("Ridership by Hour of Day")
# hourly = filtered.groupby('hour')['ridership'].sum().reset_index()
# fig_h = px.bar(hourly, x='hour', y='ridership', title="By Hour")
# fig_h.update_layout(xaxis_title='Hour', yaxis_title='Ridership')
# st.plotly_chart(fig_h, use_container_width=True)

# if 'payment_method' in filtered.columns:
#     st.subheader("Payment Method Breakdown")
#     pm = filtered['payment_method'].value_counts().reset_index()
#     pm.columns = ['method','count']
#     fig_p = px.pie(pm, names='method', values='count')
#     st.plotly_chart(fig_p, use_container_width=True)

# if 'distance_to_central' in filtered.columns:
#     st.subheader("Ridership vs Distance to Central")
#     samp = filtered.sample(min(len(filtered), 10000))
#     fig_s = px.scatter(samp, x='distance_to_central', y='ridership',
#                        color='borough',
#                        title="Ridership vs Distance")
#     st.plotly_chart(fig_s, use_container_width=True)

import streamlit as st
st.set_page_config(page_title="🚇 MTA Ridership Explorer", layout='wide')

import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
import folium
from prophet import Prophet
import json

# ---
# Dashboard: MTA Ridership Explorer
# Dependencies:
#   pip install streamlit pandas plotly streamlit-folium folium prophet
# ---

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['transit_timestamp'])
    df['hour'] = df['transit_timestamp'].dt.hour
    return df

@st.cache_data
def prep_forecast(df_station: pd.DataFrame, days: int = 7):
    ts = (
        df_station
        .set_index('transit_timestamp')['ridership']
        .resample('D').sum()
        .reset_index()
        .rename(columns={'transit_timestamp': 'ds', 'ridership': 'y'})
    )
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.fit(ts)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return ts, forecast

# Load and filter data
df_master = load_data("master_df.csv")
min_date = df_master['transit_timestamp'].dt.date.min()
max_date = df_master['transit_timestamp'].dt.date.max()
date_range = st.sidebar.date_input(
    "Date range", [min_date, max_date],
    min_value=min_date, max_value=max_date
)
filtered = df_master.loc[
    (df_master['transit_timestamp'].dt.date >= date_range[0]) &
    (df_master['transit_timestamp'].dt.date <= date_range[1])
]
stations = filtered['station_complex'].unique().tolist()
selected_station = st.sidebar.selectbox("Select station", stations)
df_station = filtered[filtered['station_complex'] == selected_station]

# App title
st.title(f"🚇 MTA Ridership Explorer: {selected_station}")

# 1. Hourly Ridership
st.header("Ridership Over Time")
hourly = df_station.groupby('hour')['ridership'].sum().reset_index()
fig_h = px.bar(hourly, x='hour', y='ridership', title='Hourly Ridership',
               color='hour', color_continuous_scale='Viridis')
fig_h.update_layout(xaxis_title='Hour', yaxis_title='Ridership')
st.plotly_chart(fig_h, use_container_width=True)

# 2. Daily & Weekly Ridership

daily = df_station.set_index('transit_timestamp')['ridership'].resample('D').sum().reset_index()
weekly = df_station.set_index('transit_timestamp')['ridership'].resample('W').sum().reset_index()

fig_daily = px.line(daily, x='transit_timestamp', y='ridership', title='Daily Ridership',
                    color_discrete_sequence=['#636EFA'])
fig_daily.update_layout(xaxis_title='Date', yaxis_title='Ridership')
st.plotly_chart(fig_daily, use_container_width=True)

fig_weekly = px.line(weekly, x='transit_timestamp', y='ridership', title='Weekly Ridership',
                     color_discrete_sequence=['#EF553B'])
fig_weekly.update_layout(xaxis_title='Date', yaxis_title='Ridership')
st.plotly_chart(fig_weekly, use_container_width=True)

# 3. Station Location
st.header("Station Location")
lat, lon = df_station['latitude'].mean(), df_station['longitude'].mean()
m = folium.Map(location=(lat, lon), zoom_start=14)
folium.CircleMarker((lat, lon), radius=10, popup=selected_station,
                    color='darkblue', fill=True, fill_color='skyblue').add_to(m)
st_folium(m, width=700)

# 4. 7-day Forecast
st.header("7-day Forecast")
with st.spinner("Training forecasting model..."):
    ts, forecast = prep_forecast(df_station)
fig_fc = px.line(forecast, x='ds', y='yhat', title='7-day Forecast',
                 color_discrete_sequence=['#00CC96'])
fig_fc.add_scatter(x=ts['ds'], y=ts['y'], mode='markers', name='Actual', marker_color='black')
fig_fc.update_layout(xaxis_title='Date', yaxis_title='Ridership')
st.plotly_chart(fig_fc, use_container_width=True)

# 5. Borough-level Summary
st.header("Borough-level Summary")
boroughs = filtered['borough'].unique().tolist()
tabs = st.tabs(boroughs)
for tab, b in zip(tabs, boroughs):
    with tab:
        df_b = filtered[filtered['borough'] == b]
        st.markdown(f"**{b}** ({df_b['station_complex'].nunique()} stations)")
        stations_b = df_b['station_complex'].unique().tolist()
        sel = st.multiselect(f"Stations in {b}", stations_b, default=stations_b[:3])
        if sel:
            df_sel = df_b[df_b['station_complex'].isin(sel)]
            df_plot = (
                df_sel.set_index('transit_timestamp')
                      .groupby('station_complex')['ridership']
                      .resample('D').sum().reset_index()
            )
            fig = px.line(df_plot, x='transit_timestamp', y='ridership',
                          color='station_complex', title=f'Daily Trends in {b}',
                          color_discrete_sequence=px.colors.qualitative.T10)
            fig.update_layout(xaxis_title='Date', yaxis_title='Ridership')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one station to view trends.")

# 6. Payment & Fare Category Breakdown
if 'payment_method' in df_station.columns and 'fare_class_category' in df_station.columns:
    rev_df = (
        df_station.groupby(['payment_method', 'fare_class_category'])['ridership']
                  .sum().reset_index(name='revenue')
    )
    fig_sf = px.sunburst(
        rev_df,
        path=['payment_method', 'fare_class_category'],
        values='revenue',
        color='payment_method',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title='Revenue by Payment Method & Fare Category'
    )
    fig_sf.update_traces(textinfo='label+percent entry', insidetextorientation='radial')
    fig_sf.update_layout(width=800, height=600, margin=dict(t=50, l=0, r=0, b=0))
    st.plotly_chart(fig_sf, use_container_width=False)

# 7. Ridership vs Distance to Central
if 'distance_to_central' in df_station.columns:
    samp = df_station.sample(min(len(df_station), 2000))
    fig_sc = px.scatter(samp, x='distance_to_central', y='ridership',
                        color='borough', title='Ridership vs Distance to Central',
                        color_discrete_sequence=px.colors.qualitative.Dark24)
    fig_sc.update_layout(xaxis_title='Distance (m)', yaxis_title='Ridership')
    st.plotly_chart(fig_sc, use_container_width=True)

# Note: order of graphs preserved: Hourly, Daily, Weekly, Map, Forecast, Borough, Payment, Distance
