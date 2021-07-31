import pandas as pd
import geopy.distance
import haversine as hs
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import dash
import dash_table
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import plotly.graph_objects as go
from datetime import datetime, timedelta


def route2read(name):
    readroute = pd.read_csv(name)
    for col in readroute.columns:
        readroute = readroute.rename(columns={col: col.strip()})
    return readroute

def trip2read(name):
    readtrip = pd.read_csv(name)
    # for each column
    for col in readtrip.columns:
        readtrip = readtrip.rename(columns={col: col.strip()})
    return readtrip

maxspeed=[]
maxspeed.append(50)
restroutespeed =[]
restroutespeed.append(30)

driverroutefile = 'finalroutelist.csv'
driverroute = route2read(driverroutefile)
df = driverroute.groupby(['assetid','driverid','drivername','seq']).agg({"lat":"mean","long":"mean",
                        "time":"mean","distance":"mean","speed":"mean","deviation":"mean"})
df.reset_index(inplace = True)

df['wait_violation']=0
df['night_violation']=0
df['restrict_route'] =0
df['delay']=0
df['asset_wait']=0

driver_column = df['driverid']
driver_column.drop_duplicates(inplace=True)
driver_column = driver_column.reset_index(drop=True)
waittime =700
nightstart1 = 1320
nightend1 = 1740
nightstart2 =300
nightend2 = 300

tripfile = 'jamshed2vishakhatrip.csv'
temp2trip = trip2read(tripfile)
temp2trip["dt_sub_trip_end_time"]= temp2trip["dt_sub_trip_end"].apply(lambda x: datetime.strptime(x.strip(),'%d-%b-%Y %H:%M:%S'))
temp2trip["dt_sub_trip_ata_out_time"]= temp2trip["dt_sub_trip_ata_out"].apply(lambda x: datetime.strptime(x.strip(),'%d-%b-%Y %H:%M:%S'))
temp2trip["delay"]=temp2trip["dt_sub_trip_end_time"]-temp2trip["dt_sub_trip_ata_out_time"]
temp2trip["delay"]= temp2trip["delay"].apply(lambda x: x.total_seconds()/60)
#mintime = mintime.total_seconds() / 60
iter=0
for drivername in driver_column:
    iter += 1
    loopstart = True
    driverdata = df[df['driverid']==drivername]
    #driverdata = driverdata.reset_index(drop = True)
    tripdelay = 0
    if drivername != 'bestdriverid':
        tripid = temp2trip[temp2trip['s_driver_cont']==int(drivername)]
        tripdelay = tripid['delay'].mean()
    length = 240
    for seq in driverdata.index:
        seq_length = (seq + 1)% length
        if seq_length !=0 | loopstart:
            loopstart=False
            timebnode = driverdata.at[seq+1,'time'] - driverdata.at[seq,'time']
            if timebnode > waittime and driverdata.at[seq+1,'distance'] - driverdata.at[seq,'distance'] <= (timebnode-waittime)*10 :
                driverdata.at[seq, 'wait_violation'] = 1
            else:
                driverdata.at[seq, 'wait_violation'] = 0

            no_day = int(driverdata.at[seq,'time']/1440)
            if (driverdata.at[seq,'time'] - no_day*1440>= nightstart1 and  driverdata.at[seq+1,'time'] - no_day*1440< nightend1) |\
                    (driverdata.at[seq,'time'] - no_day*1440>= 0 and  driverdata.at[seq+1,'time'] - no_day*1440< nightend2):
                if(driverdata.at[seq+1,'distance'] > driverdata.at[seq,'distance'] + 1):
                    driverdata.at[seq, 'night_violation'] = 1
                else:
                    driverdata.at[seq, 'night_violation'] = 0
            else:
                driverdata.at[seq, 'night_violation'] = 0
            res = seq % 5
            if res ==0:
                driverdata.at[seq,'restrict_route'] =1
            else:
                driverdata.at[seq, 'restrict_route'] = 0
                if timebnode > 0 and driverdata.at[seq+1, 'distance'] - driverdata.at[seq, 'distance']<= timebnode * 1:
                    driverdata.at[seq, 'asset_wait'] = timebnode *0.9
                else:
                    driverdata.at[seq, 'asset_wait'] = 0
    driverdata['delay']=tripdelay
    df[df['driverid']==drivername] = driverdata

rankdf = df
rankdf["speed"] = rankdf["speed"] - maxspeed[0]
rankdf["speed"] = rankdf["speed"].apply(lambda x: 0 if x < 0 else x)
rankdf["restrict_route"] = (rankdf["speed"] * rankdf["restrict_route"]) - restroutespeed[0]
rankdf["restrict_route"] = rankdf["restrict_route"].apply(lambda x: 0 if x < 0 else x)
driverdata = rankdf.groupby(['assetid', 'driverid', 'drivername']).agg({"lat": "mean", "long": "mean",
                                                                        "time": "mean", "distance": "mean",
                                                                        "speed": "mean", "deviation": "mean", "restrict_route":'mean',
                                                                            "wait_violation":"mean", "night_violation":"mean", "delay":"mean"})


driverdata = driverdata.reset_index()

assetdata = rankdf.groupby(['assetid', 'driverid', 'drivername']).agg({"time": ["mean",'min','max'], 'asset_wait':'sum'})
assetdata = assetdata.reset_index()
assetdata.columns =['assetid', 'driverid', 'drivername', 'mean_time', 'min_time','max_time', 'totalwait']
assetdata['totaltraveltime']=assetdata['max_time']-assetdata['min_time']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, suppress_callback_exceptions=True)
available_indicators = df['drivername'].unique()

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    html.H3('DSCONNECT: Data Science to Track, Predict and Decision to Evehicle', style={'color': 'blue'}),
    html.H5('Tracking: Extract realtime information to display current status', style={'color': 'blue', 'fontSize': 16}),
    html.H5('Predict: Provide future event (accident, truck burglary, delay) based on real time information',
            style={'color': 'blue', 'fontSize': 16}),
    html.H5('Decision: Strategy to improve vehicle utilization, cost reduction, and revenue maximization',
            style={'color': 'blue', 'fontSize': 16}),
    html.Br(),
    dcc.Link('Go to track page',  style={'color': 'Green', 'fontSize': 17},href='/page-1'),
    html.H4('Current Vehicle Status', style={'color': 'Green'}),
    html.Br(),
    dcc.Link('Go to driver ranking page', href='/page-2', style={'color': 'Green', 'fontSize': 17}),
    html.H4('Driver Ranking', style={'color': 'Green'}),
    dcc.Link('Go to utilization page', href='/page-3', style={'color': 'Green', 'fontSize': 17}),
    html.H4('Vehicle Utilization', style={'color': 'Green'}),
])


page_1_layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='bestdrivername'
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

    ], style={
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(id='crossfilter-indicator-scatter'),
        dcc.Graph(id='routedeviationgraph'),
        dcc.Graph(id='speeddeviationgraph'),
        dcc.Graph(id='restroutespeeddeviationgraph'),
        dcc.Graph(id='waitdeviationgraph'),
        dcc.Graph(id='nightdeviationgraph'),
    ], style={'width': '100%', 'display': 'inline-block'}),
    html.Br(),
    html.Br(),
    dcc.Link('Go to driver ranking page', style={'color': 'Green', 'fontSize': 17}, href='/page-2'),
    html.Br(),
    html.Br(),
    dcc.Link('Go to utilization page', style={'color': 'Green', 'fontSize': 17}, href='/page-3'),
    html.Br(),
    html.Br(),
    dcc.Link('Go back to home', style={'color': 'Green', 'fontSize': 17}, href='/')
])


@app.callback(
    [dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),dash.dependencies.Output('routedeviationgraph', 'figure'),
     dash.dependencies.Output('speeddeviationgraph', 'figure'),
     dash.dependencies.Output('restroutespeeddeviationgraph', 'figure'),
     dash.dependencies.Output('waitdeviationgraph', 'figure'),
     dash.dependencies.Output('nightdeviationgraph', 'figure')],
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value')])
def update_graph(xaxis_column_name):
    dff = df[df['drivername'] == xaxis_column_name]
    fig = go.Figure()
    dfbest = df[df['drivername'] == 'bestdrivername']
    latbest = dfbest["lat"]
    longbest = dfbest['long']
    ##fig.add_trace(go.Scattermapbox(lat=latbest, lon=longbest, name="bestroute", marker=go.scattermapbox.Marker(
      #  color='rgb(242, 177, 172)')))
    fig.add_trace(go.Scattermapbox(lat=latbest, lon=longbest, name="bestroute"))
    lattemp = dff["lat"]
    longtemp = dff['long']
    #fig =px.scatter_mapbox(dff, lat="lat", lon="long")
    fig.add_trace(go.Scattermapbox(lat=lattemp, lon=longtemp, name = "driverroute"))
    #fig.add_trace(px.scatter_mapbox(dfbest, lat="lat", lon="long"))
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=21,
                lon=80
            ),
            pitch=0,
            zoom=5
        )
    )

    fig_deviation = px.bar(dff, x="seq", y="deviation", title="Routedeviation")
    speed_deviation = dff["speed"] - maxspeed[0]
    speed_deviation = speed_deviation.apply(lambda x:0 if x <0 else x)
    fig_speed = px.bar(x=dff['seq'], y=speed_deviation, title="Speedviolation")
    restroute_speed = (dff["speed"]*dff["restrict_route"]) - restroutespeed[0]
    restroute_speed_deviation = restroute_speed.apply(lambda x: 0 if x < 0 else x)
    fig_rest_speed = px.bar(x=dff['seq'], y=restroute_speed_deviation, title="RestrouteSpeedviolation")
    fig_wait_deviation = px.bar(dff, x="seq", y="wait_violation", title="Waitviolation")
    fig_night_deviation = px.bar(dff, x="seq", y="night_violation", title="Nightdeviation")
    return fig, fig_deviation,fig_speed,fig_rest_speed,fig_wait_deviation,fig_night_deviation


rankparameter =[]
rankparameter.append([1, 1, 1, 1, 1, 1])
rankparameter = pd.DataFrame(rankparameter, columns=['RouteV', 'Delay', 'SpeedV','RestSpeedv', 'WaitV', 'NightV'])

page_2_layout = html.Div([
        html.H3('Driver Ranking Parameter Weight', style={'color': 'Blue'}),
        html.Div(dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i, 'deletable': True,
                      'renamable': True} for i in rankparameter.columns],
            data=rankparameter.to_dict('records'),
            style_cell={
                'height': 'auto',
                'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
                'whiteSpace': 'normal',
                'textAlign': 'center',
                'color': 'Brown'
            },
            editable=True  # ----->MAKES THE DATATABLE EDITABLE

        ), id="table_view"),
        html.Br(),
        html.Br(),
        html.Div([
        dcc.Graph(id='driverrankinggraph'),
        ], style={'width': '100%', 'display': 'inline-block'}),
        html.Br(),
        html.Br(),
        dcc.Link('Go to track page', style={'color': 'Green', 'fontSize': 17}, href='/page-1'),
        html.Br(),
        html.Br(),
        dcc.Link('Go to utilization page', style={'color': 'Green', 'fontSize': 17}, href='/page-3'),
        html.Br(),
        html.Br(),
        dcc.Link('Go back to home', style={'color': 'Green', 'fontSize': 17}, href='/')
    ]
    )

@app.callback(dash.dependencies.Output('driverrankinggraph', 'figure'),
              Input('table', 'data'))
def display_output(rows):
    # print(rows)
    rankweight = pd.DataFrame(rows)  # ----------->COLLECT NEW DATAFRAME
    tempval=int (rankweight.at[0,"SpeedV"])
    driverrank = driverdata
    driverrank["score"] = driverrank["speed"] *int(rankweight.at[0,"SpeedV"])+ driverrank["deviation"]*int(rankweight.at[0,"RouteV"])+\
                          driverrank["restrict_route"] * int(rankweight.at[0,"RestSpeedv"])+ driverrank["wait_violation"] * int(rankweight.at[0,"WaitV"])+ \
                          driverrank["night_violation"] * int(rankweight.at[0,"NightV"])+ driverrank["delay"]* int(rankweight.at[0,"Delay"])

    driverrank["score"] = driverrank["score"].mean()/(1+driverrank["score"])
    driverrank = driverrank.sort_values(by=['score'], ascending=False)
    driverrank = driverrank.loc[driverrank["driverid"] !="bestdriverid"]
    figranking = px.bar(driverrank, x="drivername", y="score", title="Routedeviation")
    return figranking


utilparameter =[]
utilparameter.append('BestAsset')
utilparameter.append('WaitTime')

page_3_layout = html.Div([
    html.H3('Vehicle Utilization', style={'color': 'Blue'}),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='util-column',
                options=[{'label': i, 'value': i} for i in utilparameter],
                value='bestdrivername'
            )
        ],
            style={'width': '49%', 'display': 'inline-block'}),

    ], style={
        'padding': '10px 5px'
    }),
    html.Br(),
    html.Br(),
    html.Div([
        dcc.Graph(id='utilizationgraph'),
    ], style={'width': '100%', 'display': 'inline-block'}),
    html.Br(),
    html.Br(),
    dcc.Link('Go to track page', style={'color': 'Green', 'fontSize': 17}, href='/page-1'),
    html.Br(),
    html.Br(),
    dcc.Link('Go to ranking page', style={'color': 'Green', 'fontSize': 17}, href='/page-2'),
    html.Br(),
    html.Br(),
    dcc.Link('Go back to home', style={'color': 'Green', 'fontSize': 17}, href='/')
    ]
    )

@app.callback(dash.dependencies.Output('utilizationgraph', 'figure'),
              [dash.dependencies.Input('util-column', 'value')])
def display_output(colname):
    assetutil = assetdata
    assetutil = assetutil.loc[assetutil["driverid"] !="bestdriverid"]
    if colname == 'BestAsset':
        assetutil['score'] = assetutil['totaltraveltime'].min()/assetutil['totaltraveltime']
    else:
        assetutil['score'] = 1 - assetutil['totalwait']/assetutil['totaltraveltime']

    assetutil = assetutil.sort_values(by=['score'], ascending=False)
    temp_asset = assetutil.groupby('assetid').agg({"score": "mean"})
    temp_asset =temp_asset.reset_index()
    temp_asset = temp_asset.sort_values(by=['score'], ascending=False)
    figutil = px.bar(temp_asset, x="assetid", y="score", title="AssetUtilization")
    return figutil


# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    elif pathname == '/page-3':
        return page_3_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here

if __name__ == '__main__':
    app.run_server(debug=True)
