import plotly.graph_objects as go
import pandas as pd
import numpy as np
import backend.queries   as u


def cpv_box(bot_year, top_year, country_list):
    try:
        boxes = u.ex1_cpv_box(bot_year, top_year, country_list)

        if len(boxes) != 5:
            raise Exception("Invalid amoung of values")
        for box in boxes:
            if not box:
                raise Exception("Invalid amoung of values")

        return int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])

    except:
        return ['-'] * 5  


def cpv_treemap(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex2_cpv_treemap(bot_year, top_year, country_list))

        labels = df['cpv']

        values = df['count']

        parents = ['Europe' for _ in range(len(df))]

        data = dict(type='treemap',
                    labels=labels,
                    parents=parents,
                    values=values,
                    text=values,
                    marker_colorscale='RdBu',
                    hovertemplate='<b>%{label} </b> <br> Contract Counts: %{text}<br> <b> Representing %{percentRoot:.0%} of total Count Value </b>',
                    name='CPV Treemap'
                    )

        layout = dict(margin=dict(t=0, b=0, l=0, r=0),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)'
                      )

        fig = go.Figure(data=data, layout=layout)

        return fig

    except:
        return gone_wrong()


def cpv_bar_1(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex3_cpv_bar_1(bot_year, top_year, country_list))
        df = df.iloc[::-1]
        data = dict(type='bar',
                    x=df['avg'],
                    y=df['cpv'],
                    text=df['cpv'],
                    orientation='h',
                    textposition='inside',
                    marker_color='#003399'
                    )

        layout = dict(yaxis=dict(title='Most', visible=False),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=0, b=0, l=0, r=0)
                      )

        return go.Figure(data=data, layout=layout)

    except:
        return gone_wrong()


def cpv_bar_2(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex4_cpv_bar_2(bot_year, top_year, country_list))

        data = dict(type='bar',
                    x=df['avg'],
                    y=df['cpv'],
                    text=df['cpv'],
                    textposition='inside',
                    orientation='h',
                    marker_color='#003399'

                    )

        layout = dict(yaxis=dict(title='Least', visible=False),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=0, b=0, l=0, r=0)
                      )

        return go.Figure(data=data, layout=layout)

    except:
        return gone_wrong()


def cpv_map(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex7_cpv_map(bot_year, top_year, country_list))

        country_name = df['country']

        values = df['avg']

        text = df['cpv']

        data = dict(type='choropleth',
                    locations=country_name,
                    # There are three ways to 'merge' your data with the data pre embedded in the map
                    locationmode='country names',
                    z=values,
                    text=text,
                    colorscale='Viridis'
                    )

        layout = dict(
            geo=dict(
                scope='europe',  # default
                projection=dict(type='equirectangular'
                                ),
                # showland=True,   # default = True
                landcolor='white',
                lakecolor='#E5E5E5',
                showocean=True,  # default = False
                oceancolor='#E5E5E5',
                countrycolor='#E5E5E5'
            ),
            dragmode=False,
            title=dict(
                text='Europe',
                x=.5  # Title relative position according to the xaxis, range (0,1)
            ),
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'

        )

        fig = go.Figure(data=data, layout=layout)

        return fig

    except:
        return gone_wrong()


def cpv_histogram(bot_year, top_year, country_list, cpv):
    try:
        df = pd.DataFrame(u.ex8_cpv_hist(bot_year, top_year, country_list, cpv))

        df = df.loc[df['bucket']!='Other']

        x = np.append(df['bucket'].values, [np.inf])

        b = [str([x[i], x[i+1]]) for i in range(len(x)-1)]

        y = df['count']

        data = dict(type='bar',
                    x=b,
                    y=y,
                    marker_color='#003399'

                    )

        layout = dict(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=0, b=0, l=0, r=0)
                      )

        return go.Figure(data=data, layout=layout)

    except:
        return gone_wrong()


def cpv_bar_3(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex5_cpv_bar_3(bot_year, top_year, country_list))
        df = df.iloc[::-1]
        data = dict(type='bar',
                    x=df['avg'],
                    y=df['cpv'],
                    text=df['cpv'],
                    textposition='inside',
                    orientation='h',
                    marker_color='#003399'

                    )

        layout = dict(yaxis=dict(visible=False),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=0, b=0, l=0, r=0)
                      )

        return go.Figure(data=data, layout=layout)

    except:
        return gone_wrong()


def cpv_bar_4(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex6_cpv_bar_4(bot_year, top_year, country_list))
        df = df.iloc[::-1]
        data = dict(type='bar',
                    x=df['avg'],
                    y=df['cpv'],
                    text=df['cpv'],
                    textposition='inside',
                    orientation='h',
                    marker_color='#003399'

                    )

        layout = dict(yaxis=dict(visible=False),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=0, b=0, l=0, r=0)

                      )

        return go.Figure(data=data, layout=layout)

    except:
        return gone_wrong()


def cpv_bar_diff(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex9_cpv_bar_diff(bot_year, top_year, country_list))

        data_1 = dict(type='bar',
                      y=df['time_difference'],
                      x=df['cpv'],
                      yaxis='y2',
                      opacity=.3,
                      name='Time Difference (Days)'

                      )

        data_2 = dict(type='bar',
                      y=df['value_difference'],
                      x=df['cpv'],
                      orientation='v',
                      name='Value Difference (Euros)'
                      )
        layout = dict(
            barmode='overlay',
            margin=dict(t=50, b=0, l=0, r=50),
            yaxis={'title': 'Value Difference (Euros)'},
            yaxis2={'title': 'Time Difference (Days)',
                    'overlaying': 'y1',
                    'side': 'right'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=1.1, y=.6)
        )

        return go.Figure(data=[data_1, data_2], layout=layout)

    except:
        return gone_wrong()


def country_box(bot_year, top_year, country_list):
    try:
        boxes = u.ex10_country_box(bot_year, top_year, country_list)

        if len(boxes) != 5:
            raise Exception("Invalid amoung of values")
        for box in boxes:
            if not box:
                raise Exception("Invalid amoung of values")

        return int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])

    except:
        return ['-'] * 5  



def country_treemap(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex11_country_treemap(bot_year, top_year, country_list))

        labels = df['country']

        values = df['count']

        parents = ['Europe' for _ in range(len(df))]

        data = dict(type='treemap',
                    labels=labels,
                    parents=parents,
                    values=values,
                    text=values,
                    marker_colorscale='RdBu',
                    hovertemplate='<b>%{label} </b> <br> Contract Counts: %{text}<br> <b> Representing %{percentRoot:.0%} of total Count Value </b>',
                    name='Country Treemap'
                    )

        layout = dict(margin=dict(t=0, b=0, l=0, r=0),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)'
                      )

        fig = go.Figure(data=data, layout=layout)

        return fig

    except:
        return gone_wrong()


def country_bar_1(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex12_country_bar_1(bot_year, top_year, country_list))

        df = df.iloc[::-1]

        data = dict(type='bar',
                    x=df['avg'],
                    y=df['country'],
                    text=df['country'],
                    textposition='inside',
                    orientation='h',
                    marker_color='#003399'

                    )

        layout = dict(yaxis=dict(visible=False),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=0, b=0, l=0, r=0)

                      )

        return go.Figure(data=data, layout=layout)

    except:
        return gone_wrong()


def country_bar_2(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex13_country_bar_2(bot_year, top_year, country_list))

        data = dict(type='bar',
                    x=df['avg'],
                    y=df['country'],
                    text=df['country'],
                    textposition='inside',
                    orientation='h',
                    marker_color='#003399'

                    )

        layout = dict(yaxis=dict(visible=False),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=0, b=0, l=0, r=0)

                      )

        return go.Figure(data=data, layout=layout)

    except:
        return gone_wrong()


def country_map(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex14_country_map(bot_year, top_year, country_list))

        df = df.dropna()

        country_name = df['country']

        values = df['sum']

        text = df['country']

        data = dict(type='choropleth',
                    locations=country_name,
                    locationmode='country names',
                    z=values,
                    text=text,
                    colorscale='Viridis'
                    )

        layout = dict(
            geo=dict(
                scope='europe',  # default
                projection=dict(type='equirectangular'
                                ),
                # showland=True,   # default = True
                landcolor='white',
                lakecolor='#E5E5E5',
                showocean=True,  # default = False
                oceancolor='#E5E5E5',
                countrycolor='#E5E5E5'
            ),
            dragmode=False,
            title=dict(
                text='Europe',
                x=.5  # Title relative position according to the xaxis, range (0,1)
            ),
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'

        )

        fig = go.Figure(data=data, layout=layout
                      )

        return fig

    except:
        return gone_wrong()


def business_box(bot_year, top_year, country_list):
    try:
        boxes = u.ex15_business_box(bot_year, top_year, country_list)

        if len(boxes) != 5:
            raise Exception("Invalid amoung of values")
        for box in boxes:
            if not box:
                raise Exception("Invalid amoung of values")

        return int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])

    except:
        return ['-'] * 5  



def business_bar_1(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex16_business_bar_1(bot_year, top_year, country_list))
        df = df.iloc[::-1]
        data = dict(type='bar',
                    x=df['avg'],
                    y=df['company'],
                    text=df['company'],
                    textposition='inside',
                    orientation='h',
                    marker_color='#003399'

                    )

        layout = dict(yaxis=dict(visible=False),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=0, b=0, l=0, r=0)

                      )

        return go.Figure(data=data, layout=layout)

    except:
        return gone_wrong()


def business_bar_2(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex17_business_bar_2(bot_year, top_year, country_list))

        data = dict(type='bar',
                    x=df['avg'],
                    y=df['company'],
                    text=df['company'],
                    textposition='inside',
                    orientation='h',
                    marker_color='#003399'

                    )

        layout = dict(yaxis=dict(visible=False),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=0, b=0, l=0, r=0)

                      )

        return go.Figure(data=data, layout=layout)

    except:
        return gone_wrong()


def business_treemap(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex18_business_treemap(bot_year, top_year, country_list))

        labels = df['company']

        values = df['count']

        parents = ['Europe' for _ in range(len(df))]

        data = dict(type='treemap',
                    labels=labels,
                    parents=parents,
                    values=values,
                    text=values,
                    marker_colorscale='RdBu',
                    hovertemplate='<b>%{label} </b> <br> Contract Counts: %{text}<br> <b> Representing %{percentRoot:.0%} of total Count Value </b>',
                    name='Company Treemap'
                    )

        layout = dict(margin=dict(t=0, b=0, l=0, r=0),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)'
                      )

        fig = go.Figure(data=data, layout=layout)

        return fig

    except:
        return gone_wrong()


def business_map(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex19_business_map(bot_year, top_year, country_list))

        country_name = df['country']

        company=df['company']

        values = df['sum']

        address = df['address']

        data = dict(type='choropleth',
                    locations=country_name,
                    # There are three ways to 'merge' your data with the data pre embedded in the map
                    locationmode='country names',
                    z=values,
                    text=address,
                    colorscale='Viridis',
                    customdata=company,
                    hovertemplate='<b>%{location} </b> <br> Top Company: %{customdata}<br> <b> Address: %{text}</b>'
                    )

        layout = dict(
            geo=dict(
                scope='europe',  # default
                projection=dict(type='equirectangular'
                                ),
                # showland=True,   # default = True
                landcolor='white',
                lakecolor='#E5E5E5',
                showocean=True,  # default = False
                oceancolor='#E5E5E5',
                countrycolor='#E5E5E5'
            ),
            dragmode=False,
            title=dict(
                text='Europe',
                x=.5  # Title relative position according to the xaxis, range (0,1)
            ),
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'

        )

        fig = go.Figure(data=data, layout=layout)

        return fig

    except:
        return gone_wrong()


def business_connection(bot_year, top_year, country_list):
    try:
        df = pd.DataFrame(u.ex20_business_connection(bot_year, top_year, country_list))

        data = dict(type='bar',
                    x=df['count'],
                    y=df['companies'],
                    orientation='h',

                    )

        layout = dict(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)'
                      )

        return go.Figure(data=data, layout=layout)

    except:
        return gone_wrong()


def gone_wrong():
    data = dict(type='indicator',
                mode='number',  # you can define the three indicators together here
                domain=dict(x=[0, 1], y=[0, 1]),
                title=dict(text='Invalid Query', font=dict(size=50, color='#003399')),

                number=dict(prefix='Invalid Query', font=dict(color='#003399')),
                )

    layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )

    return go.Figure(data=data, layout=layout)
