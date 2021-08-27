#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from cuml.decomposition import PCA
import scanpy as sc

import cudf
import cupy as cp

import plotly.graph_objects as go
import dash
from flask import request
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]

colors = ["#406278", "#e32636", "#9966cc", "#cd9575", "#915c83", "#008000",
        "#ff9966", "#848482", "#8a2be2", "#de5d83", "#800020", "#e97451",
        "#5f9ea0", "#36454f", "#008b8b", "#e9692c", "#f0b98d", "#ef9708",
        "#0fcfc0", "#9cded6", "#d5eae7", "#f3e1eb", "#f6c4e1", "#f79cd4"]

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

main_fig_height = 700

class Visualization:

    def __init__(self, adata, markers,
                 re_cluster_callback=None,
                 n_components=50,
                 n_neighbors=50,
                 knn_n_pcs=50,
                 umap_min_dist = 0.3,
                 umap_spread = 1.0,
                 leiden_resolution = 0.4):

        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        self.adata = adata
        self.curr_adata = adata
        self.new_df = cudf.DataFrame()
        self.tdf = None

        # Values used for re-clustering
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.knn_n_pcs = knn_n_pcs
        self.umap_min_dist = umap_min_dist
        self.umap_spread = umap_spread
        self.leiden_resolution = leiden_resolution

        if re_cluster_callback:
            self.re_cluster_func = re_cluster_callback
        else:
            self.re_cluster_func = self.re_cluster

        self.markers = markers
        self.reset()
        self.app.layout = self.constuct_layout()

        self.app.callback(
            Output("hidden1", "children"),
            [Input("bt_reset", "n_clicks")]) (self.reset_dialog)

        self.app.callback(
            Output("md_directions", "is_open"),
            [Input("bt_open_directions", "n_clicks"),
             Input("bt_close_directions", "n_clicks")],
            [State("md_directions", "is_open")]) (self.toggle_directions_dialog)

        self.app.callback(
            Output('md_export', 'is_open'),
            [Input('bt_export_df', 'n_clicks'),
            Input("bt_close_export", "n_clicks")]) (self.export_current_df)

        self.app.callback(
            [Output('submit_labels', 'value'),
             Output('point_index_cnt', 'children'),
             Output('point_index_labels', 'value')],
            [Input('basic-interactions', 'clickData'),
             Input('basic-interactions', 'selectedData'),
             Input('rerun_clustering', 'n_clicks'),
             Input('rerun_point_index', 'n_clicks')],
            [State("submit_labels", "value"),
             State('point_index_labels', 'value')]) (self.handle_data_selection)

        marker_outputs = [Output('basic-interactions', 'figure')]
        for marker in self.markers:
            marker_outputs.append(Output(marker + '-interactions', 'figure'))

        self.app.callback(
            marker_outputs,
            [Input('rerun_clustering', 'n_clicks'),
             Input('rerun_point_index', 'n_clicks')],
            [State("submit_labels", "value"),
             State('point_index_labels', 'value')]) (self.handle_re_cluster)

    def re_cluster(self, adata_copy):
        #### rerun clusterings
        adata_copy.obsm["X_pca"] = PCA(n_components=self.n_components, output_type="numpy").fit_transform(adata_copy.X)
        sc.pp.neighbors(adata_copy, n_neighbors=self.n_neighbors, n_pcs=self.knn_n_pcs, method='rapids')
        sc.tl.umap(adata_copy, min_dist=self.umap_min_dist, spread=self.umap_spread, method='rapids')
        adata.obs['leiden'] = rapids_scanpy_funcs.leiden(adata, resolution=self.louvain_resolution)
        return adata_copy

    def reset(self):
        self.curr_adata = self.adata
        self.tdf = self.build_tdf(self.curr_adata)
        # self.curr_adata.obs["orig_index"] = self.tdf.index.to_array()
        self.new_df = cudf.DataFrame()

    def build_tdf(self, l_adata):
        #df = cudf.DataFrame.from_gpu_matrix(
        #    l_adata.obsm["X_umap"], columns=["x", "y"]
        #)
        df = cudf.DataFrame(l_adata.obsm["X_umap"], columns=["x", "y"])

        ldf = cudf.Series(l_adata.obs["leiden"].values)
        df["labels"] = ldf.astype('int32')
        for marker in self.markers:
            df[marker] = cudf.Series(l_adata.obs[marker + "_raw"].values)
            df[marker + '_labels'] = df["labels"]

        df['point_index'] = df.index
        df['barcode'] = l_adata.obs_names
        #df["orig_index"] = l_adata.obs['orig_index'].values

        return df

    def constuct_layout(self):

        fig = self.start_graph(self.tdf)

        violins = self.update_violin_plot(self.tdf)

        col_classes = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight'}
        col_class = col_classes[12 / len(violins)]
        divs_violin = []
        for i in range(0, len(violins)):
            divs_violin.append(
                html.Div([dcc.Graph(id= self.markers[i] + '-interactions',
                    figure=violins[i])], className= col_class + ' columns'))


        return html.Div([
            html.Div(className='row', children=[
                html.Div([dcc.Graph(id='basic-interactions', figure=fig),], className='nine columns',
                        style={'verticalAlign': 'text-top',}),
                html.Div([
                    html.Div(className='row', children=[
                        dbc.Button("Directions", id="bt_open_directions"),
                        dbc.Modal([
                                dbc.ModalHeader("Directions"),
                                dbc.ModalBody(
                                    dcc.Markdown("""
                                        The main scatterplot shows the UMAP visualization of single cells.

                                        ### Re-running Clustering and Visualization
                                        #### Reclustering by Clicking on Groups:
                                        1. Click on any point in a Cluster of Interest. The cluster to which that point belongs to will populate the Cluster box.
                                        2. Click **Recluster on Selected Cluster**.
                                        #### Reclustering by entering cluster ID:
                                        1. Manually enter the IDs of the cluster or clusters of interest in the Cluster box. For example, “1” or “1,2,3”
                                        2. Click **Recluster on Selected Cluster**.
                                        #### Reclustering by Selecting Points:
                                        1. Use the **Box Select** or **Lasso Select** tool to select your points of interest. A number of points will populate the inthe Selected Points field .
                                        2. Click **Recluster on Selected Points**.

                                        ### Exporting Data to a DataFrame
                                        After performing re-clustering on selected cells, click "Export to Dataframe".

                                        ### Using the Toolbar
                                        Hover the mouse over the top right corner of the screen to see a toolbar. Hover over each tool to see its name. The tool options from left to right are:
                                        - **Camera:** download a snapshot of the current view as .png
                                        - **Zoom:** Click and drag to select a region of the plot to zoom into
                                        - **Pan:** Click and drag to shift the current view to a different region of the plot
                                        - **Box Select/Lasso Select:** both these tools can be used to select a region on the plot. The selected points are exported under ‘selection data’. See below to export the selected points to a dataframe.
                                        - **Zoom In/Zoom Out:** Zoom in and out centered on the current view.
                                    """),
                                ),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="bt_close_directions", className="ml-auto")
                                ),
                            ], id="md_directions"),
                    ]),

                    html.Div(className='row', children=[
                        dcc.Markdown("""
                            **Click Data**

                            Click on points in the graph."""), ], style={'marginTop': 18,}),

                    html.Div(className='row', children=[
                        dcc.Input(id='submit_labels', type='text', style={'width': '80%',}),]),
                    html.Div(className='row', children=[
                        dbc.Button('Recluster on Selected Clusters', id='rerun_clustering', n_clicks=0)], style={'marginTop': 6,}),

                    html.Div(className='row', children=[
                        dcc.Markdown("""
                            **Selection Data**

                            Choose the lasso or rectangle tool in the graph's menu
                            bar and then select points in the graph.
                        """),], style={'marginTop': 18,}),

                    html.Div(className='row', children=[
                        dcc.Input(id='point_index_labels', type='text', style={'width': '80%',}),]),
                    html.Div(className='row', children=[html.Div(id='point_index_cnt'),]),
                    html.Div(className='row', children=[
                        dbc.Button('Recluster on Selected Points', id='rerun_point_index', n_clicks=0),], style={'marginTop': 6,}),

                    html.Div(className='row', children=[
                        dbc.Button("Export to Dataframe", id="bt_export_df"),
                        dbc.Modal([
                                dbc.ModalHeader("Export"),
                                dbc.ModalBody(
                                    dcc.Markdown("""
                                        Export Successful. Please return to the notebook.
                                    """),
                                ),
                                dbc.ModalFooter(dbc.Button("Close", id="bt_close_export", className="ml-auto")),
                            ], id="md_export"),
                    ], style={'marginTop': 6,}),

                    html.Div(className='row', children=[html.A(dbc.Button('Reload', id='bt_reset'), href='/'),],
                            style={'marginTop': 6,}),

                ], className='three columns', style={'marginTop': 90, 'verticalAlign': 'text-top',}),
            ]),
            html.Div(className='row', children=divs_violin),
            html.Div(id='hidden1', style={'display':'none'})
        ])


    def start_graph(self, df):
        fig = go.Figure(layout = {'colorscale' : {}})
        
        for i in df['labels'].unique().values_host:
            si = str(i)
            query = 'labels == ' + si
            gdf = df.query(query)
            fig.add_trace(
                go.Scattergl({
                'x': gdf['x'].to_array(),
                'y': gdf['y'].to_array(),
                'text': gdf['labels'].to_array(),
                'customdata': gdf['point_index'].to_array(),
                'name': 'Cluster ' + si,
                'mode': 'markers',
                'marker': {'size': 3, 'color': colors[i % len(colors)]}
            }))

        fig.update_layout(
            showlegend=True, clickmode='event', height=main_fig_height, title='UMAP', dragmode='select',
            annotations=[
                dict(x=0.5, y=-0.07, showarrow=False, text='UMAP_1', xref="paper", yref="paper"),
                dict(x=-0.05, y=0.5, showarrow=False, text="UMAP_2", textangle=-90, xref="paper", yref="paper")])
        return fig

    def update_graph(self, df):
        data = []
        labels = df['labels'].unique().values_host
        for i in labels:
            si = str(labels[i])
            query = 'labels == ' + si
            gdf = df.query(query)
            fig = {
                'type':'scattergl',
                'x': gdf['x'].to_array(),
                'y': gdf['y'].to_array(),
                'text': gdf['labels'].to_array(),
                'customdata': gdf['point_index'].to_array(),
                'name': 'Cluster ' + si,
                'mode': 'markers',
                'marker': {'size': 3, 'color': colors[i % len(colors)]}        }
            data.append(fig)
        output = {
                'data':data,
                'layout':{'clickmode': 'event', 'showlegend': True, 'title': 'UMAP', 'dragmode': 'select'}
            }
        return output

    def update_umap_viz(self, df, value):
        df_labels = df['labels'].isin(value)
        filters = df_labels.values

        print(filters)
        adata_copy = self.curr_adata[filters.get()]
        self.curr_adata = adata_copy.copy()
        adata_copy = self.re_cluster_func(adata_copy)
        df = self.build_tdf(adata_copy)
        return df, self.update_graph(df)

    def update_selection(self, df, value):
        umap_df = df['point_index'].isin(value)
        filters = umap_df.values

        adata_copy = self.curr_adata[filters.get()]
        self.curr_adata = adata_copy.copy()
        adata_copy = self.re_cluster_func(adata_copy)
        df = self.build_tdf(adata_copy)
        return df, self.update_graph(df)

    def update_violin_plot(self, df):
        violins = []
        for marker in self.markers:
            violins.append(self.graph_violin(df, marker))
        return violins

    def graph_violin(self, df, marker):

        fig = go.Figure()
        clusters = df['labels'].unique().values_host
        marker_val = marker + '_val'

        df[marker + '_val'] = df[marker].round(1)
        #for i in clusters.values_host:
        for i in clusters:
            si = str(i)

            query = 'labels == ' + si
            gdf = df.query(query)

            y = gdf[marker_val].to_array()
            x = [i] * len(y)
            fig.add_trace(
                go.Violin({
                    'x': cp.asnumpy(x),
                    'y': cp.asnumpy(y),
                    'text': clusters.tolist(),
                    'name': 'Cluster ' + si
                }))

        fig.update_layout(
            showlegend=True, clickmode='event', title=marker,
            annotations=[
                dict(x=0.5, y=-0.15, showarrow=False, text='Clusters', xref="paper", yref="paper"),
                dict(x=-0.11, y=0.5, showarrow=False, text="Gene values", textangle=-90, xref="paper", yref="paper")])
        return fig

    def start(self, host, port=5000):
        self.reset()

        return self.app.run_server(
            debug=True, use_reloader=False, host=host, port=port)

    def reset_dialog(self, n_clicks):
        if not dash.callback_context.triggered:
            raise dash.exceptions.PreventUpdate
        self.reset()
        return ''

    def toggle_directions_dialog(self, n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

    def export_current_df(self, export_clicks, export_close):
        if not dash.callback_context.triggered:
            raise dash.exceptions.PreventUpdate

        button_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'bt_close_export':
            if export_close:
                func = request.environ.get('werkzeug.server.shutdown')
                if func is None:
                    raise RuntimeError('Not running with the Werkzeug Server')
                func()
                return False
        elif button_id == 'bt_export_df':
            self.new_df = self.tdf
            return True

    def handle_data_selection(self, clicked_cluster, selected_point_index, cluster_clicks, point_index_clicks,
                            selected_clusters, point_index_labels):
        if not dash.callback_context.triggered:
            raise dash.exceptions.PreventUpdate

        comp_id, event_type = dash.callback_context.triggered[0]['prop_id'].split('.')

        submit_labels = ''
        point_cnt_str = ''
        point_indexes = ''

        if comp_id == 'basic-interactions' and event_type == 'clickData':
            # Event - On selecting cluster on the main scatter plot
            if not selected_clusters:
                selected_labels = []
            else:
                selected_labels = list(map(int, selected_clusters.split(",")))

            points = clicked_cluster['points']
            for point in points:
                selected_label = point['text']
                if selected_label in selected_labels:
                    selected_labels.remove(selected_label)
                else:
                    selected_labels.append(selected_label)
            submit_labels = ','.join(map(str, selected_labels))

        elif comp_id == 'basic-interactions' and event_type == 'selectedData':
            # Event - On selection on the main scatterplot
            if not selected_point_index:
                raise dash.exceptions.PreventUpdate

            selected_point_indexes = []
            for point in selected_point_index['points']:
                selected_point_indexes.append(point['customdata'])

            if len(selected_point_indexes) <= 1:
                raise dash.exceptions.PreventUpdate

            point_cnt_str = str(len(selected_point_indexes)) + ' points selected'
            point_indexes = ', '.join(map(str, selected_point_indexes))

        elif comp_id == 'rerun_clustering' and event_type == 'n_clicks':
            pass # required to make sure submit_labels is reset
        elif comp_id == 'rerun_point_index' and event_type == 'n_clicks':
            pass # required to make sure point_indexs is reset
        else:
            raise dash.exceptions.PreventUpdate

        return submit_labels, point_cnt_str, point_indexes

    def handle_re_cluster(self, rerun_clustering, rerun_point_index, selected_clusters, point_index_labels):
        if not dash.callback_context.triggered:
            raise dash.exceptions.PreventUpdate

        comp_id, event_type = dash.callback_context.triggered[0]['prop_id'].split('.')

        if comp_id == 'rerun_clustering' and event_type == 'n_clicks':
            if not selected_clusters:
                raise dash.exceptions.PreventUpdate

            clusters = selected_clusters.split(",")
            if len(clusters) >= 1:
                clusters = list(map(int, clusters))
                (self.tdf, figure) = self.update_umap_viz(self.tdf, clusters)
                violins = self.update_violin_plot(self.tdf)

        elif comp_id == 'rerun_point_index' and event_type == 'n_clicks':
            if not point_index_labels:
                raise dash.exceptions.PreventUpdate
            # Event - On click 'recluster' buttom
            selected_point_indexes = list(map(int, point_index_labels.split(",")))
            (self.tdf, figure) = self.update_selection(self.tdf, selected_point_indexes)
            violins = self.update_violin_plot(self.tdf)

        else:
            raise dash.exceptions.PreventUpdate

        return tuple([figure] + violins)