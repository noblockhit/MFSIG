import plotly.graph_objects as go
import pickle

def show(circles=None, lines=None):
    if circles is None:
        circles = [[], [], []]
    
    if lines is None:
        lines = [[], [], []]


    plot_data_x, plot_data_y, plot_data_z = circles
    
    marker_data = go.Scatter3d(
        x=plot_data_x, 
        y=plot_data_y, 
        z=plot_data_z, 
        marker=go.scatter3d.Marker(size=3), 
        opacity=0.8, 
        mode='markers'
    )
    line_data = go.Scatter3d(
        x=lines[0],
        y=lines[1],
        z=lines[2],
        mode='lines',
        name='lines'
    )
    fig=go.Figure(data=[marker_data, line_data])
    fig.write_html('tmp.html', auto_open=True)

if __name__ == "__main__":
    show(**pickle.load(open("tmp.3dgraph", "rb")))
    input()