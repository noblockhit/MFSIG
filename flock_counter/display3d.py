import plotly.graph_objects as go
import pickle

def show(circles=None, lines=None):
    if circles == None:
        raise AttributeError("Cant display 3d graph without any lines.")
    plot_data_x, plot_data_y, plot_data_z = circles
    
    marker_data = go.Scatter3d(
        x=plot_data_x, 
        y=plot_data_y, 
        z=plot_data_z, 
        marker=go.scatter3d.Marker(size=3), 
        opacity=0.8, 
        mode='markers'
    )
    
    fig=go.Figure(data=marker_data)
    fig.write_html('tmp.html', auto_open=True)

if __name__ == "__main__":
    show(**pickle.load(open("tmp.3dgraph", "rb")))
    input()