import dash
from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import base64
import io
from PIL import Image
import numpy as np
from model import get_model
from utils import draw_predictions, create_synthetic_image

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize model
model = get_model()

# Create synthetic image
synthetic_image = create_synthetic_image()
synthetic_pil = Image.fromarray(synthetic_image)
synthetic_buffer = io.BytesIO()
synthetic_pil.save(synthetic_buffer, format='PNG')
synthetic_base64 = base64.b64encode(synthetic_buffer.getvalue()).decode('utf-8')

# App layout
app.layout = dbc.Container([
    html.H1("Car Detection Web App", className="mb-4"),

    # 1. Synthetic Car Image
    html.Div([
        html.H2("1. Synthetic Car Image"),
        html.Img(src=f'data:image/png;base64,{synthetic_base64}',
                style={'width': '100%', 'max-width': '800px'})
    ], className="mb-4"),

    # 2. Image Upload
    html.Div([
        html.H2("2. Upload Your Image"),
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select an Image')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='output-image-upload'),
    ], className="mb-4"),

    # 3. Detection Results
    html.Div([
        html.H2("3. Car Detection Result"),
        html.Button("Detect Cars", id="detect-button", className="btn btn-primary mb-3"),
        dbc.Spinner(html.Div(id='detection-result'))
    ])
], fluid=True)

def parse_contents(contents):
    """Parse uploaded image contents"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return Image.open(io.BytesIO(decoded))

@callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents')
)
def update_output(contents):
    if contents is not None:
        return html.Img(src=contents, style={'width': '100%', 'max-width': '800px'})

@callback(
    Output('detection-result', 'children'),
    Input('detect-button', 'n_clicks'),
    State('upload-image', 'contents'),
    prevent_initial_call=True
)
def update_detection(n_clicks, contents):
    if contents is None:
        return "Please upload an image first."

    # Load and process image
    image = parse_contents(contents)
    img_array = np.array(image)

    # Get predictions
    predictions = model(img_array)

    # Draw predictions
    result_image = draw_predictions(img_array, predictions)

    # Convert result to base64
    result_pil = Image.fromarray(result_image)
    result_buffer = io.BytesIO()
    result_pil.save(result_buffer, format='PNG')
    result_base64 = base64.b64encode(result_buffer.getvalue()).decode('utf-8')

    return html.Img(src=f'data:image/png;base64,{result_base64}',
                   style={'width': '100%', 'max-width': '800px'})

if __name__ == '__main__':
    # Start Dash app
    app.run_server(host='0.0.0.0', port=5000, debug=True)