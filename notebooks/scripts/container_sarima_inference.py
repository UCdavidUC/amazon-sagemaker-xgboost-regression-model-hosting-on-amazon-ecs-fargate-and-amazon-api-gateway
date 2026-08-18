"""
Copyright 2021 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
SPDX-License-Identifier: MIT-0

SARIMA Inference Server
-----------------------
Flask-based inference endpoint for SARIMA/SARIMAX models trained with
statsmodels. The model pickle file contains a fitted SARIMAXResultsWrapper
object that supports .forecast() and .get_forecast() methods.

Endpoints:
  POST /          - Generate forecast
  GET /healthcheck - Health check for ECS/ALB

Request format (POST /):
{
  "response_content_type": "application/json" | "text/plain",
  "steps": 5,                          # Number of periods to forecast (required)
  "exog": [[1.2, 0.5], [1.3, 0.6]]    # Exogenous variables for SARIMAX (optional)
}

Response (application/json):
{
  "forecast": [1.23, 1.45, 1.67, 1.89, 2.01],
  "steps": 5
}
"""
import flask
from flask import Flask, request
import json
from logging.config import dictConfig
import numpy as np
import os
import pickle


dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': os.environ.get('FLASK_SERVER_LOG_LEVEL', 'INFO'),
        'handlers': ['wsgi']
    }
})


# Initialize the Flask app
app = Flask(__name__)


# Load the SARIMA model from the pickle file
model_pickle_file_path = os.environ.get('MODEL_PICKLE_FILE_PATH', '/app/model.pkl')
app.logger.info('Loading SARIMA model from \'{}\'...'.format(model_pickle_file_path))
with open(model_pickle_file_path, 'rb') as f:
    model = pickle.load(f)
app.logger.info('SARIMA model loaded successfully.')
app.logger.info('Model type: {}'.format(type(model).__name__))


def forecast(steps, exog=None):
    """Generate forecast for the specified number of steps."""
    app.logger.debug('Generating forecast for {} steps...'.format(steps))
    response_status_code = '200'
    try:
        if steps < 1 or steps > 365:
            response_status_code = '400'
            response_content_body = {'error': 'steps must be between 1 and 365'}
            return response_status_code, response_content_body

        # Use get_forecast for prediction intervals, or forecast for point estimates
        exog_array = None
        if exog is not None:
            exog_array = np.array(exog)
            if exog_array.ndim == 1:
                exog_array = exog_array.reshape(-1, 1)
            if len(exog_array) != steps:
                response_status_code = '400'
                response_content_body = {
                    'error': 'exog length ({}) must match steps ({})'.format(
                        len(exog_array), steps)
                }
                return response_status_code, response_content_body

        forecast_result = model.get_forecast(steps=steps, exog=exog_array)
        predicted_mean = forecast_result.predicted_mean.tolist()
        conf_int = forecast_result.conf_int()

        response_content_body = {
            'forecast': predicted_mean,
            'confidence_interval_lower': conf_int.iloc[:, 0].tolist(),
            'confidence_interval_upper': conf_int.iloc[:, 1].tolist(),
            'steps': steps
        }
        app.logger.info('Forecast generated: {} steps, first={:.4f}, last={:.4f}'.format(
            steps, predicted_mean[0], predicted_mean[-1]))

    except ValueError as e:
        response_status_code = '400'
        response_content_body = {'error': 'Invalid input: {}'.format(str(e))}
        app.logger.warning('ValueError during forecast: {}'.format(str(e)))
    except Exception as e:
        response_status_code = '500'
        response_content_body = {'error': 'Forecast failed: {}'.format(str(e))}
        app.logger.error('Unexpected error during forecast: {}'.format(str(e)))

    app.logger.debug('Completed forecast.')
    return response_status_code, response_content_body


def parse_request_data(request):
    """Parse the incoming JSON request."""
    app.logger.debug('Parsing request data...')
    request_data = json.loads(request.get_data(as_text=True))

    # Response content type
    response_content_type = request_data.get('response_content_type', 'application/json')

    # Number of forecast steps (required)
    steps = request_data.get('steps')
    if steps is None:
        raise ValueError("'steps' is required in the request body")
    steps = int(steps)

    # Exogenous variables (optional, for SARIMAX)
    exog = request_data.get('exog', None)

    app.logger.info('Request: steps={}, exog={}, response_type={}'.format(
        steps, 'provided' if exog else 'none', response_content_type))
    return response_content_type, steps, exog


def format_response_data(response_status_code, response_content_type, response_content_body):
    """Format the response based on content type."""
    app.logger.debug('Formatting response...')
    if response_content_type == 'text/plain':
        if response_status_code == '200':
            # Return forecast values as comma-separated string
            body_str = ','.join('{:.6f}'.format(v) for v in response_content_body['forecast'])
        else:
            body_str = response_content_body.get('error', 'Unknown error')
        response = flask.make_response(body_str)
        response.headers['Content-Type'] = 'text/plain'
    else:
        response = flask.make_response(json.dumps(response_content_body))
        response.headers['Content-Type'] = 'application/json'

    response.status = response_status_code
    return response


@app.route('/healthcheck', methods=['GET'])
def process_health_check():
    """Health check endpoint for ECS/ALB."""
    response = flask.make_response('OK')
    response.headers['Content-Type'] = 'text/plain'
    response.status = '200'
    return response


@app.route('/', methods=['POST'])
def handler():
    """Main inference endpoint."""
    app.logger.debug('Processing forecast request...')
    try:
        response_content_type, steps, exog = parse_request_data(request)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        response = flask.make_response(json.dumps({'error': str(e)}))
        response.headers['Content-Type'] = 'application/json'
        response.status = '400'
        return response

    response_status_code, response_content_body = forecast(steps, exog)
    response = format_response_data(response_status_code, response_content_type, response_content_body)
    return response


if __name__ == '__main__':
    app.logger.info('Starting SARIMA Inference Server...')
    app.run(host=os.environ.get('FLASK_SERVER_HOSTNAME', '0.0.0.0'),
            port=int(os.environ.get('FLASK_SERVER_PORT', '80')),
            debug=bool(os.environ.get('FLASK_SERVER_DEBUG', 'False') == 'True'))
