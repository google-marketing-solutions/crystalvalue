# Copyright 2021 Google LLC.
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

"""Prediction server to receive and serve predictions on the uploaded model.
"""

import json
import os
from typing import Any

import joblib
import pandas as pd
import tornado.escape
import tornado.httputil
import tornado.ioloop
import tornado.web

_MODEL_LOCAL_PATH = '/model.joblib'


class HealthCheckHandler(tornado.web.RequestHandler):
  """Request handler for health checks for prediction server."""

  def get(self) -> None:
    """Responds to health checks on GET requests."""
    ready = os.path.exists(_MODEL_LOCAL_PATH)
    if ready:
      self.set_status(200, 'OK')
      self.write('200: OK')
      self.write(json.dumps({'is_healthy': True}))
    else:
      self.set_status(503, 'Service Unavailable')
      self.write('503: Service Unavailable')
      self.write(json.dumps({'is_healthy': False}))
    self.finish()


class PredictionHandler(tornado.web.RequestHandler):
  """Request handler for the predictions requests in our server.

  Attributes:
    model: Model to use to run predictions and return them in the response.
  """

  def __init__(self, application: tornado.web.Application,
               request: tornado.httputil.HTTPServerRequest,
               **kwargs: Any) -> None:
    """Constructor for PredictionHandler.

    Args:
      application: Tornado application to use.
      request: Request to be handled.
      **kwargs: Named extra arguments.
    """
    super().__init__(application, request, **kwargs)
    self.model = joblib.load(_MODEL_LOCAL_PATH)

  def post(self) -> None:
    """Gets the features data and produces predictions to return as response.
    """
    response_body = None
    try:
      req_body = tornado.escape.json_decode(self.request.body)
      instances = pd.DataFrame(req_body.get('instances', {}))
      instances = instances[self.model.column_order]
      predictions = self.model.predict(instances).tolist()
      response_body = json.dumps({'predictions': predictions})
    except Exception as e:
      response_body = json.dumps({'error:': str(e)})

    self.set_header('Content-Type', 'application/json')
    self.set_header('Content-Length', len(response_body))
    self.write(response_body)
    self.finish()


def make_app() -> None:
  tornado_app = tornado.web.Application([('/health_check', HealthCheckHandler),
                                         ('/predict', PredictionHandler)],
                                        debug=False)
  tornado_app.listen(8080)
  tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
  make_app()
