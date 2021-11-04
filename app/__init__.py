from flask import Flask
from flask_restful import Api
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec import APISpec
from flask_apispec import FlaskApiSpec
from .ml_models import MLModelsDAO

app = Flask(__name__)
api = Api(app)

models_dao = MLModelsDAO()

spec = APISpec(
    title='Minimalistic ML API',
    version='v0.4',
    plugins=[MarshmallowPlugin()],
    openapi_version='2.0.0'
    )

from app import views

app.config.update({
    'APISPEC_SPEC': spec,
    'APISPEC_SWAGGER_URL': '/swagger-spec/',
    'APISPEC_SWAGGER_UI_URL': '/api/'
})
docs = FlaskApiSpec(app)
docs.register(views.Classes)
docs.register(views.Parameters)
docs.register(views.MLModels)

ml_model_id_method_view = views.MLModelsID.as_view('MLModelsID')
app.add_url_rule('/api/ml_models/<int:id>', view_func=ml_model_id_method_view)
docs.register(views.MLModelsID)

ml_model_pred_method_view = views.MLModelsPred.as_view('MLModelsPred')
app.add_url_rule('/api/ml_models/<int:id>/prediction', view_func=ml_model_pred_method_view)
docs.register(views.MLModelsPred)