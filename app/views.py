from flask_restful import Resource, abort
from app import api, models_dao
from flask_apispec import MethodResource, use_kwargs, doc
from webargs.flaskparser import parser
from .schemas import TrainingSchema, UpdateSchema, PredictionSchema


class Classes(MethodResource, Resource):

    @doc(
        description='This method returns the DAO class attribute that contains model classes that are available to '
                    'train in API.',
        responses={'200': {'description': 'List of available model classes.'}}
    )
    def get(self) -> list:
        return models_dao.available_classes


api.add_resource(Classes, '/api/classes')


class Parameters(MethodResource, Resource):

    @doc(
        description='This method returns the DAO class attribute that contains model parameters that are available to '
                    'train in API.',
        responses={'200': {'description': 'JSON of model classes with corresponding parameters.'}}
    )
    def get(self) -> dict:
        return models_dao.available_params


api.add_resource(Parameters, '/api/parameters')


class MLModels(MethodResource, Resource):

    @doc(
        description='This method returns the DAO class attribute that contains information about trained models.',
        responses={'200': {'description': 'JSON of dictionaries with entries corresponding to trained models.'}}
    )
    def get(self) -> dict:
        return models_dao.models_info

    @doc(
        description='Calls the DAO class method that creates model and train it on provided training set.',
        responses={'200': {'description': 'JSON with information about trained model.'}}
    )
    @use_kwargs(TrainingSchema)
    def post(self, **kwargs) -> dict:
        # TODO Проверить на ошибки. Если есть, сделать try except
        return models_dao.create(**kwargs)


api.add_resource(MLModels, '/api/ml_models')


class MLModelsID(MethodResource, Resource):

    @doc(
        description='Calls the DAO class method that retrains the existing model ``id`` on provided training set.',
        params = {'id': {'description': 'The model ID'}},
        responses={'200': {'description': 'JSON with information about retrained model.'}}
    )
    @use_kwargs(UpdateSchema)
    def put(self, id: int, **kwargs) -> dict:
        try:
            return models_dao.update(id, **kwargs)
        except KeyError as e:  # TODO Проверить на другие ошибки. Если есть, сделать try except
            abort(400, meta=eval(str(e)))

    @doc(
        description='Calls the DAO class method that deletes the existing model ``id``.',
        params={'id': {'description': 'The model ID'}}
    )
    def delete(self, id: int) -> tuple:
        try:
            models_dao.delete(id)
            return '', 204
        except KeyError as e:  # TODO Проверить на другие ошибки. Если есть, сделать try except
            abort(400, meta=eval(str(e)))


api.add_resource(MLModelsID, '/api/ml_models/<int:id>')


class MLModelsPred(MethodResource, Resource):


    @doc(
        description='Calls the DAO class method that makes the prediction with existing model ``id`` from provided '
                    'features.',
        params={'id': {'description': 'The model ID'}},
        responses={'200': {'description': 'JSON with model predictions.'}}
    )
    @use_kwargs(PredictionSchema)
    def get(self, id: int, **kwargs) -> dict:
        try:
            return models_dao.predict(id, **kwargs)
        except KeyError as e:  # TODO Проверить на другие ошибки. Если есть, сделать try except
            abort(400, meta=eval(str(e)))


api.add_resource(MLModelsPred, '/api/ml_models/<int:id>/prediction')


@parser.error_handler
def handle_request_parsing_error(err, req, schema, error_status_code, error_headers):
    """webargs error handler that uses Flask-RESTful's abort function to return
    a JSON error response to the client.
    """
    abort(400, meta=err.messages['json'])
