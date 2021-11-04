from marshmallow import Schema,  post_load
from webargs import fields, validate, ValidationError
from app import models_dao
from typing import Union, Any, Optional, Mapping

lightgbm_objectives_list = [
    'binary',
    'multiclass',
    'multiclassnova',
    'cross_entropy',
    'cross_entropy_lambda'
]

negative_error_message = 'Must be greater than zero'
nan_error_message = 'Not a number'

positive_validator = validate.Range(min = 0, min_inclusive=True, error = negative_error_message)
share_validator = validate.Range(min = 0., max = 1., error = 'Must be in range [0., 1.]')
non_negative_validator = validate.Range(min = 0., error = 'Must be non-negative')

def str2num(value: str, raise_: bool = True) -> Union[int,float,str]:
    """
    Transforms string to integer or float number.

    :param value: Input string.
    :param raise_: Specifies whether the function raise error if it is not possible to transform input string to number.
    :raise ValidationError: If ``raise_`` is set to False.
    :return: Number retrieved from the input string.
    """
    if value.isnumeric():
        return int(value)
    elif value.replace('.', '').isnumeric():
        return float(value)
    else:
        if raise_:
            raise ValidationError(nan_error_message)
        else:
            return value


class IntFloat(fields.Field):
    def _deserialize(
            self,
            value: Any,
            attr: Optional[str],
            data: Optional[Mapping[str, Any]],
            **kwargs
    ) -> Union[int,float]:
        """
        Deserialize value. Overwrites the method of parent class.

        :param value: The value to be deserialized.
        :param attr: The attribute/key in `data` to be deserialized.
        :param data: The raw input data passed to the `Schema.load`.
        :param kwargs: Field-specific keyword arguments.
        :raise ValidationError: In case of formatting or validation failure.
        :return: The deserialized value.
        """
        if isinstance(value, (int, float)):
            return value
        elif isinstance(value, str):
            return str2num(value)
        else:
            raise ValidationError(nan_error_message)


class StrIntFloat(fields.Field):
    def _deserialize(
            self,
            value: Any,
            attr: Optional[str],
            data: Optional[Mapping[str, Any]],
            **kwargs
    ) -> Union[int,float,str]:
        """
        Deserialize value. Overwrites the method of parent class.

        :param value: The value to be deserialized.
        :param attr: The attribute/key in `data` to be deserialized.
        :param data: The raw input data passed to the `Schema.load`.
        :param kwargs: Field-specific keyword arguments.
        :raise ValidationError: In case of formatting or validation failure.
        :return: The deserialized value.
        """
        if isinstance(value, (int, float)):
            return value
        elif isinstance(value, str):
            return str2num(value, False)
        else:
            raise ValidationError('Not string, integer or float')


class StrInt(fields.Field):
    def _deserialize(
            self,
            value: Any,
            attr: Optional[str],
            data: Optional[Mapping[str, Any]],
            **kwargs
    ) -> Union[int,float,str]:
        """
        Deserialize value. Overwrites the method of parent class.

        :param value: The value to be deserialized.
        :param attr: The attribute/key in `data` to be deserialized.
        :param data: The raw input data passed to the `Schema.load`.
        :param kwargs: Field-specific keyword arguments.
        :raise ValidationError: In case of formatting or validation failure.
        :return: The deserialized value.
        """
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            if value.isnumeric():
                return int(value)
            else:
                return value
        else:
            raise ValidationError('Not string or integer')


def validate_int_float(value: Union[int,float]) -> None:
    """
    Validator which succeeds if ``value`` is a positive integer or float in range [0., 1.].

    :param value: The value to validate.
    :raise ValidationError: If required conditions are not satisfied.
    """
    if value < 0:
        raise ValidationError(negative_error_message)
    if isinstance(value, float) and value > 1:
        raise ValidationError('Float value must be in range (0., 1.]')

def validate_column_names(value: Union[int,float]):
    if value < 0 and isinstance(value, int):
        positive_validator(value)

def validate_rf_max_features(value: Union[int,float,str]) -> None:
    """
    Validator which succeeds if ``value`` is a positive integer or float in range [0., 1.] or one of
    ['auto', 'sqrt', 'log2'].

    :param value: The value to validate.
    :raise ValidationError: If required conditions are not satisfied.
    """
    if isinstance(value, (int, float)):
        validate_int_float(value)
    else:
        if value not in ['auto', 'sqrt', 'log2']:
            raise ValidationError('String must be auto, sqrt or log2')

class ParametersSchema(Schema):
    # random forest parameters
    n_estimators = fields.Int(validate = positive_validator)
    criterion = fields.Str(
        validate = validate.OneOf(
            ['gini', 'entropy'],
            error = 'Must be gini or entropy'
        )
    )
    max_depth = fields.Int(validate = positive_validator)
    min_samples_split = IntFloat(validate=validate_int_float)
    min_samples_leaf = IntFloat(validate=validate_int_float)
    min_weight_fraction_leaf = fields.Float(validate = share_validator)
    max_features = StrIntFloat(validate=validate_rf_max_features)
    max_leaf_nodes = fields.Int(validate = positive_validator)
    min_impurity_decrease = fields.Float(validate = share_validator)
    bootstrap = fields.Bool()
    max_samples = IntFloat(validate=validate_int_float)

    # lgbm parameters
    boosting_type = fields.String(
        validate = validate.OneOf(
            ['gbdt', 'dart', 'goss', 'rf'],
            error = 'Must be gbdt, dart, goss or rf'
        )
    )
    num_leaves = fields.Int(validate = positive_validator)
    learning_rate = fields.Float(validate = positive_validator)
    num_iterations = fields.Int(validate = positive_validator)
    subsample_for_bin = fields.Int(validate = positive_validator)
    objective = fields.String(
        validate = validate.OneOf(
            lightgbm_objectives_list,
            error = 'Must be one of the following: ' \
            + ', '.join(lightgbm_objectives_list)
        )
    )
    min_split_gain = fields.Float(validate=non_negative_validator)
    min_child_samples = fields.Int(validate = positive_validator)
    subsample = fields.Float(validate = share_validator)
    subsample_freq = fields.Int()
    colsample_bytree = fields.Float(validate = share_validator)
    reg_alpha = fields.Float(validate = non_negative_validator)
    reg_lambda = fields.Float(validate = non_negative_validator)


class PredictionSchema(Schema):
    X = fields.Dict(keys=StrInt(validate=validate_column_names),
                    values=fields.Dict(
                        keys=fields.Int(validate=positive_validator),
                        value=IntFloat
                    ),
        required=True,
        error_messages={"required": "X is required."}
    )

class UpdateSchema(PredictionSchema):
    y = fields.Dict(
        keys=fields.Int(validate=positive_validator),
        values=IntFloat,
        required=True,
        error_messages={"required": "y is required."}
    )


class TrainingSchema(UpdateSchema):
    class_ = fields.Str(
        required=True,
        error_messages={"required": "Model class is required."},
        data_key='class',
        validate=validate.OneOf(
            models_dao.available_classes,
            error='Must be one of available model classes'
        )
    )
    params = fields.Nested(ParametersSchema, missing = {})

    @post_load
    def check_params(self, data: dict, **kwargs) -> dict:
        """
        Checks whether input parameters are available for requested model class. Throws ValidationError if there are
        invalid parameters in the imput.

        :param data: Dictionary with parsed input request.
        :raise ValidationError: If the required condition is not satisfied.
        :return: Dictionary with processed parsed input request.
        """
        unavailable_params = set(data['params'].keys()).difference(models_dao.available_params[data['class_']])
        if len(unavailable_params) > 0:
            raise ValidationError("Params '" + "', '".join(unavailable_params) + f"' are not available for model {data['class_']}")
        return data