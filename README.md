# API для для машинного обучения

Данный API для машинного обучения основан на классах `sklearn.ensemble.RandomForestClassifier` и `lightgbm.LGBMClassifier`. Для каждого из алгоритмов предусмотрена установка нескольких параметров. Подробные списки доступны в `app/mlmodels/mlmodels.py`. Сам интерфейс прикладного программирования реализован с помощью ряда бибилиотек и фреймворков:
- Flask (основа для API)
- flask-restful (бибилиотека с готовыми формами для REST API, была выбрана из-за совместимости с flask-apispec)
- marshmallow (для валидирования схем)
- webargs (для парсинга запросов)
- apispec (для автоматической документации Swagger из схем marshmallow)
- flask-apispec (для нормальной совместимости предыдущих трёх библиотек)

Папка `app` содержит в себе файлы `__init__.py`, `schemas.py`, `views.py` и папку `ml_models`, которая в свою очередь содержит в себе файлы `__init__.py` и `mlmodels.py`. Flask и API инциализируются в файле `app/__init__.py`, там же собирается спецификация для документации Swagger. Файл `schemas.py` содержит схемы, классы кастомных полей и функции валидаторов для marshmallow. В файле `views.py` представлены ресурсы и методы API. `ml_models/mlmodels.py` содержит класс `MLModelsDAO`, который осуществляет учёт моделей и исполнение операций с ними. Более подробная документация представлена в самих файлах.

Также в репозитории присутствуют технические файлы. `requirements.txt` содержит список зависимостей данного проекта. API запускается из командной строки командой `python cli.py`. В файле `cli.py` содержатся инструкции запуска. Хост и порт по умолчанию `localhost` и 5000 и находятся в файле `config.py`.

Swagger запускается по адресу `<host>/api`, при желании это можно настроить в `app/__init__.py`. Документацию, к сожалению, не получилось автоматически сгенерировать полностью.

Это первая часть ДЗ1, вторая часть - бот - представлена здесь https://github.com/Maxim-Fyodorov/ml_api_bot
