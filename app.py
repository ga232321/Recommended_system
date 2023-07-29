import hashlib
from datetime import datetime

from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from loguru import logger
from pandas import DataFrame
from pydantic import BaseModel
import os
import pandas as pd
from typing import List, Iterator

from sqlalchemy import create_engine


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


app = FastAPI()

SALT = "burov_start_ml"
ENGINE = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml")


def get_user_group(id: int):
    value_str = str(id) + SALT
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num % 100
    if percent < 50:
        return 'control'
    elif percent < 100:
        return 'test'
    return 'unknown'


def get_model_path(model_version: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально
        MODEL_PATH = f"/workdir/user_input/{model_version}"
    else:

        MODEL_PATH = (
            f"{model_version}"
        )
    return MODEL_PATH


def load_models(model_version: str):
    model_path = get_model_path(model_version)
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def batch_load_sql(query: str) -> pd.DataFrame:
    global ENGINE
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml",
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
        logger.info("chunk")
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features() -> list[DataFrame | Iterator[DataFrame]]:
    # уникальные записи post_id, user_id где был совершен лайк
    logger.info("loading liked posts")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        where action='like'"""
    liked_posts = batch_load_sql(liked_posts_query)

    # Фичи по постам на основе tf-idf
    logger.info("loading posts features")
    posts_features = pd.read_sql("""SELECT * FROM posts_info_features_burov_dl2""",
                                 con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
                                     "postgres.lab.karpov.courses:6432/startml"
                                 )
    posts_features.head().to_csv('post_data_h.csv', index=False)

    # фичи по юзерам
    logger.info("loading user features")
    user_features = pd.read_sql(
        """SELECT * FROM public.user_data""",
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml"
    )

    return [liked_posts, posts_features, user_features]


control_model = load_models('model_control.cbm')
test_model = load_models('model_test.cbm')
print("Models loaded sucsessfully")
features = load_features()
print("Data loaded sucsessfully")


def get_recommended_feed(id: int,
                         time: datetime,
                         limit: int = 10) -> Response:
    user_features = features[2].loc[features[2].user_id == id]
    if user_features is None:
        raise HTTPException(404, 'not found')
    user_features = user_features.drop('user_id', axis=1)
    user_features['hour'] = time.hour
    user_features['month'] = time.month

    post_features = features[1].drop(['index', 'text'], axis=1)
    content = features[1][['post_id', 'text', 'topic']]

    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    #     user_posts_features = post_features.assign(**add_user_features)
    df = pd.DataFrame({'post_id': range(7023)})
    assigned = df.assign(**add_user_features)
    user_posts_features = assigned.assign(**post_features)
    user_posts_features = user_posts_features.set_index('post_id')

    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month
    user_posts_features.head().to_csv('user_post_features.csv', index=False)

    user_group = get_user_group(id=id)
    logger.info(f'user group is "{user_group}"')

    if user_group == 'control':
        model = control_model
    elif user_group == 'test':
        model = test_model
    else:
        raise ValueError("unknown group")

    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts

    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    recommended_posts = filtered_.sort_values('predicts')[-limit:].index

    return Response(recommendations=[
        PostGet(**{
            'id': i,
            'text': content[content.post_id == i].text.values[0],
            'topic': content[content.post_id == i].topic.values[0]
        }) for i in recommended_posts
    ], exp_group=user_group)


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int,
                      time: datetime,
                      limit: int = 10) -> Response:
    return get_recommended_feed(id, time, limit)
