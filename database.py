import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.config import get_settings

config = get_settings()

# if config.RDS_DB_NAME:
#     DATABASE_URL = \
#         'postgresql://{username}:{password}@{host}:{port}/{database}'.format(
#             username=config.RDS_USERNAME,
#             password=config.RDS_PASSWORD,
#             host=config.RDS_HOSTNAME,
#             port=config.RDS_PORT,
#             database=config.RDS_DB_NAME,
#         )
# else:
#     DATABASE_URL = \
#         'postgresql://{username}:{password}@{host}:{port}/{database}'.format(
#             username='postgres',
#             password='sirwhite',
#             host='localhost',
#             port='5432',
#             database='enflux-ai-db',
#         )
DATABASE_URL = config.RDS_URL

engine = create_engine(
    DATABASE_URL,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()
