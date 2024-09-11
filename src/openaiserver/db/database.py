from typing import Callable

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm.session import Session, sessionmaker


class Database:

    def __init__(self) -> None:
        self.__SQLALCHEMY_DATABASE_URL: str = "sqlite:///./cache.db"
        self.__engine: Engine = create_engine(
            self.__SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
        )
        self.__SessionFactory: Callable[[], Session] = sessionmaker(
            autocommit=False, autoflush=False, bind=self.__engine
        )

    def getEngine(self) -> Engine:
        return self.__engine

    def getSessionFactory(self) -> Callable[[], Session]:
        return self.__SessionFactory
