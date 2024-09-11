from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String


class Base(DeclarativeBase):
    pass


class Cache(Base):
    __tablename__: str = "cache"
    prompt: Mapped[str] = mapped_column(String, primary_key=True)
    response: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, primary_key=True)
