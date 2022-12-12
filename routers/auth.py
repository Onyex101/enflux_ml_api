import sys
sys.path.append("..")

import models
from jose import jwt, JWTError
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from database import get_db
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from typing import Optional
from pydantic import BaseModel
from fastapi import Depends, HTTPException, status, APIRouter


SECRET_KEY = "gascaxwhwdiwdexwuixdwxcq"
ALGORITHM = "HS256"


class CreateUser(BaseModel):
    username: str
    email: Optional[str]
    first_name: str
    last_name: str
    password: str


bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_bearer = OAuth2PasswordBearer(tokenUrl="token")

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    responses={401: {"user": "Not authorized"}}
)


def get_password_hash(password):
    return bcrypt_context.hash(password)


def verify_password(plain_password, hashed_password):
    return bcrypt_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str, db: Session):
    user = db.query(models.Users).filter(
        models.Users.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(username: str, user_id: str, expires_delta: Optional[timedelta] = None):
    encode = {"sub": username, "id": user_id}
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    encode.update({"exp": expire})
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_bearer)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("id")
        if username is None or user_id is None:
            raise get_user_exceptions()
        return {"username": username, "id": user_id}
    except JWTError():
        raise get_user_exceptions()


@router.post('/create/user')
async def create_new_user(create_user: CreateUser, db: Session = Depends(get_db)):
    create_user_mnodel = models.Users()
    create_user_mnodel.email = create_user.email
    create_user_mnodel.first_name = create_user.first_name
    create_user_mnodel.last_name = create_user.last_name
    create_user_mnodel.username = create_user.username

    hash_password = get_password_hash(create_user.password)

    create_user_mnodel.hashed_password = hash_password
    create_user_mnodel.is_active = True

    db.add(create_user_mnodel)
    db.commit()

    return {
        'status': 201,
        'transaction': 'Succesful'
    }


@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise token_exception()
    token_expires = timedelta(minutes=20)
    token = create_access_token(
        user.username, user.id, expires_delta=token_expires)
    return {"token": token}


# Exceptions
def get_user_exceptions():
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )
    return credentials_exception


def token_exception():
    token_exception_response = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"}
    )
    return token_exception_response
