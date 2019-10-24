
from flask import Flask, request, make_response, redirect
from flask_restful import Resource, Api

from util.logs import get_logger
import cv2
import numpy as np
import json
import hashlib
from peewee import PostgresqlDatabase, Model, TextField,\
	BlobField, DateTimeField, IntegerField, IntegrityError,\
	DatabaseProxy
import playhouse.db_url
from playhouse.postgres_ext import JSONField
import datetime
import psycopg2
import math
import base64
import uuid


psql_db = DatabaseProxy()


class BaseModel(Model):
	"""A base model that will use our Postgresql database"""
	class Meta:
		database = psql_db


class ImageStorage(BaseModel):
	md5 = TextField(unique=True)
	payload = BlobField()
	timestamp = DateTimeField(default=datetime.datetime.now)
	session_name = TextField()
	class Meta:
		indexes = (
            # create a unique on from/to/date
            (('md5', 'session_name'), True),
        )


class ImageAnnotation(BaseModel):
	timestamp = DateTimeField(default=datetime.datetime.now)
	points = JSONField(null=True)
	image_id = IntegerField()
	class_id = IntegerField(default=0)
	session_name = TextField()

class Session(BaseModel):
	session_name = TextField(
		default=lambda : uuid.uuid4().hex,
		unique=True)
	timestamp = DateTimeField(default=datetime.datetime.now)


try:
	psql_db.initialize(playhouse.db_url.connect(
	'postgresql://db_user:123456@db:5432/fuckdb'))
	psql_db.connect()
	psql_db.create_tables([ImageStorage, ImageAnnotation, Session])
except Exception as e:
	print(e)

logger = get_logger('interactive learning')

def image2base64(f):
	# check if there are any image, parse image to base64
	def wrapper(*args, **kwargs):
		result = f(*args, **kwargs)
		if type(result) is not dict:
			if type(result) is np.ndarray:
				result = {'data': result}
			else:
				logger.warn(f'cannot parse {type(result)} type.')
		for k, v in result.items():
			if type(v) is np.ndarray:
				if np.issubdtype(v.dtype, np.floating):
					v = (v * 255).astype(np.uint8)
				succ, code = cv2.imencode('.jpg', v)
				if not succ:
					return None
				code = code.tobytes()
				code = base64.b64encode(code)
				code = code.decode('ascii')
				result[k] = code
			else:
				logger.warn(f'cannot parse type: {type(v)}')
		return result

	# make this decorator work with FLASK
	wrapper.__name__ = f.__name__
	return wrapper


def any2jsonResp(f):
	def wrapper(*args, **kwargs):
		result = f(*args, **kwargs) 
		ffres = json.dumps(result)
		resp = make_response(ffres)
		resp.headers['content-type'] = 'application/json'
		return resp

	wrapper.__name__ = f.__name__
	return wrapper


def requestJson2args(f):
	# parse raw string or bytes from requests to json_obj and
	# and then call the func with fucking arguments
	def wrapper(*args, **kwargs):
		jsonObj = request.get_data()
		if type(jsonObj) in (str, bytes):
			jsonObj = json.loads(jsonObj)
			for k, v in jsonObj.items():
				kwargs[k] = v
		result = f(*args, **kwargs)
		return result

	wrapper.__name__ = f.__name__
	return wrapper


def requestForm2args(f):
	# parse form data to dict, and call func with the fucking arguments
	def wrapper(*args, **kwargs):
		for k, v in request.form.items():
			kwargs[k] = v
		return f(*args, **kwargs)

	wrapper.__name__ = f.__name__
	return wrapper


def requestFile2imageArgs(f):
	def wrapper(*args, **kwargs):
		logger.info('CALL ME')
		logger.info(request.files)
		logger.info(list(request.files.keys()))
		for k, v in request.files.items():
			logger.info(f'processing image file {k}')
			content = v.read()
			img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_ANYCOLOR)
			if img is None:
				logger.warn(f'error when parsing image file:{k}')
			kwargs[k] = img
		return f(*args, **kwargs)

	wrapper.__name__ = f.__name__
	return wrapper


app = Flask(__name__, static_folder='statics', static_url_path='/static')
api = Api(app)


@app.route('/')
def serve_index():
	return redirect('/static/index.html')


@app.route('/api/add_img', methods=['POST'])
def serve_add_img():
	session_name = request.form['session_name']
	for fname, file in request.files.items():
		xx = file.read()
		md5_val = hashlib.md5(xx).hexdigest()
		app.logger.info(len(xx))
		try:
			result = ImageStorage.create(
				payload=xx, md5=md5_val, session_name=session_name)
			app.logger.info(result)
		except IntegrityError as e:
			psql_db.rollback()
			app.logger.info(e)
	return "OK"


@app.route('/api/image_length')
def serve_imageLength():
	length = ImageStorage.select().count()
	app.logger.info(length)
	return dict(length=length)


@app.route('/api/image_list/<string:session_name>/<int:page>/<int:items_per_page>')
def image_list(session_name=None, page=0, items_per_page=10):
	length = ImageStorage.select().where(ImageStorage.session_name==session_name).count()
	result = ImageStorage.\
		select(ImageStorage.id, ImageStorage.session_name).\
		where(ImageStorage.session_name==session_name).\
		order_by(ImageStorage.id).\
		paginate(page, items_per_page).dicts()
	result = list(result)
	for i in result:
		i['url'] = '/api/get_image_by_id/' + str(i['id'])
	app.logger.info(result)
	return dict(
		result=result,
		num_page=math.ceil(length / items_per_page),
		current_page=page)


@app.route('/api/get_image_by_id/<int:index>', methods=['GET'])
def get_image_by_index(index):
	img = ImageStorage.get_by_id(index)
	payload = img.payload.tobytes()
	resp = make_response(payload, 200)
	resp.headers['Content-Type'] = 'image'
	return resp


@app.route('/api/annotation/<int:img_id>', methods=['GET'])
def api_get_annotation(img_id):
	result = ImageAnnotation.select().\
			where(ImageAnnotation.image_id == img_id).\
			dicts().\
			execute()
	app.logger.info(result)
	return result


@app.route('/api/annotation/<int:img_id>', methods=['POST'])
def api_set_or_add_annotation(img_id):
	reqs = request.json
	if isinstance(req, dict):
		reqs = [reqs]
	for req in reqs:
		logger.info(req)
		if 'id' not in req:
			ImageAnnotation.create(**req)
		else:
			rec = ImageAnnotation.get_by_id(req['id'])
			for k, v in req.items():
				setattr(rec, k, v)
				rec.save()
		app.logger.info()


@app.route('/api/sessions', methods=['GET'])
def api_sessions():
	sessons = Session.select(Session.session_name).dicts()
	sessons = list(sessons)
	app.logger.info(sessons)
	return dict(sessions=sessons)


@app.route('/api/session', methods=['POST'])
def api_session():
	logger.info(request.json)
	retval = Session.create(**request.json)
	logger.info(retval)
	ss = Session.get_by_id(retval)
	logger.info(ss.__data__)
	logger.info(type(ss.__data__))
	return ss.__data__



if __name__ == "__main__":
	pass
