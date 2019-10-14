import PIL
import pandas as pd
import cv2
import numpy as np
import pickle
from torch.utils.data import Dataset
from main import ImageAnnotation, psql_db
import playhouse.db_url
from util.logs import get_logger

logger = get_logger('fuck sql db')


class SqlDB(Dataset):
    def __init__(
            self,
            db_url='postgresql://db_user:123456@db:5432/fuckdb',
            table_name='imageannotation'):
        psql_db.initialize(playhouse.db_url.connect(
	        'postgresql://db_user:123456@localhost:25068/fuckdb'))
        result = ImageAnnotation.select().\
			dicts().\
			execute()
        logger.info(result[0]['timestamp'])
        logger.info(len(result))


if __name__ == "__main__":
    db = SqlDB()