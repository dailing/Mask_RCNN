import click

from main import ImageStorage, psql_db
import playhouse.db_url
import os
from util.logs import get_logger


logger = get_logger('dataset log')


psql_db.initialize(playhouse.db_url.connect('postgresql://db_user:123456@localhost:25068/fuckdb'))

@click.command()
@click.option('--path', help='path of images')
@click.option('--session_name', help='session name')
def add_image(path, session_name):
    for p, _, f in os.walk(path):
        for fname in f:
            fname = os.path.join(p, fname)
            ImageStorage.add_file(fname, session_name)
            logger.info(fname)


if __name__ == "__main__":
    add_image()
