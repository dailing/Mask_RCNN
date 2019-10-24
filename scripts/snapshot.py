from sqlitedict import SqliteDict
from fire import Fire


def save_snapshot(snapshot, key, output):
    dd = SqliteDict(snapshot)
    payload = dd[key]
    with open(output, 'wb') as f:
        f.write(payload)


def list_snapshot(snapshot):
    dd = SqliteDict(snapshot)
    keys = list(dd.keys())
    print(keys)
    return


if __name__ == "__main__":
    Fire(dict(
        save=save_snapshot,
        list=list_snapshot,
    ))