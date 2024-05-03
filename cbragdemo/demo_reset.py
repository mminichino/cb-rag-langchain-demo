##
##

import logging
import argparse
from cbcmgr.cb_operation_s import CBOperation

logger = logging.getLogger('demo_reset')
logger.addHandler(logging.NullHandler())


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-u', '--user', action='store', help="User Name", default="Administrator")
    parser.add_argument('-p', '--password', action='store', help="User Password", default="password")
    parser.add_argument('-h', '--host', action='store', help="Cluster Node Name", default="localhost")
    parser.add_argument('-b', '--bucket', action='store', help="Bucket", default="vectordemos")
    parser.add_argument('-s', '--scope', action='store', help="Scope", default="langchain")
    parser.add_argument('-c', '--collection', action='store', help="Collection", default="webrag")
    options = parser.parse_args()
    return options


def cluster_reset(hostname, username, password, bucket, scope, collection):
    keyspace = f"{bucket}.{scope}.{collection}"
    db = CBOperation(hostname, username, password, ssl=True).connect(keyspace)
    db.flush()


def main():
    options = parse_args()
    cluster_reset(options.host,
                  options.user,
                  options.password,
                  options.bucket,
                  options.scope,
                  options.collection)


if __name__ == '__main__':
    main()
