##
##

import logging
import argparse
from cbcmgr.cb_operation_s import CBOperation
from cbcmgr.cb_capella import Capella
from cbcmgr.cb_bucket import Bucket

logger = logging.getLogger('cluster_prep')
logger.addHandler(logging.NullHandler())


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-u', '--user', action='store', help="User Name", default="Administrator")
    parser.add_argument('-p', '--password', action='store', help="User Password", default="password")
    parser.add_argument('-h', '--host', action='store', help="Cluster Node Name", default="localhost")
    parser.add_argument('-b', '--bucket', action='store', help="Bucket", default="vectordemos")
    parser.add_argument('-s', '--scope', action='store', help="Scope", default="langchain")
    parser.add_argument('-c', '--collection', action='store', help="Collection", default="webrag")
    parser.add_argument('-i', '--index', action='store', help="Index Name", default="webrag_index")
    parser.add_argument('-P', '--project', action='store', help="Project Name")
    parser.add_argument('-D', '--database', action='store', help="Capella Database")
    parser.add_argument('-R', '--profile', action='store', help="Capella API Profile", default="default")
    options = parser.parse_args()
    return options


def create_bucket(profile: str, project: str, database: str, name: str, quota: int, replicas: int = 1):
    project_config = Capella(profile=profile).get_project(project)
    project_id = project_config.get('id')

    bucket = Bucket.from_dict(dict(
        name=name,
        ram_quota_mb=quota,
        num_replicas=replicas,
        max_ttl=0
    ))

    cluster = Capella(project_id=project_id, profile=profile).get_cluster(database)
    cluster_id = cluster.get('id')
    Capella(project_id=project_id, profile=profile).add_bucket(cluster_id, bucket)


def cluster_prep(hostname, username, password, bucket, scope, collection, index_name, quota=256, replicas=1, profile="default", project=None, database=None):
    if project and database:
        create_bucket(profile, project, database, bucket, quota, replicas)

    keyspace = f"{bucket}.{scope}.{collection}"
    db = CBOperation(hostname, username, password, ssl=True, quota=quota, create=True, replicas=replicas).connect(keyspace)
    db.vector_index(index_name, ["embedding"], [1536], text_field="text")


def main():
    options = parse_args()
    cluster_prep(options.host,
                 options.user,
                 options.password,
                 options.bucket,
                 options.scope,
                 options.collection,
                 options.index,
                 256,
                 1,
                 options.profile,
                 options.project,
                 options.database)


if __name__ == '__main__':
    main()
